import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .multimodal import (
    AbsEncoderVisual,
    AbsFeatureFuser,
    BlankDecoderPrefix,
    build_feature_fuser,
    build_visual_encoder,
    build_visual_prompt_adapter,
)
from .transcribe import transcribe as transcribe_function

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class LoRALinear(Linear):
    """A state-dict-compatible linear layer with a trainable low-rank update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        layer: Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        wrapped = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        wrapped.weight = layer.weight
        wrapped.weight.requires_grad = False
        if layer.bias is not None:
            wrapped.bias = layer.bias
            wrapped.bias.requires_grad = False
        return wrapped

    def forward(self, x: Tensor) -> Tensor:
        base = super().forward(x)
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + update.to(base.dtype) * self.scaling


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        kv_cache: Optional[dict] = None,
        prefix_embeds: Optional[Tensor] = None,
        prefix_insert_pos: Optional[int] = None,
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # Keep this branch byte-for-byte equivalent to the original decoder path.
        # It is also used after the first cached step, even if a caller mistakenly
        # keeps supplying prefix_embeds.
        self_cache_key = self.blocks[0].attn.key if len(self.blocks) else None
        has_self_cache = bool(
            kv_cache is not None
            and self_cache_key is not None
            and self_cache_key in kv_cache
        )
        use_prefix = prefix_embeds is not None and not has_self_cache

        if not use_prefix:
            offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
            if offset + x.shape[-1] > self.positional_embedding.shape[0]:
                raise ValueError(
                    f"Decoder context overflow: cached={offset}, tokens={x.shape[-1]}, "
                    f"n_text_ctx={self.positional_embedding.shape[0]}"
                )
            x = (
                self.token_embedding(x)
                + self.positional_embedding[offset : offset + x.shape[-1]]
            )
            token_positions = None
        else:
            if prefix_embeds.dim() != 3:
                raise ValueError(
                    "prefix_embeds must have shape [batch, prefix_len, n_text_state], "
                    f"got {tuple(prefix_embeds.shape)}"
                )
            batch_size, token_length = x.shape
            if prefix_embeds.shape[0] != batch_size:
                raise ValueError(
                    f"Prefix batch size {prefix_embeds.shape[0]} does not match token batch size {batch_size}"
                )
            if prefix_embeds.shape[2] != self.token_embedding.embedding_dim:
                raise ValueError(
                    f"Prefix width {prefix_embeds.shape[2]} does not match n_text_state "
                    f"{self.token_embedding.embedding_dim}"
                )
            prefix_length = prefix_embeds.shape[1]
            total_length = token_length + prefix_length
            if total_length > self.positional_embedding.shape[0]:
                raise ValueError(
                    f"Decoder prefix + token length ({prefix_length} + {token_length} = "
                    f"{total_length}) exceeds n_text_ctx={self.positional_embedding.shape[0]}"
                )
            insert_pos = 0 if prefix_insert_pos is None else int(prefix_insert_pos)
            if not 0 <= insert_pos <= token_length:
                raise ValueError(
                    f"prefix_insert_pos must be in [0, {token_length}], got {insert_pos}"
                )
            token_embeds = self.token_embedding(x)
            prefix_embeds = prefix_embeds.to(
                device=token_embeds.device, dtype=token_embeds.dtype
            )
            x = torch.cat(
                [
                    token_embeds[:, :insert_pos],
                    prefix_embeds,
                    token_embeds[:, insert_pos:],
                ],
                dim=1,
            )
            x = x + self.positional_embedding[:total_length]
            token_positions = torch.cat(
                [
                    torch.arange(insert_pos, device=x.device),
                    torch.arange(
                        insert_pos + prefix_length,
                        total_length,
                        device=x.device,
                    ),
                ]
            )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        if token_positions is not None:
            x = x.index_select(1, token_positions)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def enable_decoder_lora(
        self,
        *,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.05,
        last_n_layers: int = 4,
        targets: str = "self_attn_qv,cross_attn_qv,mlp",
    ) -> int:
        """Attach lightweight LoRA updates to selected final decoder layers."""

        target_set = {item.strip() for item in targets.split(",") if item.strip()}
        valid_targets = {"self_attn_qv", "cross_attn_qv", "mlp"}
        unknown = target_set - valid_targets
        if unknown:
            raise ValueError(f"Unsupported LoRA targets: {sorted(unknown)}")
        first_layer = max(0, len(self.decoder.blocks) - max(1, last_n_layers))
        replaced = 0

        def wrap(parent: nn.Module, name: str) -> None:
            nonlocal replaced
            layer = getattr(parent, name)
            if isinstance(layer, LoRALinear):
                return
            if not isinstance(layer, Linear):
                raise TypeError(f"Expected custom Linear at {name}, got {type(layer)!r}")
            setattr(
                parent,
                name,
                LoRALinear.from_linear(
                    layer, rank=rank, alpha=alpha, dropout=dropout
                ),
            )
            replaced += 1

        for block in list(self.decoder.blocks)[first_layer:]:
            if "self_attn_qv" in target_set:
                wrap(block.attn, "query")
                wrap(block.attn, "value")
            if "cross_attn_qv" in target_set and block.cross_attn is not None:
                wrap(block.cross_attn, "query")
                wrap(block.cross_attn, "value")
            if "mlp" in target_set:
                for index in (0, 2):
                    layer = block.mlp[index]
                    if not isinstance(layer, LoRALinear):
                        block.mlp[index] = LoRALinear.from_linear(
                            layer, rank=rank, alpha=alpha, dropout=dropout
                        )
                        replaced += 1
        return replaced

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function


class AudioImageWhisper(Whisper):
    supports_image_inputs = True

    def __init__(
        self,
        dims: ModelDimensions,
        *,
        visual_encoder: Union[str, AbsEncoderVisual] = "resnet18",
        feature_fuser: Union[str, AbsFeatureFuser] = "concat_proj",
        visual_pretrained: bool = False,
        image_size: int = 224,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_return_sequence: bool = False,
        num_gmlp_layers: int = 1,
        num_resnet_layers: int = 18,
        p_speech: float = 0.5,
        use_residual: bool = True,
        dim_speech_inter: int = 128,
        dim_visual_inter: int = 128,
        use_layer_norm: bool = True,
        attn_num_heads: int = 8,
        attn_dropout: float = 0.1,
        attn_gate_init: float = -4.0,
        attn_num_queries: int = 8,
        fusion_location: str = "encoder_memory",
        decoder_prompt_adapter: str = "none",
        decoder_prompt_len: int = 16,
        decoder_prompt_heads: int = 8,
        decoder_prompt_dropout: float = 0.1,
        decoder_prompt_insert: str = "before_tokens",
        decoder_prompt_special_tokens: Optional[int] = None,
        decoder_prompt_missing: str = "audio_only",
        blip2_model_name: str = "",
        freeze_visual_encoder: bool = False,
        freeze_whisper: bool = False,
        visual_local_files_only: bool = False,
        enable_decoder_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 4,
        lora_targets: str = "self_attn_qv,cross_attn_qv,mlp",
    ):
        super().__init__(dims)
        if fusion_location not in {"encoder_memory", "decoder_prefix"}:
            raise ValueError(f"Unsupported fusion_location: {fusion_location}")
        if decoder_prompt_insert not in {"before_tokens", "after_special_tokens"}:
            raise ValueError(f"Unsupported decoder_prompt_insert: {decoder_prompt_insert}")
        if decoder_prompt_missing not in {"audio_only", "error"}:
            raise ValueError(f"Unsupported decoder_prompt_missing: {decoder_prompt_missing}")
        self.fusion_location = fusion_location
        self.decoder_prompt_adapter_name = decoder_prompt_adapter
        self.decoder_prompt_len = decoder_prompt_len
        self.decoder_prompt_insert = decoder_prompt_insert
        if decoder_prompt_special_tokens is not None and decoder_prompt_special_tokens < 0:
            raise ValueError("decoder_prompt_special_tokens must be non-negative or None")
        self.decoder_prompt_special_tokens = decoder_prompt_special_tokens
        self.decoder_prompt_missing = decoder_prompt_missing
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_whisper = freeze_whisper
        self.partial_init_report: Dict[str, Any] = {}
        self.visual_config: Dict[str, Any] = {
            "visual_encoder": visual_encoder if isinstance(visual_encoder, str) else visual_encoder.__class__.__name__,
            "feature_fuser": feature_fuser if isinstance(feature_fuser, str) else feature_fuser.__class__.__name__,
            "visual_pretrained": visual_pretrained,
            "image_size": image_size,
            "clip_model_name": clip_model_name,
            "clip_return_sequence": clip_return_sequence,
            "num_gmlp_layers": num_gmlp_layers,
            "num_resnet_layers": num_resnet_layers,
            "p_speech": p_speech,
            "use_residual": use_residual,
            "dim_speech_inter": dim_speech_inter,
            "dim_visual_inter": dim_visual_inter,
            "use_layer_norm": use_layer_norm,
            "attn_num_heads": attn_num_heads,
            "attn_dropout": attn_dropout,
            "attn_gate_init": attn_gate_init,
            "attn_num_queries": attn_num_queries,
            "fusion_location": fusion_location,
            "decoder_prompt_adapter": decoder_prompt_adapter,
            "decoder_prompt_len": decoder_prompt_len,
            "decoder_prompt_heads": decoder_prompt_heads,
            "decoder_prompt_dropout": decoder_prompt_dropout,
            "decoder_prompt_insert": decoder_prompt_insert,
            "decoder_prompt_special_tokens": decoder_prompt_special_tokens,
            "decoder_prompt_missing": decoder_prompt_missing,
            "blip2_model_name": blip2_model_name,
            "freeze_visual_encoder": freeze_visual_encoder,
            "freeze_whisper": freeze_whisper,
            "visual_local_files_only": visual_local_files_only,
            "enable_decoder_lora": enable_decoder_lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_last_n_layers": lora_last_n_layers,
            "lora_targets": lora_targets,
        }

        self.encoder_visual = (
            visual_encoder
            if isinstance(visual_encoder, AbsEncoderVisual)
            else build_visual_encoder(
                visual_encoder,
                pretrained=visual_pretrained,
                image_size=image_size,
                clip_model_name=clip_model_name,
                clip_return_sequence=clip_return_sequence,
                local_files_only=visual_local_files_only,
                num_gmlp_layers=num_gmlp_layers,
                num_resnet_layers=num_resnet_layers,
            )
        )
        self.feature_fuser = (
            feature_fuser
            if isinstance(feature_fuser, AbsFeatureFuser)
            else build_feature_fuser(
                feature_fuser,
                dim_speech=self.dims.n_audio_state,
                dim_visual=self.encoder_visual.output_size(),
                p_speech=p_speech,
                use_residual=use_residual,
                dim_speech_inter=dim_speech_inter,
                dim_visual_inter=dim_visual_inter,
                use_layer_norm=use_layer_norm,
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                attn_gate_init=attn_gate_init,
                attn_num_queries=attn_num_queries,
            )
        )
        self.visual_prompt_adapter = build_visual_prompt_adapter(
            decoder_prompt_adapter,
            dim_visual=self.encoder_visual.output_size(),
            dim_text=self.dims.n_text_state,
            num_queries=decoder_prompt_len,
            num_heads=decoder_prompt_heads,
            dropout=decoder_prompt_dropout,
            blip2_model_name=blip2_model_name,
        )
        if enable_decoder_lora:
            self.enable_decoder_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                last_n_layers=lora_last_n_layers,
                targets=lora_targets,
            )
        self._visual_context: Dict[str, Any] = {}
        if freeze_whisper or freeze_visual_encoder:
            self.configure_trainable_parameters(
                freeze_whisper=freeze_whisper,
                freeze_visual_encoder=freeze_visual_encoder,
            )

    def configure_trainable_parameters(
        self,
        *,
        freeze_whisper: bool = True,
        freeze_visual_encoder: bool = True,
    ) -> Dict[str, int]:
        """Freeze base encoders/decoder and expose only the active lightweight modules."""

        self.freeze_whisper = freeze_whisper
        self.freeze_visual_encoder = freeze_visual_encoder
        if freeze_whisper:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.decoder.parameters():
                parameter.requires_grad = False
            for module in self.decoder.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
        if freeze_visual_encoder:
            for parameter in self.encoder_visual.parameters():
                parameter.requires_grad = False
        active_module = (
            self.visual_prompt_adapter
            if self.fusion_location == "decoder_prefix"
            else self.feature_fuser
        )
        if active_module is not None:
            for parameter in active_module.parameters():
                parameter.requires_grad = True
        return self.trainable_parameter_summary()

    def trainable_parameter_summary(self) -> Dict[str, int]:
        """Return total, trainable, adapter, and LoRA parameter counts."""

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        adapter = (
            sum(parameter.numel() for parameter in self.visual_prompt_adapter.parameters())
            if self.visual_prompt_adapter is not None
            else 0
        )
        lora = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if "lora_A" in name or "lora_B" in name
        )
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "adapter_params": adapter,
            "lora_params": lora,
        }

    def _prepare_image_batch(self, image: Any) -> Tensor:
        if torch.is_tensor(image):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.dim() != 4:
                raise ValueError(f"Expected image tensor with 3 or 4 dims, got {tuple(image.shape)}")
            return image

        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            images = [image]
        elif isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]

        batch = self.encoder_visual.prepare_images(images)
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        return batch

    def _expand_visual_features(
        self,
        visual_features: Tensor,
        batch_size: int,
    ) -> Tensor:
        if visual_features.shape[0] == batch_size:
            return visual_features
        if visual_features.shape[0] == 1:
            repeat_shape = [batch_size] + [1] * (visual_features.dim() - 1)
            return visual_features.repeat(*repeat_shape)
        if batch_size % visual_features.shape[0] == 0:
            return visual_features.repeat_interleave(
                batch_size // visual_features.shape[0], dim=0
            )
        raise ValueError(
            f"Visual batch size {visual_features.shape[0]} does not match audio batch size {batch_size}"
        )

    def set_visual_context(
        self,
        *,
        image: Any = None,
        image_features: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if image is None and image_features is None:
            self._visual_context = {}
            return None

        if image_features is None:
            image_features = self.encode_image(image)
        else:
            image_features = image_features.to(self.device)

        self._visual_context = {
            "image_features": image_features.detach(),
        }
        return image_features

    def clear_visual_context(self) -> None:
        self._visual_context = {}

    @contextmanager
    def use_visual_context(
        self,
        *,
        image: Any = None,
        image_features: Optional[Tensor] = None,
    ):
        previous_context = dict(self._visual_context)
        self.set_visual_context(image=image, image_features=image_features)
        try:
            yield self
        finally:
            self._visual_context = previous_context

    def encode_image(self, image: Any) -> Tensor:
        image_batch = self._prepare_image_batch(image).to(self.device)
        if self.freeze_visual_encoder:
            self.encoder_visual.eval()
            with torch.no_grad():
                visual_features = self.encoder_visual(image_batch)
        else:
            visual_features = self.encoder_visual(image_batch)
        return visual_features

    def maybe_expand_prefix(self, prefix: Tensor, batch_size: int) -> Tensor:
        """Expand a prefix for batched decoding and repeat-interleaved beams."""

        if prefix.shape[0] == batch_size:
            return prefix
        if prefix.shape[0] == 1:
            return prefix.expand(batch_size, -1, -1)
        if batch_size % prefix.shape[0] == 0:
            return prefix.repeat_interleave(batch_size // prefix.shape[0], dim=0)
        raise ValueError(
            f"Prefix batch size {prefix.shape[0]} does not match decoding batch size {batch_size}"
        )

    def get_decoder_prefix(
        self,
        batch_size: int,
        image: Any = None,
        image_features: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Build or retrieve image-conditioned decoder prefix embeddings."""

        if self.fusion_location != "decoder_prefix":
            return None
        override = self._visual_context.get("prefix_override")
        if override == "disabled":
            return None
        if override == "zero":
            return torch.zeros(
                batch_size,
                self.decoder_prompt_len,
                self.dims.n_text_state,
                device=self.device,
            )
        if override == "trained_blank":
            if not isinstance(self.visual_prompt_adapter, BlankDecoderPrefix):
                raise ValueError(
                    "trained blank-prefix evaluation requires a checkpoint configured "
                    "with decoder_prompt_adapter='blank_prefix'"
                )
            return self.visual_prompt_adapter(batch_size)
        if self.visual_prompt_adapter is None:
            return None
        if isinstance(self.visual_prompt_adapter, BlankDecoderPrefix):
            return self.visual_prompt_adapter(batch_size)
        visual_features = self._resolve_visual_features(
            image=image,
            image_features=image_features,
            batch_size=(
                image_features.shape[0]
                if image_features is not None
                else batch_size
            ),
        )
        if visual_features is None:
            if self.decoder_prompt_missing == "error":
                raise ValueError(
                    "decoder_prefix mode requires image/image_features; use blank_prefix "
                    "or decoder_prompt_missing='audio_only' for an audio-only fallback"
                )
            return None
        prefix = self.visual_prompt_adapter(visual_features)
        return self.maybe_expand_prefix(prefix, batch_size)

    def set_decoder_prompt_special_token_count(self, count: int) -> None:
        """Set the exact number of leading Whisper special tokens in training inputs."""

        if count < 0:
            raise ValueError("Special-token prefix length must be non-negative")
        self.decoder_prompt_special_tokens = int(count)
        self.visual_config["decoder_prompt_special_tokens"] = int(count)

    def decoder_prefix_insert_pos(
        self,
        tokens: Tensor,
        special_token_count: Optional[int] = None,
    ) -> int:
        """Resolve the configured soft-prompt insertion position."""

        if self.decoder_prompt_insert == "before_tokens":
            return 0
        count = (
            self.decoder_prompt_special_tokens
            if special_token_count is None
            else int(special_token_count)
        )
        if count is None:
            raise ValueError(
                "decoder_prompt_insert='after_special_tokens' requires an explicit "
                "special-token prefix length. Set decoder_prompt_special_tokens or "
                "pass special_token_count from the tokenizer/decode task."
            )
        if count < 0:
            raise ValueError("special_token_count must be non-negative")
        return min(count, tokens.shape[-1])

    @contextmanager
    def use_decoder_prefix_override(self, mode: Optional[str]):
        """Temporarily disable or zero the decoder prefix for diagnostics."""

        if mode not in {None, "disabled", "zero", "trained_blank"}:
            raise ValueError(f"Unsupported prefix override: {mode}")
        previous_context = dict(self._visual_context)
        if mode is not None:
            self._visual_context["prefix_override"] = mode
        try:
            yield self
        finally:
            self._visual_context = previous_context

    def _resolve_visual_features(
        self,
        *,
        image: Any = None,
        image_features: Optional[Tensor] = None,
        batch_size: int,
    ) -> Optional[Tensor]:
        if image_features is None and image is None and self._visual_context:
            image_features = self._visual_context.get("image_features")

        if image_features is None and image is not None:
            image_features = self.encode_image(image)

        if image_features is None:
            return None

        image_features = image_features.to(self.device)
        image_features = self._expand_visual_features(image_features, batch_size=batch_size)
        return image_features

    def fuse_audio_image_features(
        self,
        audio_features: Tensor,
        image_features: Optional[Tensor] = None,
    ) -> Tensor:
        if image_features is None:
            return audio_features
        image_features = image_features.to(dtype=audio_features.dtype, device=audio_features.device)
        fused = self.feature_fuser(audio_features, image_features)
        return fused.to(dtype=audio_features.dtype)

    def embed_audio(
        self,
        mel: Tensor,
        image: Any = None,
        image_features: Optional[Tensor] = None,
    ) -> Tensor:
        audio_features = super().embed_audio(mel)
        if self.fusion_location == "decoder_prefix":
            return audio_features
        visual_features = self._resolve_visual_features(
            image=image,
            image_features=image_features,
            batch_size=audio_features.shape[0],
        )
        return self.fuse_audio_image_features(audio_features, image_features=visual_features)

    def forward(
        self,
        mel: Tensor,
        tokens: Tensor,
        image: Any = None,
        image_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        audio_features = self.embed_audio(mel, image=image, image_features=image_features)
        if self.fusion_location == "decoder_prefix":
            prefix = self.get_decoder_prefix(
                audio_features.shape[0], image=image, image_features=image_features
            )
            return self.decoder(
                tokens,
                audio_features,
                prefix_embeds=prefix,
                prefix_insert_pos=self.decoder_prefix_insert_pos(tokens),
            )
        return self.decoder(tokens, audio_features)

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Return logits with the active visual prefix, including language ID calls."""

        prefix = self.get_decoder_prefix(tokens.shape[0])
        partial_special_count = (
            tokens.shape[-1]
            if (
                self.decoder_prompt_insert == "after_special_tokens"
                and self.decoder_prompt_special_tokens is None
            )
            else None
        )
        return self.decoder(
            tokens,
            audio_features,
            prefix_embeds=prefix,
            prefix_insert_pos=self.decoder_prefix_insert_pos(
                tokens, special_token_count=partial_special_count
            ),
        )

    def transcribe(
        self,
        audio,
        *,
        image: Any = None,
        image_features: Optional[Tensor] = None,
        **kwargs,
    ):
        if image is None and image_features is None:
            return transcribe_function(self, audio, **kwargs)
        with self.use_visual_context(image=image, image_features=image_features):
            return transcribe_function(self, audio, **kwargs)

    def decode(
        self,
        mel: Tensor,
        options,
        *,
        image: Any = None,
        image_features: Optional[Tensor] = None,
        **kwargs,
    ):
        if image is None and image_features is None:
            return decode_function(self, mel, options, **kwargs)
        with self.use_visual_context(image=image, image_features=image_features):
            return decode_function(self, mel, options, **kwargs)

    def detect_language(
        self,
        mel: Tensor,
        tokenizer=None,
        *,
        image: Any = None,
        image_features: Optional[Tensor] = None,
    ):
        if image is None and image_features is None:
            return detect_language_function(self, mel, tokenizer=tokenizer)
        with self.use_visual_context(image=image, image_features=image_features):
            return detect_language_function(self, mel, tokenizer=tokenizer)
