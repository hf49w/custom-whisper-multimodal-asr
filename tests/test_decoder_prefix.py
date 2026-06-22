"""CPU smoke tests for decoder image soft prompts and legacy fusion."""

import unittest

import torch

from custom_whisper.model import AudioImageWhisper, LoRALinear, ModelDimensions, TextDecoder
from custom_whisper.decoding import PyTorchInference
from custom_whisper.multimodal import AbsEncoderVisual, VisualPromptAdapter
from scripts.visspeech_custom_whisper_utils import (
    forward_multimodal_loss,
    visual_token_loss_weights,
)


class DummyVisualEncoder(AbsEncoderVisual):
    """Small visual encoder used without external model weights."""

    def __init__(self, width: int = 6):
        super().__init__()
        self.width = width
        self.proj = torch.nn.Linear(3, width)

    def output_size(self) -> int:
        return self.width

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        pooled = visual.mean(dim=(-1, -2))
        return self.proj(pooled).unsqueeze(1).repeat(1, 4, 1)

    def prepare_images(self, images):
        return torch.stack(list(images), dim=0)


def tiny_dims() -> ModelDimensions:
    return ModelDimensions(
        n_mels=4,
        n_audio_ctx=4,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=32,
        n_text_ctx=48,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=2,
    )


class DecoderPrefixSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dims = tiny_dims()
        self.tokens = torch.randint(0, self.dims.n_vocab, (2, 5))
        self.audio = torch.randn(2, self.dims.n_audio_ctx, self.dims.n_audio_state)

    def test_decoder_shapes_without_and_with_prefix(self):
        decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.assertEqual(decoder(self.tokens, self.audio).shape, (2, 5, 32))
        for prefix_len in (16, 32):
            prefix = torch.randn(2, prefix_len, self.dims.n_text_state)
            self.assertEqual(
                decoder(self.tokens, self.audio, prefix_embeds=prefix).shape,
                (2, 5, 32),
            )

    def test_blank_prefix_forward_needs_no_image(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder="none",
            feature_fuser="select_speech",
            fusion_location="decoder_prefix",
            decoder_prompt_adapter="blank_prefix",
            decoder_prompt_len=16,
            freeze_whisper=True,
            freeze_visual_encoder=True,
        )
        mel = torch.randn(2, self.dims.n_mels, self.dims.n_audio_ctx * 2)
        logits = model(mel, self.tokens)
        self.assertEqual(logits.shape, (2, 5, self.dims.n_vocab))
        summary = model.trainable_parameter_summary()
        self.assertGreater(summary["adapter_params"], 0)
        self.assertTrue(model.visual_prompt_adapter.prefix.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))

    def test_after_special_tokens_uses_explicit_prefix_length(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder="none",
            feature_fuser="select_speech",
            fusion_location="decoder_prefix",
            decoder_prompt_adapter="blank_prefix",
            decoder_prompt_insert="after_special_tokens",
            decoder_prompt_special_tokens=4,
        )
        self.assertEqual(model.decoder_prefix_insert_pos(self.tokens), 4)
        model.set_decoder_prompt_special_token_count(3)
        self.assertEqual(model.decoder_prefix_insert_pos(self.tokens), 3)

    def test_zero_and_trained_blank_prefix_are_distinct(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder="none",
            feature_fuser="select_speech",
            fusion_location="decoder_prefix",
            decoder_prompt_adapter="blank_prefix",
            decoder_prompt_len=3,
        )
        with torch.no_grad():
            model.visual_prompt_adapter.prefix.fill_(1.0)
        with model.use_decoder_prefix_override("zero"):
            self.assertTrue(torch.equal(model.get_decoder_prefix(2), torch.zeros(2, 3, 8)))
        with model.use_decoder_prefix_override("trained_blank"):
            self.assertTrue(torch.equal(model.get_decoder_prefix(2), torch.ones(2, 3, 8)))

    def test_resampler_accepts_mock_sequence_features(self):
        adapter = VisualPromptAdapter(6, self.dims.n_text_state, 16, 2, 0.0)
        output = adapter(torch.randn(2, 4, 6))
        self.assertEqual(output.shape, (2, 16, self.dims.n_text_state))

    def test_kv_cache_does_not_repeat_prefix(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder="none",
            feature_fuser="select_speech",
            fusion_location="decoder_prefix",
            decoder_prompt_adapter="blank_prefix",
            decoder_prompt_len=3,
        )
        cache, hooks = model.install_kv_cache_hooks()
        try:
            first_tokens = self.tokens[:, :2]
            prefix = model.get_decoder_prefix(2)
            model.decoder(first_tokens, self.audio, cache, prefix_embeds=prefix)
            key = model.decoder.blocks[0].attn.key
            self.assertEqual(cache[key].shape[1], 5)
            model.decoder(
                self.tokens[:, 2:3], self.audio, cache, prefix_embeds=prefix
            )
            self.assertEqual(cache[key].shape[1], 6)
        finally:
            for hook in hooks:
                hook.remove()

    def test_inference_injects_prefix_on_first_step_only(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder="none",
            feature_fuser="select_speech",
            fusion_location="decoder_prefix",
            decoder_prompt_adapter="blank_prefix",
            decoder_prompt_len=3,
        )
        inference = PyTorchInference(model, initial_token_length=2)
        try:
            inference.logits(self.tokens[:, :2], self.audio)
            key = model.decoder.blocks[0].attn.key
            self.assertEqual(inference.kv_cache[key].shape[1], 5)
            inference.logits(self.tokens[:, :3], self.audio)
            self.assertEqual(inference.kv_cache[key].shape[1], 6)
        finally:
            inference.cleanup_caching()

    def test_lora_only_update_is_trainable(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder="none",
            feature_fuser="select_speech",
            fusion_location="decoder_prefix",
            decoder_prompt_adapter="blank_prefix",
            enable_decoder_lora=True,
            lora_rank=2,
            lora_last_n_layers=1,
            freeze_whisper=True,
            freeze_visual_encoder=True,
        )
        lora_modules = [m for m in model.decoder.modules() if isinstance(m, LoRALinear)]
        self.assertGreater(len(lora_modules), 0)
        self.assertTrue(all(not m.weight.requires_grad for m in lora_modules))
        self.assertTrue(all(m.lora_A.requires_grad and m.lora_B.requires_grad for m in lora_modules))

    def test_rank_loss_with_shuffled_visual_features(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder=DummyVisualEncoder(),
            feature_fuser="select_speech",
            fusion_location="decoder_prefix",
            decoder_prompt_adapter="resampler",
            decoder_prompt_len=3,
            decoder_prompt_heads=2,
            decoder_prompt_dropout=0.0,
            freeze_whisper=True,
            freeze_visual_encoder=True,
        )
        batch = {
            "mel": torch.randn(2, self.dims.n_mels, self.dims.n_audio_ctx * 2),
            "input_tokens": self.tokens,
            "labels": torch.randint(0, self.dims.n_vocab, self.tokens.shape),
            "image_paths": [torch.randn(3, 2, 2), torch.randn(3, 2, 2)],
        }
        metrics = forward_multimodal_loss(
            model,
            batch,
            device=torch.device("cpu"),
            loss_rank_shuffle=True,
            loss_rank_weight=0.1,
            loss_rank_margin=0.2,
        )
        metrics["loss"].backward()
        self.assertEqual(
            set(metrics), {"loss", "loss_asr", "loss_rank", "logp_true", "logp_shuf"}
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.visual_prompt_adapter.parameters())
        )

    def test_pos_weighting_requires_explicit_mask(self):
        labels = torch.tensor([[1, 2, -100]])
        with self.assertRaisesRegex(ValueError, "requires a precomputed visual_pos_mask"):
            visual_token_loss_weights(labels, mode="pos", pos_mask=None)
        weights = visual_token_loss_weights(
            labels,
            mode="pos",
            visual_token_weight=1.5,
            pos_mask=torch.tensor([[False, True, False]]),
        )
        self.assertEqual(weights.tolist(), [[1.0, 1.5, 1.0]])

    def test_legacy_encoder_memory_path(self):
        model = AudioImageWhisper(
            self.dims,
            visual_encoder=DummyVisualEncoder(),
            feature_fuser="select_speech",
            fusion_location="encoder_memory",
        )
        mel = torch.randn(2, self.dims.n_mels, self.dims.n_audio_ctx * 2)
        image_features = torch.randn(2, 4, 6)
        logits = model(mel, self.tokens, image_features=image_features)
        self.assertEqual(logits.shape, (2, 5, self.dims.n_vocab))


if __name__ == "__main__":
    unittest.main()
