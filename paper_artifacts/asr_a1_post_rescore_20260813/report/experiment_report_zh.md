# 发音评估系统中的 ASR 模块实验报告

## 1. 任务定位

本实验的 ASR 模块用于发音评估系统的前端识别。它的目标不是直接输出发音评分，而是为后续发音评估模块提供更稳定的转写文本、候选转写和置信/重打分信息。实验数据采用 Flickr8K 语音-图像数据，固定划分为训练集 32,000 条、验证集 4,000 条、测试集 4,000 条。

正式汇报只使用验证集选择参数，然后在测试集上评估一次。所有使用 test 调参得到的结果只作为诊断分析，不作为正式结果。

## 2. 最终模型结构

最终系统不是单一端到端重新训练的大模型，而是一个两阶段 ASR 推理 pipeline：

1. A1 blank decoder prefix ASR 生成 beam20 候选。
2. 使用验证集选出的 post-rescore reranker 从 beam20 候选中选择最终输出。

### 2.1 A1 blank decoder prefix ASR

A1 基于 Whisper large-v3，冻结 Whisper 主体，只训练一个 16-token 的 decoder prefix。这个 prefix 不来自图像，因此 A1 本质上是一个学习 Flickr8K 领域偏置的轻量适配器。

核心配置：

- 基座模型：Whisper large-v3
- 融合位置：`decoder_prefix`
- prefix 类型：`blank_prefix`
- prefix 长度：16
- visual encoder：`none`
- Whisper 主体：冻结
- 可训练参数：20,480
- 训练轮数设置：20 epoch，最终选择验证集最优 checkpoint；该 best-val checkpoint 来自 20 epoch 训练过程中的最优轮次。

相关代码：

- `custom_whisper/model.py`：`AudioImageWhisper` 和 decoder prefix 插入逻辑。
- `custom_whisper/multimodal.py`：`BlankDecoderPrefix`。
- `scripts/train_visspeech_custom_whisper_fuser.py`：训练入口和 checkpoint 保存逻辑。
- `scripts/run_flickr8k_decoder_prompt_experiments.sh`：A1/A2/A4/A5/A6 等实验配置。

### 2.2 beam20 + post-rescore reranker

A1 使用 decode-only 路径生成 beam20 n-best candidates。该路径使用 `pad_or_trim + model.decode`，避免和 `model.transcribe()` 路径混用。

最终重打分公式为：

```text
score =
  0.5  * asr_mean_logprob
+ 0.01 * clip_zscore
+ 1.0  * mbr_score
+ 0.5  * A1_logprob
+ 0.05 * A5_logprob
```

验证集选参后，权重为 0 的项已舍弃，包括 A0_logprob、A2_logprob 和 length_score。

各项含义：

- `asr_mean_logprob`：A1 beam decode 自身候选平均 log probability。
- `clip_zscore`：候选文本和图像的 CLIP 相似度，按 sample 内 z-score 标准化。
- `mbr_score`：候选到同一样本其他候选的平均 normalized word edit distance 的相反数，用于偏向候选集合共识。
- `A1_logprob`：A1 对候选文本的 teacher-forcing normalized logprob。
- `A5_logprob`：A5 BLIP2-QFormer decoder-prompt 模型对候选文本的 teacher-forcing normalized logprob。A5 只作为重打分 teacher，不作为最终直接输出模型。

完整推理代码：

- `scripts/run_a1_beam20_post_rescore_infer.sh`
- `scripts/dump_nbest_candidates.py`
- `scripts/rescore_nbest_with_models.py`
- `scripts/rerank_nbest_candidates.py`
- `scripts/a9_candidate_utils.py`

## 3. 正式实验结果

正式最佳结果是 A1 beam20 + validation-tuned post-rescore reranker：

| 系统 | Test WER | Test CER | 说明 |
|---|---:|---:|---|
| A0 no-image Whisper beam20 decode-only top1 | 2.6299% | 1.1449% | 无图 Whisper decode-only baseline |
| A1 beam20 top1 | 2.1173% | 0.8437% | A1 blank prefix，未 post-rescore |
| A9 beam20 MBR/CLIP grid | 2.0759% | 0.8329% | test grid 诊断结果，不作为正式选参依据 |
| A1 beam20 + val-tuned post-rescore | 2.0713% | 0.8338% | 正式最佳结果 |

相对无图 Whisper beam20 decode-only baseline：

- WER：2.6299% → 2.0713%，绝对下降 0.5586 个百分点，相对下降 21.24%。
- CER：1.1449% → 0.8338%，绝对下降 0.3111 个百分点，相对下降 27.17%。

与 CIEASR 的比较需要谨慎。CIEASR 报告的 Flickr8K WER 从 2.42% 降到 2.05%，相对下降约 15.29%。本系统的绝对 WER 2.0713% 略高于 2.05%，不能声称绝对超过 CIEASR；但在“相对各自无图/基线模型的 WER 降低幅度”这个口径上，本系统 21.24% 的相对下降高于 CIEASR 报告的 15.29%。由于训练/测试划分和验证集设置可能不同，这一比较应写成趋势性对比，而不是严格同划分 SOTA 结论。

## 4. 主要模型和消融结果

| 实验 | 配置摘要 | Test WER | Test CER | 结论 |
|---|---|---:|---:|---|
| A0 normal eval | 无图 Whisper baseline | 2.6322% | 1.1219% | 基线 |
| A1 10ep | blank decoder prefix k16 | 2.3633% | 0.9856% | 领域 prefix 有效 |
| A1 20ep | blank decoder prefix k16 | 2.2782% | 0.9402% | 比 10ep 更好 |
| A2 10ep | CLIP sequence decoder prefix k16 | 2.3771% | 0.9566% | 接近 A1，但没有稳定超过 |
| A2 20ep | CLIP sequence decoder prefix k16 | 2.3012% | 0.9805% | 仍弱于 A1 normal eval |
| A3 20ep | CLIP sequence decoder prefix k32 | 2.4483% | 1.0624% | prefix 加长无收益 |
| A4 10ep | CLIP prefix + shuffle ranking loss | 2.4093% | 1.0123% | ranking loss 未带来收益 |
| A5 10ep | BLIP2-QFormer decoder prompt | 2.3725% | 0.9851% | 直接输出弱于 A1 |
| A5 20ep | BLIP2-QFormer decoder prompt | 2.4897% | 1.1130% | 继续训练退化 |
| A6 | A4 init + decoder LoRA | 2.4897% | 1.2105% | LoRA 路线失败 |
| A7 | A4 + CLIP rerank | 2.2253% | 0.8779% | 比 A4 好，但仍弱于最终 A1 post-rescore |

关键发现：A1 blank prefix 是最强的基础模型。这说明 Flickr8K 场景下，学习领域/数据集先验比直接训练视觉 soft prompt 更稳定。

## 5. 候选集诊断

A1 beam20 的 top1 WER 为 2.1173%，但 beam20 oracle WER 为 0.6138%。这说明 beam 中大量样本存在更好的候选，性能瓶颈不在候选生成，而在候选选择。

因此后续实验转向 A9/A10/A11/A12/A13/B1/B2 候选重排序和后处理，而不是继续训练 ASR 主模型。

## 6. 排除的错误路线

### 6.1 直接视觉 soft prompt

A2/A4/A5 等图像 prefix 模型没有稳定超过 A1。进一步做 true image、shuffled image、disable image、zero prefix 对照时，true image 和 shuffled image 接近，说明模型并没有稳定利用图像语义，而更像是在学习另一个领域 prefix。

### 6.2 学习型 reranker

Ridge / learning-based candidate reranker 在验证集或训练子集上可以表现较好，但在 test 上变差，说明特征不足且存在过拟合。正式结果没有采用学习型 reranker。

### 6.3 conservative gated rerank 和 ROVER

A10 conservative gated rerank 的目标是减少“top1 正确但 rerank 改坏”的样本，但验证集选择后没有超过 final post-rescore。A11 ROVER/confusion network 也没有形成稳定收益。

### 6.4 图像语义 caption/tag rerank

A12/A13 使用离线 caption/tag 作为候选重打分特征，但 true/shuffle/disable semantics 三组结果完全一致，且验证集选出的 semantic weights 全为 0。这说明 caption/tag 特征没有被有效利用，正式结果不采用该路线。

### 6.5 union n-best + normalized features

B1/B2 将 A0/A1/A2/A5 多来源候选 union 后做 normalized post-rescore。full union 的 oracle 仍很低，说明候选池有潜力，但 val-tuned normalized rerank 没有超过 final post-rescore。该路线保留为诊断材料，不作为正式系统。

## 7. 对发音评估系统的意义

在发音评估系统中，ASR 错误会影响后续音素对齐、单词级错误定位和发音评分。因此，本 ASR 模块的价值主要体现在：

1. 降低转写错误率，减少后端评估模块的输入噪声。
2. 保留 n-best 候选和分数，后续可以用于不确定性估计。
3. 使用轻量 prefix 适配而不是全量微调 Whisper，训练成本低，便于迁移到特定发音评估场景。
4. 明确区分“ASR 识别准确率提升”和“发音质量评分”，避免将 ASR WER 改善直接等同于发音评分能力。

## 8. 可复现实验命令摘要

最终推理 pipeline 在 lab-252 的部署脚本为：

```bash
cd /DATA_2/guest/custom-whisper-dev
conda activate /DATA_4/guest/envs/custom-whisper-mm

bash scripts/run_a1_beam20_post_rescore_infer.sh \
  --manifest <manifest.jsonl> \
  --output-dir <output_dir> \
  --gpu <free_gpu>
```

pipeline 内部执行三步：

1. `dump_nbest_candidates.py`：A1 checkpoint beam20 decode-only 生成 n-best。
2. `rescore_nbest_with_models.py`：用 A1/A5 teacher-forcing logprob 重打分候选。
3. `rerank_nbest_candidates.py`：按验证集固定参数选择最终候选。

本地材料包中不包含 checkpoint。实际服务器 checkpoint 路径和训练配置保存在：

- `results/lab252/paper_asr_lab252_20260813/configs/A1_blank_prefix_best_val_train_config.json`
- `results/lab252/paper_asr_lab252_20260813/configs/A5_blip2_qformer_best_val_train_config.json`
- `results/lab252/paper_asr_lab252_20260813/configs/checkpoint_config_summary.json`

## 9. 写论文时建议使用的表述

建议表述：

> We use a frozen Whisper large-v3 model with a lightweight trainable decoder-domain prefix to adapt the ASR frontend to the Flickr8K speech domain. Decoding is performed with beam search, and the final hypothesis is selected by a validation-tuned candidate rescoring module combining ASR likelihood, MBR consensus, CLIP image-text consistency, and teacher-forcing likelihood from auxiliary ASR models.

中文对应：

> 本文采用冻结 Whisper large-v3 主体、仅训练轻量 decoder-domain prefix 的方式，将 ASR 前端适配到 Flickr8K 语音场景。推理阶段先通过 beam search 生成候选，再使用验证集调参的候选重打分模块，综合 ASR 似然、MBR 候选共识、CLIP 图文一致性以及辅助 ASR 模型 teacher-forcing 似然选择最终转写。

建议谨慎表述：

> 由于 CIEASR 未提供完全相同的 checkpoint 和测试划分，本文不声称严格同划分超越其绝对 WER。本文在固定可复现划分上，相比无图 Whisper baseline 获得 21.24% 的相对 WER 降低，高于 CIEASR 报告的相对 WER 降低幅度。

