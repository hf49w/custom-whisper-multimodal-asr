# 给 GPT 写论文的输入说明

请基于本目录材料写论文中“方法”和“实验”部分。研究背景是：该 ASR 模块是发音评估系统的一部分，用于提供更准确、稳定的自动转写，不直接输出发音评分。

必须遵守：

1. 正式最佳结果只能写 `A1 beam20 + val-tuned post-rescore reranker`：WER 2.0713%，CER 0.8338%。
2. `test-tuned` 的 2.0391% 只能写成分析结果，不能写成正式结果。
3. 不要声称严格超过 CIEASR 的绝对 WER 2.05%，因为划分和验证集设置可能不同。
4. 可以写：相对无图 Whisper beam20 decode-only baseline，本方法 WER 相对下降 21.24%，CER 相对下降 27.17%；这个相对 WER 改善幅度高于 CIEASR 报告的 15.29%。
5. 不要把 A12/A13 caption/tag semantic rerank 写成有效方法；它已经被实验排除。
6. 不要把 A5 写成最终输出模型；A5 只在最终 pipeline 中作为 teacher-forcing rescore 的辅助 scorer，权重为 0.05。

最终系统结构：

```text
Input audio (+ image in Flickr8K manifest)
  -> A1 frozen Whisper large-v3 + trainable blank decoder prefix
  -> beam20 n-best candidates
  -> candidate feature extraction:
       asr_mean_logprob
       CLIP per-sample z-score
       MBR consensus score
       A1 teacher-forcing logprob
       A5 teacher-forcing logprob
  -> validation-tuned linear score
  -> final transcript
```

最终重打分公式：

```text
score =
  0.5  * asr_mean_logprob
+ 0.01 * clip_zscore
+ 1.0  * mbr_score
+ 0.5  * A1_logprob
+ 0.05 * A5_logprob
```

主要文件：

- 模型定义：`code/model/custom_whisper/model.py`、`code/model/custom_whisper/multimodal.py`
- 训练配置：`code/training_config/scripts/run_flickr8k_decoder_prompt_experiments.sh`
- 训练入口：`code/training_config/scripts/train_visspeech_custom_whisper_fuser.py`
- 完整推理：`code/inference_pipeline/scripts/run_a1_beam20_post_rescore_infer.sh`
- n-best dump：`code/inference_pipeline/scripts/dump_nbest_candidates.py`
- teacher-forcing rescore：`code/inference_pipeline/scripts/rescore_nbest_with_models.py`
- rerank：`code/inference_pipeline/scripts/rerank_nbest_candidates.py`
- 统一结果表：`results/summaries/paper_metrics_table.csv`
- 中文实验报告：`report/experiment_report_zh.md`

可以组织为：

1. ASR frontend for pronunciation assessment
2. Domain prefix adaptation on frozen Whisper
3. Candidate generation and validation-tuned post-rescoring
4. Experimental setup: Flickr8K fixed split, train/val/test sizes
5. Main results and relative improvement over no-image Whisper baseline
6. Ablation: direct visual prompt, BLIP2 prompt, LoRA, learning reranker, semantic caption/tag rerank
7. Limitations: split mismatch with CIEASR, reranker generalization, oracle gap
