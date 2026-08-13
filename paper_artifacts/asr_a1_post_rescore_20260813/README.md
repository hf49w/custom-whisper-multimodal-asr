# ASR 论文材料包

本目录用于写论文/喂给 GPT 的材料整理，主题是发音评估系统中的 ASR 模块。

内容范围：

- `code/model/`：模型定义相关代码。
- `code/training_config/`：训练、实验配置、监控代码。
- `code/inference_pipeline/`：最终 A1 beam20 + val-tuned post-rescore 推理链路代码。
- `code/diagnostics_and_ablation/`：A9/A10/A11/A12/A13/B1/B2 等诊断、消融和失败路线代码。
- `docs/`：原实验说明文档和项目布局文档快照。
- `results/lab250/`：从 lab-250 拉取的小型实验结果文件；不包含 n-best 全量文件、checkpoint、数据集。
- `results/lab252/`：从 lab-252 拉取的部署配置、checkpoint 内部 train_config、smoke-test 小结果。
- `results/summaries/`：统一整理后的结果表。
- `report/experiment_report_zh.md`：中文实验报告。
- `report/paper_prompt_for_gpt.md`：给 GPT 写论文时可直接粘贴的说明。

关键正式结果：

- 正式最佳：A1 beam20 + val-tuned post-rescore reranker
- Test WER：2.0713%
- Test CER：0.8338%
- 对比无图 Whisper beam20 decode-only baseline：WER 从 2.6299% 降到 2.0713%，相对下降 21.24%；CER 从 1.1449% 降到 0.8338%，相对下降 27.17%。

注意：

- `test-tuned` 的 2.0391% 只能作为分析结果，不能作为正式结果。
- 本材料包没有复制大文件：checkpoint、数据集、全量 n-best、全量 predictions、`.pkl` 模型文件。
