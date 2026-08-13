# lab-252 Server Layout

Verified on 2026-06-22.

## Path Mapping

```text
/DATA_4/guest/custom-whisper -> /DATA_2/guest/custom-whisper
```

Use `/DATA_2/guest/custom-whisper` as the canonical deployment path. Existing commands may continue
to use `/DATA_4/guest/custom-whisper` because the symlink resolves to the same directory.

## Project Tree

```text
/DATA_2/guest/custom-whisper/
|-- custom_whisper/              # Deployable Python package
|   |-- model.py                 # Whisper and AudioImageWhisper
|   |-- multimodal.py            # Visual encoders and feature fusers
|   |-- decoding.py
|   |-- transcribe.py
|   |-- audio.py
|   |-- tokenizer.py
|   |-- assets/
|   `-- normalizers/
|-- scripts/                     # Deployable train/eval/inference and run scripts
|-- espnet_specaug_vendor.py     # Deployable SpecAugment implementation
|-- data/                        # Protected runtime data and model files
|   |-- flickr8k/
|   |   |-- audio/
|   |   |-- images/
|   |   |-- captions/
|   |   `-- prepared/
|   |-- flickr30k/
|   |-- flickr30k-images/
|   |-- flickr30k_localized_narratives/
|   |-- visspeech/
|   `-- models/
|       |-- whisper/medium.en.pt
|       |-- whisper/large-v3.pt
|       `-- clip/clip-vit-base-patch32/
|-- outputs/                     # Protected checkpoints, metrics and predictions
|   |-- flickr8k_subset2_seed42_split42_test20_ep5/
|   |-- flickr8k_full_seed42_val10_test10_ep5/
|   |-- flickr8k_full_seed42_val10_test10_ep50/
|   `-- flickr8k_full_large_seed42_val10_test10_ep50/
|-- .cache/torch/                # Protected offline pretrained-model cache
|-- envs/create_conda_env.sh
|-- .git/
|-- README.md
`-- *.log                        # Historical and active suite logs
```

## Full Flickr8k Split

```text
data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/
|-- train_manifest.jsonl  # 32,000 rows
|-- val_manifest.jsonl    # 4,000 rows
`-- test_manifest.jsonl   # 4,000 rows
```

The top-level `data/flickr8k/prepared/manifest.jsonl` contains historical `/home/cvlab/...` paths.
Use the split manifests above for current server runs.

## Runtime Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /DATA_4/guest/envs/custom-whisper-mm
cd /DATA_4/guest/custom-whisper
export TMPDIR=/DATA_2/guest/tmp TMP=/DATA_2/guest/tmp TEMP=/DATA_2/guest/tmp
export TORCH_HOME=/DATA_2/guest/custom-whisper/.cache/torch
export OFFLINE=1
```

Do not use GPU 3. Check availability with `nvidia-smi` before a smoke test or experiment.
