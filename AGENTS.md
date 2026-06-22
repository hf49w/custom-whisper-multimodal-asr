# Codex Project Instructions

## Project Role

This repository is the local source-only mirror of the multimodal ASR code running on `lab-252`.
Do not assume the local directory contains datasets, model weights, checkpoints, Conda environments, or experiment outputs.

## Server Mapping

- SSH: `ssh -o PreferredAuthentications=publickey -o PasswordAuthentication=no lab-252`
- Local source root: this repository root (the current Codex working directory)
- Server entry path: `/DATA_4/guest/custom-whisper`
- Server real path: `/DATA_2/guest/custom-whisper`
- Conda environment: `/DATA_4/guest/envs/custom-whisper-mm`
- Server temporary files: `/DATA_2/guest/tmp`
- Server code backups: `/DATA_2/guest/code_backups`

`/DATA_4/guest/custom-whisper` is a symlink to `/DATA_2/guest/custom-whisper`. Prefer the real
`/DATA_2` path for deployment, temporary files, backups, and downloads.

## Deployment Scope

Only these local paths are deployable source code:

- `custom_whisper/`
- `scripts/`
- `espnet_specaug_vendor.py`

Never mirror or delete the server project root. In particular, never overwrite or remove:

- `data/`
- `outputs/`
- `.cache/`
- `.git/`
- `envs/`
- `README.md`, `.gitignore`, or training logs

Never use `rsync --delete` against `/DATA_2/guest/custom-whisper/`, and never SCP the entire local
project onto that root. The local project intentionally omits about 50 GB of required runtime assets.

## Sync Workflow

Do not deploy unless the user explicitly requests it.

1. Review local changes and run local syntax/tests where possible.
2. Preview connectivity and deployment scope:
   `powershell -ExecutionPolicy Bypass -File .\sync_to_lab252.ps1`
3. Confirm no train/eval/inference process is running on the server.
4. Deploy with backup, staging checks, exact code replacement, live checks, and automatic rollback:
   `powershell -ExecutionPolicy Bypass -File .\sync_to_lab252.ps1 -Apply`
5. For model, data loader, fusion, training, evaluation, or inference changes, also run a one-sample
   GPU smoke test after deployment. Use a free GPU other than GPU 3.
6. Report the backup path, deployed files, checks performed, and any warnings.

All server downloads and temporary artifacts must be placed under `/DATA_2`, never `/`, `/tmp`, or
`/DATA_4`.

See `SERVER_LAYOUT.md` for the current runtime tree and important data/model/output paths.
