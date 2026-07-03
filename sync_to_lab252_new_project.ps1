[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$SshHost = "lab-252",
    [string]$RemoteProjectRoot = "/DATA_2/guest/custom-whisper-dev"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path

# 原项目只作为 data/.cache 来源，不会被替换
$RemoteSourceRoot = "/DATA_2/guest/custom-whisper"

# 新项目部署位置
$RemoteRoot = $RemoteProjectRoot

$RemoteTmp = "/DATA_2/guest/tmp"
$RemoteBackupRoot = "/DATA_2/guest/code_backups"
$CondaEnv = "/DATA_4/guest/envs/custom-whisper-mm"

$SshOptions = @(
    "-o", "PreferredAuthentications=publickey",
    "-o", "PasswordAuthentication=no"
)

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)] [string]$Command,
        [Parameter(Mandatory)] [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-SshChecked {
    param([Parameter(Mandatory)] [string]$RemoteCommand)
    Invoke-NativeChecked -Command "ssh" -Arguments ($SshOptions + @($SshHost, $RemoteCommand))
}

$RequiredLocalPaths = @(
    (Join-Path $ProjectRoot "custom_whisper"),
    (Join-Path $ProjectRoot "scripts"),
    (Join-Path $ProjectRoot "espnet_specaug_vendor.py")
)

foreach ($Path in $RequiredLocalPaths) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required local path is missing: $Path"
    }
}

if ($RemoteRoot -eq $RemoteSourceRoot) {
    throw "RemoteProjectRoot must not be the original project path: $RemoteSourceRoot"
}

Write-Host "Local source:        $ProjectRoot"
Write-Host "Server:              $SshHost"
Write-Host "Original project:    $RemoteSourceRoot"
Write-Host "New project root:    $RemoteRoot"

$LayoutCheck = @"
set -e
test -d '$RemoteSourceRoot'
test -d '$RemoteSourceRoot/data'
test -d '$RemoteSourceRoot/outputs'
test -d '$RemoteSourceRoot/.cache' || true
mkdir -p '$RemoteTmp' '$RemoteBackupRoot'
echo server_layout_ok
"@

Invoke-SshChecked $LayoutCheck

$DeployFiles = Get-ChildItem -Recurse -File -LiteralPath @(
    (Join-Path $ProjectRoot "custom_whisper"),
    (Join-Path $ProjectRoot "scripts")
) | Where-Object {
    $_.FullName -notmatch "[\\/]__pycache__[\\/]" -and $_.Extension -ne ".pyc"
}

$DeployFiles += Get-Item -LiteralPath (Join-Path $ProjectRoot "espnet_specaug_vendor.py")

Write-Host "Deployable files: $($DeployFiles.Count)"
$DeployFiles |
    ForEach-Object { $_.FullName.Substring($ProjectRoot.Length + 1) } |
    Sort-Object |
    ForEach-Object { Write-Host "  $_" }

if (-not $Apply) {
    Write-Host ""
    Write-Host "Preview only. Re-run with -Apply to deploy new project."
    exit 0
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LocalTemp = Join-Path ([System.IO.Path]::GetTempPath()) "custom-whisper-new-project-$Stamp"
$ArchiveName = "custom-whisper-new-project-code-$Stamp.tar.gz"
$RemoteScriptName = "custom-whisper-new-project-deploy-$Stamp.sh"

$LocalArchive = Join-Path $LocalTemp $ArchiveName
$LocalRemoteScript = Join-Path $LocalTemp $RemoteScriptName

$RemoteArchive = "$RemoteTmp/$ArchiveName"
$RemoteScript = "$RemoteTmp/$RemoteScriptName"
$RemoteStage = "$RemoteTmp/custom-whisper-new-stage-$Stamp"
$RemoteBackup = "$RemoteBackupRoot/custom-whisper-new-project-code-$Stamp.tar.gz"

New-Item -ItemType Directory -Force -Path $LocalTemp | Out-Null

$DeployTemplate = @'
set -euo pipefail

SOURCE_ROOT="__REMOTE_SOURCE_ROOT__"
ROOT="__REMOTE_ROOT__"
TMP_ROOT="__REMOTE_TMP__"
STAGE="__REMOTE_STAGE__"
ARCHIVE="__REMOTE_ARCHIVE__"
SCRIPT_PATH="__REMOTE_SCRIPT__"
BACKUP_ROOT="__REMOTE_BACKUP_ROOT__"
BACKUP="__REMOTE_BACKUP__"
CONDA_ENV="__CONDA_ENV__"

cleanup() {
    status=$?
    trap - EXIT
    rm -rf -- "$STAGE"
    rm -f -- "$ARCHIVE" "$SCRIPT_PATH"
    exit "$status"
}
trap cleanup EXIT

if [ "$ROOT" = "$SOURCE_ROOT" ]; then
    echo "ERROR: target ROOT equals original SOURCE_ROOT; abort." >&2
    exit 1
fi

test -d "$SOURCE_ROOT"
test -d "$SOURCE_ROOT/data"
test -d "$SOURCE_ROOT/outputs"

mkdir -p -- "$ROOT" "$TMP_ROOT" "$BACKUP_ROOT"
rm -rf -- "$STAGE"
mkdir -p -- "$STAGE"

echo "[INFO] Original project: $SOURCE_ROOT"
echo "[INFO] New project:      $ROOT"

tar -xzf "$ARCHIVE" -C "$STAGE"

test -f "$STAGE/custom_whisper/model.py"
test -f "$STAGE/custom_whisper/multimodal.py"
test -f "$STAGE/scripts/train_visspeech_custom_whisper_fuser.py"
test -f "$STAGE/scripts/eval_visspeech_custom_whisper_fuser.py"
test -f "$STAGE/espnet_specaug_vendor.py"

echo "[INFO] Prepare data/cache/outputs layout"

if [ -e "$ROOT/data" ] && [ ! -L "$ROOT/data" ]; then
    echo "ERROR: $ROOT/data exists and is not a symlink. Please inspect it manually." >&2
    exit 1
fi
ln -sfn "$SOURCE_ROOT/data" "$ROOT/data"

if [ -e "$SOURCE_ROOT/.cache" ]; then
    if [ -e "$ROOT/.cache" ] && [ ! -L "$ROOT/.cache" ]; then
        echo "ERROR: $ROOT/.cache exists and is not a symlink. Please inspect it manually." >&2
        exit 1
    fi
    ln -sfn "$SOURCE_ROOT/.cache" "$ROOT/.cache"
fi

# 新项目使用独立 outputs，避免覆盖原项目实验结果
mkdir -p "$ROOT/outputs"

echo "[INFO] Validate staged code"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$TMP_ROOT" TMP="$TMP_ROOT" TEMP="$TMP_ROOT"
export TORCH_HOME="$ROOT/.cache/torch"
export OFFLINE=1

cd "$STAGE"
python -B -c 'import custom_whisper; from custom_whisper.model import AudioImageWhisper; from custom_whisper.multimodal import build_feature_fuser'
python -B scripts/train_visspeech_custom_whisper_fuser.py --help >/dev/null
python -B scripts/eval_visspeech_custom_whisper_fuser.py --help >/dev/null

echo "[INFO] Backup existing code in new project if present"
BACKUP_ITEMS=()
[ -d "$ROOT/custom_whisper" ] && BACKUP_ITEMS+=("custom_whisper")
[ -d "$ROOT/scripts" ] && BACKUP_ITEMS+=("scripts")
[ -f "$ROOT/espnet_specaug_vendor.py" ] && BACKUP_ITEMS+=("espnet_specaug_vendor.py")

if [ "${#BACKUP_ITEMS[@]}" -gt 0 ]; then
    tar -czf "$BACKUP" -C "$ROOT" "${BACKUP_ITEMS[@]}"
    echo "[INFO] Backup saved: $BACKUP"
else
    echo "[INFO] No existing code to backup in new project."
fi

echo "[INFO] Deploy code to new project"
rm -rf -- "$ROOT/custom_whisper" "$ROOT/scripts" "$ROOT/espnet_specaug_vendor.py"
mv -- "$STAGE/custom_whisper" "$ROOT/custom_whisper"
mv -- "$STAGE/scripts" "$ROOT/scripts"
cp -p -- "$STAGE/espnet_specaug_vendor.py" "$ROOT/espnet_specaug_vendor.py"

find "$ROOT/custom_whisper" "$ROOT/scripts" -type d -exec chmod 755 {} +
find "$ROOT/custom_whisper" "$ROOT/scripts" -type f -exec chmod 644 {} +
find "$ROOT/scripts" -type f -name '*.sh' -exec chmod 755 {} +

echo "[INFO] Validate deployed new project"
cd "$ROOT"
python -B -c 'import custom_whisper; from custom_whisper.model import AudioImageWhisper; from custom_whisper.multimodal import build_feature_fuser'
python -B scripts/train_visspeech_custom_whisper_fuser.py --help >/dev/null
python -B scripts/eval_visspeech_custom_whisper_fuser.py --help >/dev/null

echo "New project deployment complete."
echo "New project root: $ROOT"
echo "Shared data:      $ROOT/data -> $SOURCE_ROOT/data"
echo "Shared cache:     $ROOT/.cache -> $SOURCE_ROOT/.cache"
echo "Separate outputs: $ROOT/outputs"
'@

$DeployScriptContent = $DeployTemplate.Replace("__REMOTE_SOURCE_ROOT__", $RemoteSourceRoot).
    Replace("__REMOTE_ROOT__", $RemoteRoot).
    Replace("__REMOTE_TMP__", $RemoteTmp).
    Replace("__REMOTE_STAGE__", $RemoteStage).
    Replace("__REMOTE_ARCHIVE__", $RemoteArchive).
    Replace("__REMOTE_SCRIPT__", $RemoteScript).
    Replace("__REMOTE_BACKUP_ROOT__", $RemoteBackupRoot).
    Replace("__REMOTE_BACKUP__", $RemoteBackup).
    Replace("__CONDA_ENV__", $CondaEnv)

try {
    Push-Location $ProjectRoot
    try {
        Invoke-NativeChecked -Command "tar" -Arguments @(
            "-czf", $LocalArchive,
            "--exclude=*/__pycache__/*",
            "--exclude=*.pyc",
            "custom_whisper",
            "scripts",
            "espnet_specaug_vendor.py"
        )
    }
    finally {
        Pop-Location
    }

    [System.IO.File]::WriteAllText(
        $LocalRemoteScript,
        ($DeployScriptContent -replace "`r", ""),
        [System.Text.UTF8Encoding]::new($false)
    )

    Invoke-SshChecked "mkdir -p '$RemoteTmp' '$RemoteBackupRoot'"

    Invoke-NativeChecked -Command "scp" -Arguments (
        $SshOptions + @($LocalArchive, "${SshHost}:$RemoteArchive")
    )

    Invoke-NativeChecked -Command "scp" -Arguments (
        $SshOptions + @($LocalRemoteScript, "${SshHost}:$RemoteScript")
    )

    Invoke-SshChecked "exec bash '$RemoteScript'"
}
finally {
    if (Test-Path -LiteralPath $LocalTemp) {
        Remove-Item -Recurse -Force -LiteralPath $LocalTemp
    }
}