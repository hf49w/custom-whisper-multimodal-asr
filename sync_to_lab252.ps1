[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$SshHost = "lab-252"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$RemoteRoot = "/DATA_2/guest/custom-whisper"
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

Write-Host "Local source: $ProjectRoot"
Write-Host "Server:       $SshHost"
Write-Host "Remote root:  $RemoteRoot"
$LayoutCheck = 'test "$(readlink -f {0})" = {0} && test -d {0}/data && test -d {0}/outputs && echo server_layout_ok' -f $RemoteRoot
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
    Write-Host "Preview only. Re-run with -Apply to deploy."
    exit 0
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LocalTemp = Join-Path ([System.IO.Path]::GetTempPath()) "custom-whisper-deploy-$Stamp"
$ArchiveName = "custom-whisper-code-$Stamp.tar.gz"
$RemoteScriptName = "custom-whisper-deploy-$Stamp.sh"
$LocalArchive = Join-Path $LocalTemp $ArchiveName
$LocalRemoteScript = Join-Path $LocalTemp $RemoteScriptName
$RemoteArchive = "$RemoteTmp/$ArchiveName"
$RemoteScript = "$RemoteTmp/$RemoteScriptName"
$RemoteStage = "$RemoteTmp/custom-whisper-stage-$Stamp"
$RemoteOld = "$RemoteTmp/custom-whisper-old-$Stamp"
$RemoteBackup = "$RemoteBackupRoot/custom-whisper-code-$Stamp.tar.gz"

New-Item -ItemType Directory -Force -Path $LocalTemp | Out-Null

$DeployTemplate = @'
set -euo pipefail

ROOT="__REMOTE_ROOT__"
TMP_ROOT="__REMOTE_TMP__"
STAGE="__REMOTE_STAGE__"
OLD="__REMOTE_OLD__"
ARCHIVE="__REMOTE_ARCHIVE__"
SCRIPT_PATH="__REMOTE_SCRIPT__"
BACKUP_ROOT="__REMOTE_BACKUP_ROOT__"
BACKUP="__REMOTE_BACKUP__"
CONDA_ENV="__CONDA_ENV__"
DEPLOYED=0

cleanup() {
    status=$?
    trap - EXIT
    if [ "$status" -ne 0 ] && [ "$DEPLOYED" -eq 1 ] && [ -d "$OLD/custom_whisper" ]; then
        echo "Deployment failed; restoring previous server code." >&2
        rm -rf -- "$ROOT/custom_whisper" "$ROOT/scripts"
        mv -- "$OLD/custom_whisper" "$ROOT/custom_whisper"
        mv -- "$OLD/scripts" "$ROOT/scripts"
        cp -p -- "$OLD/espnet_specaug_vendor.py" "$ROOT/espnet_specaug_vendor.py"
    fi
    rm -rf -- "$STAGE" "$OLD"
    rm -f -- "$ARCHIVE" "$SCRIPT_PATH"
    exit "$status"
}
trap cleanup EXIT

[ "$(readlink -f "$ROOT")" = "/DATA_2/guest/custom-whisper" ]
[ "$(readlink -f "$TMP_ROOT")" = "/DATA_2/guest/tmp" ]
test -d "$ROOT/data"
test -d "$ROOT/outputs"

if pgrep -af '[t]rain_visspeech_custom_whisper_fuser.py|[e]val_visspeech_custom_whisper_fuser.py|[t]ranscribe_multimodal_checkpoint.py|[r]un_flickr8k' >/dev/null; then
    echo "A custom-whisper train/eval/inference process is running; deployment aborted." >&2
    exit 1
fi

rm -rf -- "$STAGE" "$OLD"
mkdir -p -- "$STAGE" "$OLD" "$BACKUP_ROOT"
tar -xzf "$ARCHIVE" -C "$STAGE"
test -f "$STAGE/custom_whisper/model.py"
test -f "$STAGE/custom_whisper/multimodal.py"
test -f "$STAGE/scripts/train_visspeech_custom_whisper_fuser.py"
test -f "$STAGE/scripts/eval_visspeech_custom_whisper_fuser.py"
test -f "$STAGE/scripts/transcribe_multimodal_checkpoint.py"
test -f "$STAGE/espnet_specaug_vendor.py"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$TMP_ROOT" TMP="$TMP_ROOT" TEMP="$TMP_ROOT"

cd "$STAGE"
python -B -c 'import custom_whisper; from custom_whisper.model import AudioImageWhisper; from custom_whisper.multimodal import build_feature_fuser'
python -B scripts/train_visspeech_custom_whisper_fuser.py --help >/dev/null
python -B scripts/eval_visspeech_custom_whisper_fuser.py --help >/dev/null
python -B scripts/transcribe_multimodal_checkpoint.py --help >/dev/null

tar -czf "$BACKUP" -C "$ROOT" custom_whisper scripts espnet_specaug_vendor.py
cp -a -- "$ROOT/custom_whisper" "$OLD/custom_whisper"
cp -a -- "$ROOT/scripts" "$OLD/scripts"
cp -p -- "$ROOT/espnet_specaug_vendor.py" "$OLD/espnet_specaug_vendor.py"
DEPLOYED=1
rm -rf -- "$ROOT/custom_whisper" "$ROOT/scripts"
mv -- "$STAGE/custom_whisper" "$ROOT/custom_whisper"
mv -- "$STAGE/scripts" "$ROOT/scripts"
cp -p -- "$STAGE/espnet_specaug_vendor.py" "$ROOT/espnet_specaug_vendor.py"
find "$ROOT/custom_whisper" "$ROOT/scripts" -type d -exec chmod 755 {} +
find "$ROOT/custom_whisper" "$ROOT/scripts" -type f -exec chmod 644 {} +
find "$ROOT/scripts" -type f -name '*.sh' -exec chmod 755 {} +

cd "$ROOT"
python -B -c 'import custom_whisper; from custom_whisper.model import AudioImageWhisper; from custom_whisper.multimodal import build_feature_fuser'
python -B scripts/train_visspeech_custom_whisper_fuser.py --help >/dev/null
python -B scripts/eval_visspeech_custom_whisper_fuser.py --help >/dev/null
python -B scripts/transcribe_multimodal_checkpoint.py --help >/dev/null

rm -rf -- "$OLD"
DEPLOYED=0
echo "Deployment complete."
echo "Backup: $BACKUP"
'@

$DeployScriptContent = $DeployTemplate.Replace("__REMOTE_ROOT__", $RemoteRoot).
    Replace("__REMOTE_TMP__", $RemoteTmp).
    Replace("__REMOTE_STAGE__", $RemoteStage).
    Replace("__REMOTE_OLD__", $RemoteOld).
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
