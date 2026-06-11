$ErrorActionPreference = "Continue"

Set-Location $PSScriptRoot

$Window = 128
$Epoch = 10
$Dataset = "online_boutique_train_prometheus"
$Testset = "test_prometheus"
$Snapshot = "snapshot_prometheus"
$Results = "results_prometheus"
$LogPath = Join-Path $Results "run.log"

function Invoke-PythonStep {
  param(
    [string]$StepName,
    [string[]]$Arguments
  )

  "[$(Get-Date -Format s)] $StepName" | Tee-Object -FilePath $LogPath -Append
  $StepLogPath = Join-Path $Results ([System.IO.Path]::GetRandomFileName())
  & $PythonExe @Arguments *> $StepLogPath
  $ExitCode = $LASTEXITCODE
  if (Test-Path $StepLogPath) {
    $StepLines = Get-Content -Path $StepLogPath
    Add-Content -Path $LogPath -Value $StepLines
    $StepLines | ForEach-Object { Write-Host $_ }
  }
  Remove-Item -LiteralPath $StepLogPath -Force -ErrorAction SilentlyContinue

  if ($ExitCode -ne 0) {
    Write-Host ""
    Write-Host "$StepName failed. See log: $LogPath"
    exit $ExitCode
  }
}

New-Item -ItemType Directory -Force $Snapshot | Out-Null
New-Item -ItemType Directory -Force $Results | Out-Null

if ($env:PYTHON_EXE -and (Test-Path $env:PYTHON_EXE)) {
  $PythonExe = $env:PYTHON_EXE
} elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
  $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
} elseif ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $PythonExe = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
} else {
  $PythonExe = "python"
}

& $PythonExe -c "import importlib.util, sys; missing=[name for name in ('torch','torchvision','numpy','sklearn','tqdm') if importlib.util.find_spec(name) is None]; print('Python:', sys.executable); print('Missing:', ', '.join(missing) if missing else 'none'); sys.exit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "Please install the missing dependencies in this same environment, then run this script again:"
  Write-Host "& `"$PythonExe`" -m pip install numpy pandas Cython tqdm scikit-learn"
  Write-Host "& `"$PythonExe`" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
  exit 1
}

"" | Set-Content -Path $LogPath
Invoke-PythonStep "Generate training data" @("-m", "srcnn.generate_data", "--data", $Dataset, "--window", $Window, "--step", "64")
Invoke-PythonStep "Train SR-CNN" @("-m", "srcnn.train", "--data", $Dataset, "--window", $Window, "--epoch", $Epoch, "--batch_size", "256", "--num_workers", "0", "--save", $Snapshot)
Invoke-PythonStep "Evaluate SR-CNN" @("-m", "srcnn.evalue", "--data", $Testset, "--window", $Window, "--epoch", $Epoch, "--model_path", $Snapshot, "--delay", "3", "--result_dir", $Results)

$DonutSummary = "..\donut\donut_benchmark_prometheus\summary.json"
if (Test-Path $DonutSummary) {
  Invoke-PythonStep "Summarize SR-CNN on Donut-valid Prometheus files" @(
    ".\tools\summarize_srcnn_scores.py",
    "--summary", (Join-Path $Results "test_prometheus_eval_summary.json"),
    "--saved-scores", (Join-Path $Results "test_prometheus_saved_scores.json"),
    "--donut-summary", $DonutSummary,
    "--align-to-donut-scores",
    "--delay", "3",
    "--output", (Join-Path $Results "test_prometheus_donut_valid_subset_summary.json")
  )
} else {
  Write-Host "Donut Prometheus summary not found; skip Donut-valid subset summary: $DonutSummary"
}

"[$(Get-Date -Format s)] Done" | Tee-Object -FilePath $LogPath -Append
Write-Host "Results saved in: $Results"
