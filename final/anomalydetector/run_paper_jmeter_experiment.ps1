$ErrorActionPreference = "Continue"

Set-Location $PSScriptRoot

$Window = 128
$Epoch = 10
$Dataset = "paper_train_jmeter"
$Testset = "paper_test_jmeter"
$Snapshot = "snapshot_paper_jmeter"
$Results = "results_paper_jmeter"
$LogPath = Join-Path $Results "run.log"

function Invoke-PythonStep {
  param(
    [string]$StepName,
    [string[]]$Arguments
  )

  "[$(Get-Date -Format s)] $StepName" | Tee-Object -FilePath $LogPath -Append
  & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "$StepName failed. See log: $LogPath"
    exit $LASTEXITCODE
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
  Write-Host "Please run this script in the PyTorch SR-CNN environment."
  Write-Host "Example: conda activate testenv"
  exit 1
}

"" | Set-Content -Path $LogPath
Invoke-PythonStep "Generate training data" @("-m", "srcnn.generate_data", "--data", $Dataset, "--window", $Window, "--step", "64")
Invoke-PythonStep "Train SR-CNN" @("-m", "srcnn.train", "--data", $Dataset, "--window", $Window, "--epoch", $Epoch, "--batch_size", "256", "--num_workers", "0", "--save", $Snapshot)
Invoke-PythonStep "Evaluate SR-CNN" @("-m", "srcnn.evalue", "--data", $Testset, "--window", $Window, "--epoch", $Epoch, "--model_path", $Snapshot, "--delay", "3", "--result_dir", $Results)

"[$(Get-Date -Format s)] Done" | Tee-Object -FilePath $LogPath -Append
Write-Host "Results saved in: $Results"
