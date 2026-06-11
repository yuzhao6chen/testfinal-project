$ErrorActionPreference = "Continue"

Set-Location $PSScriptRoot

$Window = 128
$Epoch = 50
$Delay = 3
$DataDir = "..\anomalydetector\test_jmeter"
$Results = "donut_benchmark_jmeter"
$LogPath = Join-Path $Results "run.log"

# TensorFlow 1.15 GPU on Windows requires old CUDA 10 DLLs.  Force CPU so the
# benchmark does not print scary-but-harmless GPU loader warnings.
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

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

& $PythonExe -c "import importlib.util, sys; missing=[name for name in ('tensorflow','tfsnippet','zhusuan','numpy') if importlib.util.find_spec(name) is None]; print('Python:', sys.executable); print('Missing:', ', '.join(missing) if missing else 'none'); sys.exit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "Please run this script in the TensorFlow 1.x Donut environment."
  Write-Host "Example:"
  Write-Host "conda activate donut-tf1"
  exit 1
}

"" | Set-Content -Path $LogPath
Invoke-PythonStep "Run Donut JMeter benchmark" @(
  ".\tools\run_donut_benchmark.py",
  "--data-dir", $DataDir,
  "--output-dir", $Results,
  "--epochs", "$Epoch",
  "--x-dims", "$Window",
  "--delay", "$Delay",
  "--mask-labels",
  "--skip-existing"
)

"[$(Get-Date -Format s)] Done" | Tee-Object -FilePath $LogPath -Append
Write-Host "Results saved in: $Results"
