param(
  [switch]$Fresh
)

$ErrorActionPreference = "Continue"

Set-Location $PSScriptRoot

$Window = 128
$Epoch = 50
$Delay = 3
$DataDir = "..\anomalydetector\paper_test_prometheus"
$Results = "donut_benchmark_paper_prometheus"
$LogPath = Join-Path $Results "run.log"

$env:CUDA_VISIBLE_DEVICES = "-1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

function Write-LogLine {
  param([string]$Message)

  Write-Host $Message
  Add-Content -Path $LogPath -Value $Message -Encoding UTF8
}

function Invoke-PythonStep {
  param(
    [string]$StepName,
    [string[]]$Arguments
  )

  Write-LogLine "[$(Get-Date -Format s)] $StepName"
  $StepStdoutPath = Join-Path $Results ([System.IO.Path]::GetRandomFileName())
  $StepStderrPath = Join-Path $Results ([System.IO.Path]::GetRandomFileName())
  $Process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $Arguments `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $StepStdoutPath `
    -RedirectStandardError $StepStderrPath `
    -Wait `
    -PassThru `
    -WindowStyle Hidden
  $ExitCode = $Process.ExitCode

  foreach ($StepLogPath in @($StepStdoutPath, $StepStderrPath)) {
    if (Test-Path $StepLogPath) {
      $StepLines = Get-Content -Path $StepLogPath
      Add-Content -Path $LogPath -Value $StepLines -Encoding UTF8
      $StepLines | ForEach-Object { Write-Host $_ }
    }
    Remove-Item -LiteralPath $StepLogPath -Force -ErrorAction SilentlyContinue
  }

  return $ExitCode
}

function Test-SummaryHasFileName {
  param([string]$FileName)

  $SummaryPath = Join-Path $Results "summary.json"
  if (-not (Test-Path $SummaryPath)) {
    return $false
  }

  try {
    $Summary = Get-Content -Path $SummaryPath -Raw | ConvertFrom-Json
    foreach ($Item in $Summary.files) {
      if ((Split-Path $Item.file -Leaf) -eq $FileName) {
        return $true
      }
    }
  } catch {
    return $false
  }
  return $false
}

if ($Fresh -and (Test-Path $Results)) {
  Remove-Item -LiteralPath $Results -Recurse -Force
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
  Write-Host "Example: conda activate donut-tf1"
  exit 1
}

"" | Set-Content -Path $LogPath -Encoding UTF8
$BenchmarkBaseArgs = @(
  ".\tools\run_donut_benchmark.py",
  "--data-dir", $DataDir,
  "--output-dir", $Results,
  "--epochs", "$Epoch",
  "--x-dims", "$Window",
  "--delay", "$Delay",
  "--mask-labels",
  "--skip-existing"
)

$DataFiles = @(Get-ChildItem -Path $DataDir -Filter "*.csv" | Sort-Object FullName)
for ($Index = 1; $Index -le $DataFiles.Count; $Index++) {
  $CurrentFile = $DataFiles[$Index - 1]
  $BenchmarkArgs = $BenchmarkBaseArgs + @(
    "--start-index", "$Index",
    "--max-files", "1"
  )

  $ExitCode = Invoke-PythonStep `
    "Run Donut Prometheus benchmark [$Index/$($DataFiles.Count)]" `
    $BenchmarkArgs

  if ($ExitCode -ne 0) {
    if (Test-SummaryHasFileName $CurrentFile.Name) {
      Write-Host "Python exited with code $ExitCode after recording $($CurrentFile.Name); continuing with next KPI."
      continue
    }

    Write-Host ""
    Write-Host "Run Donut Prometheus benchmark failed on $($CurrentFile.Name). See log: $LogPath"
    exit $ExitCode
  }
}

Write-LogLine "[$(Get-Date -Format s)] Done"
Write-Host "Results saved in: $Results"
