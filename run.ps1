param(
    [string]$Model = "",
    [string]$Dataset = "coqa",
    [double]$Fraction = 0.01,
    [int]$NumGen = 5,
    [string]$Device = "",
    [int]$ProjectInd = 0,
    [double]$Temperature = 0.5,
    [double]$TopP = 0.99,
    [int]$TopK = 10,
    [switch]$Overwrite
)

# 1. Locate Python executable
$pythonExe = $null
if (Test-Path "..\.venv\Scripts\python.exe") {
    $pythonExe = (Resolve-Path "..\.venv\Scripts\python.exe").Path
} elseif (Test-Path ".\.venv\Scripts\python.exe") {
    $pythonExe = (Resolve-Path ".\.venv\Scripts\python.exe").Path
} elseif ($env:VIRTUAL_ENV -and (Test-Path "$env:VIRTUAL_ENV\Scripts\python.exe")) {
    $pythonExe = "$env:VIRTUAL_ENV\Scripts\python.exe"
}

# 2. Determine default model
if (-not $Model) {
    if (Test-Path ".\data\weights\llama-7b-hf") {
        $Model = "llama-7b-hf"
    } else {
        $Model = "facebook/opt-125m"
    }
}

# 3. Setup output and log paths
$safeModel = $Model.Replace("/", "_")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFolder = Join-Path (Get-Location) "data\output"
if (-not (Test-Path $outputFolder)) {
    New-Item -ItemType Directory -Path $outputFolder -Force | Out-Null
}
$logFile = Join-Path $outputFolder "run_${safeModel}_${Dataset}_${timestamp}.log"
$targetDir = Join-Path $outputFolder "${safeModel}_${Dataset}_${ProjectInd}"

Write-Host "=========================================================="
Write-Host " INSIDE / EigenScore Pipeline: Generation & Evaluation"
Write-Host "=========================================================="
Write-Host " Model:       $Model"
Write-Host " Dataset:     $Dataset"
Write-Host " Fraction:    $Fraction"
Write-Host " NumGen:      $NumGen"
Write-Host " ProjectInd:  $ProjectInd"
Write-Host " Log File:    $logFile"
Write-Host "=========================================================="

# 4. Build command arguments
$genArgs = @("-m", "pipeline.generate",
             "--model", $Model,
             "--dataset", $Dataset,
             "--fraction_of_data_to_use", "$Fraction",
             "--num_generations_per_prompt", "$NumGen",
             "--temperature", "$Temperature",
             "--top_p", "$TopP",
             "--top_k", "$TopK",
             "--project_ind", "$ProjectInd")

if ($Device) {
    $genArgs += @("--device", $Device)
}
if ($Overwrite) {
    $genArgs += @("--overwrite")
}

# Helper to run commands and log output
function Invoke-PipelineStep {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $logFile -Append
}

# 5. Run Generation Loop
Write-Host "`n[1/2] Running Generation and EigenScore computation..." -ForegroundColor Cyan
if ($pythonExe) {
    Invoke-PipelineStep -Executable $pythonExe -Arguments $genArgs
} else {
    $condaExe = $env:CONDA_EXE
    if (-not $condaExe) { $condaExe = Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe' }
    if (Test-Path $condaExe) {
        $cArgs = @("run", "-n", "eigenscore-py310", "python") + $genArgs
        Invoke-PipelineStep -Executable $condaExe -Arguments $cArgs
    } else {
        Invoke-PipelineStep -Executable "python" -Arguments $genArgs
    }
}

# 6. Locate generated .pkl and Run Evaluation
Write-Host "`n[2/2] Running Evaluation & Computing AUROC / Correlation Metrics..." -ForegroundColor Cyan
$pklFile = $null
if (Test-Path $targetDir) {
    $candidates = Get-ChildItem -Path $targetDir -Filter "*.pkl" | Where-Object { $_.Name -notlike "*_partial*" } | Sort-Object LastWriteTime -Descending
    if ($candidates) {
        $pklFile = $candidates[0].FullName
    }
}

if ($pklFile) {
    Write-Host "Evaluating generated output: $pklFile"
    $evalArgs = @("-m", "func.evalFunc", $pklFile)
    if ($pythonExe) {
        Invoke-PipelineStep -Executable $pythonExe -Arguments $evalArgs
    } else {
        Invoke-PipelineStep -Executable "python" -Arguments $evalArgs
    }
} else {
    Write-Host "Warning: No .pkl file found in $targetDir for evaluation." -ForegroundColor Yellow
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Pipeline Finished!" -ForegroundColor Green
Write-Host " Full execution log saved to: $logFile" -ForegroundColor Green
Write-Host " ROC Curves exported to:      ./Figure/AUROC_${Dataset}.png" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green