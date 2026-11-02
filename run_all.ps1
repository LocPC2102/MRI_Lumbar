# Runs the whole segmentation pipeline from the project root.
#   .\src\cnn_segmentation\run_all.ps1

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")   # project root (test/)

function Invoke-Step {
    param([string]$Title, [string]$Module)

    Write-Host ""
    Write-Host "==== $Title ====" -ForegroundColor Cyan
    python -m $Module
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed: $Title" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Step "1/5 Audit dataset (manifest)"      "src.cnn_segmentation.datasets.audit_dataset"
Invoke-Step "2/5 Split dataset (train/val/test)" "src.cnn_segmentation.datasets.split_dataset"
Invoke-Step "3/5 Train model"                    "src.cnn_segmentation.train"
Invoke-Step "4/5 Report metrics"                 "src.cnn_segmentation.eval.report_metrics"
Invoke-Step "5/5 Predict"                        "src.cnn_segmentation.predict"

Write-Host ""
Write-Host "Pipeline finished." -ForegroundColor Green