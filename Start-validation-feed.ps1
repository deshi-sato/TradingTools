cd C:\Users\Owner\documents\desshi_signal_viewer
C:/Users/Owner/Documents/desshi_signal_viewer/.venv/Scripts/Activate.ps1

param(
    [string]$Source = '',
    [string]$Dest = 'db\naut_market.db',
    [string]$DatasetId = 'VALIDATION_FEED',
    [double]$Speed = 1.0,
    [double]$StartDelay = 0.0,
    [int]$Limit = 0
)

$arguments = @('--dest', $Dest, '--dataset-id', $DatasetId, '--speed', $Speed, '--start-delay', $StartDelay, '--limit', $Limit)
if ($Source) {
    $arguments = @('--source', $Source) + $arguments
}

Write-Host "Launching validation feed" -ForegroundColor Cyan
Write-Host "  Source: $Source" -ForegroundColor Cyan
Write-Host "  Dest:   $Dest" -ForegroundColor Cyan
Write-Host "  DatasetId: $DatasetId" -ForegroundColor Cyan
Write-Host "  Speed: $Speed" -ForegroundColor Cyan
Write-Host "  StartDelay: $StartDelay" -ForegroundColor Cyan
Write-Host "  Limit: $Limit" -ForegroundColor Cyan

py -m scripts.validation_feed @arguments
