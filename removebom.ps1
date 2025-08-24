param(
    [string]$Path = "C:\Users\Owner\Documents\desshi_signal_viewer\scripts\analyze_scores.py"
)

# ファイルをバイナリで読み込む
$bytes = [System.IO.File]::ReadAllBytes($Path)

# UTF-8 BOM (EF BB BF) が付いていれば削除
if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    $bytes = $bytes[3..($bytes.Length-1)]
    Write-Host "BOM removed from $Path"
} else {
    Write-Host "No BOM found in $Path"
}

# BOMなしで書き戻す
[System.IO.File]::WriteAllBytes($Path, $bytes)
