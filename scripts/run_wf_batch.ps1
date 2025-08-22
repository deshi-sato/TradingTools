param(
  [string]$Py = "scripts\walkforward_from_codes.py"
)

$cfgs = @(
  @{T=60;  V=30},
  @{T=90;  V=30},
  @{T=180; V=30}
)

$out = "data\analysis"
New-Item -ItemType Directory -Force -Path $out | Out-Null

foreach($c in $cfgs){
  $env:WF_TRAIN = $c.T
  $env:WF_TEST  = $c.V

  Write-Host ">>> Run TRAIN=$($c.T) TEST=$($c.V)"
  python $Py | Out-Host
  $code = $LASTEXITCODE

  $tag = "T$($c.T)_V$($c.V)"

  if($code -ne 0){
    Write-Warning "Python exited with code $code for $tag. Skipping rename."
  } else {
    if(Test-Path "$out\wf_results.csv"){     Move-Item "$out\wf_results.csv"     "$out\wf_results_$tag.csv"     -Force }
    if(Test-Path "$out\wf_daily.csv"){       Move-Item "$out\wf_daily.csv"       "$out\wf_daily_$tag.csv"       -Force }
    if(Test-Path "$out\wf_cum.png"){         Move-Item "$out\wf_cum.png"         "$out\wf_cum_$tag.png"         -Force }
    if(Test-Path "$out\wf_window_bars.png"){ Move-Item "$out\wf_window_bars.png" "$out\wf_window_bars_$tag.png" -Force }
  }

  # 環境変数をクリーンにして次のループへ
  Remove-Item Env:\WF_TRAIN -ErrorAction SilentlyContinue
  Remove-Item Env:\WF_TEST  -ErrorAction SilentlyContinue
}

Write-Host "Done. Outputs under data\\analysis\\*_(T*_V*).{csv,png}"
