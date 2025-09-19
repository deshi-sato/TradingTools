$env:KABU_BASE_URL = $env:KABU_BASE_URL ? $env:KABU_BASE_URL : 'http://localhost:18080'
if (-not $env:KABU_API_PW -and -not $env:KABU_API_PASSWORD) {
  throw "KABU_API_PW (or KABU_API_PASSWORD) is not set."
}
py .\scripts\fetch_ranking.py
Get-Content .\data\perma_regulars.csv -First 8
