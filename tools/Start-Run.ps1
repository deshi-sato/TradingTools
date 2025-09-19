param([ValidateSet("DRYRUN","PAPER","LIVE")][string]$Mode="PAPER",[int]$Minutes=20)
$env:MODE=$Mode
$env:RUN_MINUTES="$Minutes"
if ($Mode -eq "DRYRUN") { $env:USE_FAKE_TICKS='1' } else { $env:USE_FAKE_TICKS='0' }
py -m orchestrate.run_intraday
