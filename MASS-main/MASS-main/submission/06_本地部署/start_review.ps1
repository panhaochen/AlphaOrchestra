param(
    [switch]$Online
)

$ErrorActionPreference = "Stop"
$deployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = Resolve-Path (Join-Path $deployDir "..\..")

Write-Host "AlphaOrchestra local review entry" -ForegroundColor DarkRed
Write-Host "Project directory: $repoDir"

if (-not $Online) {
    Write-Host "Mode: offline evidence review (no network or API required)" -ForegroundColor Green
    $submissionDir = Split-Path -Parent $deployDir
    $evidenceDir = Get-ChildItem $submissionDir -Directory | Where-Object { $_.Name -like "05_*" } | Select-Object -First 1 -ExpandProperty FullName
    $resultCsv = Get-ChildItem $evidenceDir -Recurse -Filter "*.csv" | Select-Object -First 1 -ExpandProperty FullName
    Write-Host ("Evidence directory: " + $evidenceDir)
    Write-Host "Key results" -ForegroundColor Cyan
    Write-Host "- 2024Q2 10d RankIC: 0.018666 vs 0.009591 (+94.6%)"
    Write-Host "- 2024Q2 10d RankICIR: 0.188458 vs 0.122253 (+54.1%)"
    Write-Host "- Daily activation: 14.14 / 64 candidates (77.9% fewer than full activation)"
    Write-Host "- 2026 extended 10d RankIC: 0.011121; RankICIR: 0.131179"
    if (Test-Path $resultCsv) {
        Write-Host ""
        Write-Host "Metric snapshot (first 10 rows)" -ForegroundColor Cyan
        Import-Csv $resultCsv | Select-Object -First 10 | Format-Table -AutoSize
    } else {
        Write-Warning "Metric snapshot was not found. Check the 05 evidence folder."
    }
    Write-Host ""
    Write-Host "Next: open 05 evidence/00 index, then 03 validation reports." -ForegroundColor Green
    exit 0
}

Write-Host "Mode: online reproduction environment check" -ForegroundColor Yellow
python -c "import sys; print('Python:', sys.executable); print('Version:', sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Python is not available." }

$required = @("MASS_LLM_API_KEY", "MASS_LLM_BASE_URL", "MASS_CHAT_MODEL")
$missing = @()
foreach ($name in $required) {
    if (-not [Environment]::GetEnvironmentVariable($name)) { $missing += $name }
}
if ($missing.Count -gt 0) {
    Write-Host "Missing environment variables: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "See deployment guide and the repository .env.example file."
    exit 2
}

python -c "import backoff, pandas, numpy, pyarrow; print('Core dependencies: OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Core dependencies are missing. Activate the mass environment and run pdm install." -ForegroundColor Red
    exit 3
}

Write-Host "Online environment check passed. See the deployment guide for the smoke command." -ForegroundColor Green
