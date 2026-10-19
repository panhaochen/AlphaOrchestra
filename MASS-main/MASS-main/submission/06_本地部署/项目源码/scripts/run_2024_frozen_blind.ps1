$ErrorActionPreference = "Stop"

if (-not $env:MASS_LLM_API_KEY) {
    throw "MASS_LLM_API_KEY is not set in this terminal. Set it before running the frozen blind test."
}

$env:MASS_LLM_CACHE_MODE = "readwrite"
$pythonExe = if ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable was not found: $pythonExe"
}
& $pythonExe -c "import backoff, pandas, numpy; print('Using Python:', __import__('sys').executable)"
if ($LASTEXITCODE -ne 0) {
    throw "The active Python environment is missing required packages. Activate the mass environment first."
}

$runId = "csi300_2024_frozen_blind_seed42"
$runPath = Join-Path "stock_disagreement/res/runs" $runId
if (Test-Path -LiteralPath $runPath) {
    throw "Frozen run directory already exists: $runPath. Do not overwrite or retune the blind run."
}

$modelArgs = @(
    "--random_seed", "42",
    "--num_investor_type", "4",
    "--num_agents_per_investor", "1",
    "--stock_pool", "csi_300",
    "--stock_num", "50",
    "--selected_stock_num", "5",
    "--start_date", "20231009",
    "--end_date", "20241231",
    "--evaluation_start_date", "20240102",
    "--no-use_agent_distribution_modification",
    "--use_adaptive_router",
    "--no-use_heterogeneous_agent_pool",
    "--router_top_k", "3",
    "--router_diversity_lambda", "0.35",
    "--router_cost_lambda", "0.05",
    "--use_disagreement_diagnosis",
    "--abstention_threshold", "0.7",
    "--profile_maturity_days", "15",
    "--use_contextual_aggregator",
    "--no-use_conditional_debate",
    "--max_agent_calls_per_date", "3",
    "--max_debate_calls_per_date", "0",
    "--max_tokens_per_date", "150000",
    "--signal_alpha", "0.7",
    "--signal_alpha_1d", "0.7",
    "--signal_alpha_5d", "0.7",
    "--signal_alpha_10d", "0.7",
    "--no-use_adaptive_signal_alpha",
    "--run_id", $runId
)

& $pythonExe -m stock_disagreement.main @modelArgs
if ($LASTEXITCODE -ne 0) { throw "Frozen signal run failed with exit code $LASTEXITCODE" }

$backtestArgs = @(
    "-m", "scripts.backtest_run",
    "--run_id", $runId,
    "--signal_column", "Signal_10d",
    "--evaluation_start_date", "20240102",
    "--top_n", "20",
    "--rebalance_every", "10",
    "--industry_balanced",
    "--min_value_quantile", "0.2",
    "--residualize_signal",
    "--market_regime_control",
    "--regime_lookback", "60",
    "--risk_off_exposure", "0.5",
    "--output_tag", "frozen_commercial_portfolio"
)

& $pythonExe @backtestArgs
if ($LASTEXITCODE -ne 0) { throw "Frozen portfolio backtest failed with exit code $LASTEXITCODE" }

Write-Host "Frozen 2024 blind test completed: $runPath"
