from stock_disagreement import Modality, StockDisagreementAgent
import pandas as pd
import numpy as np
from tqdm import tqdm 
from typing import Any, List
from dataclasses import asdict
import random
from stock_disagreement import InvestmentAnalyzer, BaseOptimizer, SimulatedAnnealingOptimizer
import threading
import concurrent.futures
import itertools
import copy
from scipy.stats import percentileofscore
from stock_disagreement.config import first_existing_data_path
from stock_disagreement.signals import HorizonTarget, RollingHorizonAlphaCalibrator, add_horizon_signals
from stock_disagreement.exp.orchestration import (
    AgentProfileStore,
    ContextualAggregator,
    ConditionalDebateCoordinator,
    DailyInferenceBudget,
    InferenceBudgetController,
    DiversityCostAwareRouter,
    MarketStateEncoder,
    StructuredDisagreementDiagnoser,
)
class StockDisagreementTrainer():
    INVESTOR_PERSONAS = (
        "short-horizon momentum investor; emphasize recent price strength and fast reversal risk",
        "contrarian value investor; prefer valuation support and mean reversion",
        "quality-growth investor; emphasize durable growth, profitability, and balance-sheet quality",
        "defensive low-risk investor; emphasize stability, drawdown control, and liquidity",
        "industry-rotation investor; emphasize relative sector strength and breadth",
        "event-driven investor; emphasize news catalysts and near-term repricing",
        "macro-sensitive investor; emphasize rates, inflation, policy, and market regime",
        "liquidity-aware investor; avoid crowded or difficult-to-trade names",
        "medium-horizon trend investor; require confirmation across multiple observations",
        "deep-value investor; tolerate short-term weakness when valuation dispersion is large",
        "earnings-revision investor; prioritize improving fundamentals and expectation changes",
        "risk-parity investor; balance conviction against volatility and concentration",
        "sentiment investor; emphasize positioning, attention, and behavioral overreaction",
        "cross-sectional factor investor; compare candidates consistently rather than in isolation",
        "capital-preservation investor; demand strong evidence before taking exposure",
        "balanced generalist investor; combine price, industry, and fundamental evidence",
    )
    agent_distributions: dict[int, float] = {}
    date_agent_distributions: dict[int, Any] = {}
    def __init__(self, 
                 num_investor_type: int, 
                 num_agents_per_investor: int,
                 stock_selector_for_per_investor: int,
                 stock_pool: pd.DataFrame,
                 stock_labels: pd.DataFrame,
                 stock_num: int,
                 look_back_window: int,
                 start_date: int | None = None,
                 end_date: int | None = None,
                 use_prev_stock: bool = False,
                 use_self_reflection: bool = False,
                 use_macro_data: bool = False,
                 use_agent_distribution_modification: bool = False,
                 optimizer_look_back_window: int = 2,
                 data_leakage: bool = False,
                 use_adaptive_router: bool = False,
                 router_top_k: int = 0,
                 router_min_k: int = 1,
                 router_max_k: int = 0,
                 router_diversity_lambda: float = 0.35,
                 router_cost_lambda: float = 0.05,
                 router_exploration_lambda: float = 0.0,
                 router_family_diversity_lambda: float = 0.0,
                 use_disagreement_diagnosis: bool = False,
                 abstention_threshold: float = 0.45,
                 profile_maturity_days: int = 15,
                 profile_label_column: str = "1_15_labelB",
                 use_contextual_aggregator: bool = False,
                 aggregator_performance_lambda: float = 1.0,
                 aggregator_confidence_lambda: float = 0.6,
                 aggregator_evidence_lambda: float = 0.15,
                 aggregator_agreement_lambda: float = 0.4,
                 aggregator_cost_lambda: float = 0.05,
                 use_conditional_debate: bool = False,
                 max_debate_agents_per_stock: int = 2,
                 max_debate_stocks_per_date: int = 5,
                 max_agent_calls_per_date: int = 0,
                 max_debate_calls_per_date: int = 0,
                 max_tokens_per_date: int = 0,
                 max_latency_seconds_per_date: float = 0.0,
                 max_monetary_cost_per_date: float = 0.0,
                 signal_alpha: float = 0.5,
                 signal_alpha_1d: float | None = None,
                 signal_alpha_5d: float | None = None,
                 signal_alpha_10d: float | None = None,
                 use_adaptive_signal_alpha: bool = False,
                 signal_alpha_grid_step: float = 0.1,
                 signal_alpha_min_observations: int = 20,
                 signal_alpha_lookback_observations: int = 60,
                 signal_alpha_shrinkage: float = 0.5,
                 random_seed: int = 42,
                 use_heterogeneous_agent_pool: bool = False,
                 ):
        self.interval = 4
        self.agent_distributions = {}
        self.date_agent_distributions = {}
        self.date_market_states = {}
        self.date_selected_agents = {}
        self.date_routing_traces = {}
        self.date_diagnostics = {}
        self.date_aggregation_traces = {}
        self.date_debate_traces = {}
        self.date_agent_profile_stores = {}
        self.num_investor_type = num_investor_type
        self.stock_selector_for_per_investor = stock_selector_for_per_investor
        self.num_agents_per_investor = num_agents_per_investor
        self.stock_num = stock_num
        self.random_seed = int(random_seed)
        self.random = random.Random(self.random_seed)
        self.use_heterogeneous_agent_pool = bool(use_heterogeneous_agent_pool)
        if not 0.0 <= signal_alpha <= 1.0:
            raise ValueError("signal_alpha must be between 0 and 1.")
        self.signal_alpha = float(signal_alpha)
        self.horizon_signal_alphas = {
            "1d": self._validated_optional_alpha(signal_alpha_1d, self.signal_alpha),
            "5d": self._validated_optional_alpha(signal_alpha_5d, self.signal_alpha),
            "10d": self._validated_optional_alpha(signal_alpha_10d, self.signal_alpha),
        }
        self.stock_pool = stock_pool
        self.stock_labels = stock_labels
        self.look_back_window = look_back_window
        self.stock_labels = self.stock_labels.merge(self.stock_pool[["Stock", "Date"]], on=["Stock", "Date"])
        self.optimizer_look_back_window = optimizer_look_back_window
        self.use_agent_distribution_modification = use_agent_distribution_modification

        self.use_prev_stock = use_prev_stock
        if start_date is not None:
            self.start_date = start_date
        else:
            self.start_date = stock_pool["Date"].min()
        if end_date is not None:
            self.end_date = end_date
        else:
            self.end_date = stock_pool["Date"].max()
        self.stock_pool = self.stock_pool[(self.stock_pool["Date"] >= self.start_date) & (self.stock_pool["Date"] <= self.end_date)]

        self.stock_labels = self.stock_labels[(self.stock_labels["Date"] >= self.start_date) & (self.stock_labels["Date"] <= self.end_date)]
        self.dates = sorted(self.stock_pool[(self.stock_pool["Date"] >= self.start_date) & (self.stock_pool["Date"]<= self.end_date)]["Date"].unique().tolist())
        
        self.agents: List[StockDisagreementAgent] = []
        self.use_self_reflection = use_self_reflection
        self.use_macro_data = use_macro_data
        self.use_adaptive_router = use_adaptive_router
        self.use_disagreement_diagnosis = use_disagreement_diagnosis
        self.profile_maturity_days = max(int(profile_maturity_days), 1)
        self.profile_label_column = profile_label_column
        self.use_adaptive_signal_alpha = bool(use_adaptive_signal_alpha)
        self.signal_alpha_calibrator = RollingHorizonAlphaCalibrator(
            targets=[
                HorizonTarget("1d", "1_15_labelB", self.profile_maturity_days),
                HorizonTarget("5d", "5_15_labelB", self.profile_maturity_days),
                HorizonTarget("10d", "10_15_labelB", self.profile_maturity_days),
            ],
            fallback_alphas=self.horizon_signal_alphas,
            grid_step=signal_alpha_grid_step,
            min_observations=signal_alpha_min_observations,
            lookback_observations=signal_alpha_lookback_observations,
            shrinkage=signal_alpha_shrinkage,
        )
        self.use_contextual_aggregator = use_contextual_aggregator
        self.use_conditional_debate = use_conditional_debate
        budget_policy = DailyInferenceBudget(
            max_agent_calls=max_agent_calls_per_date,
            max_debate_calls=max_debate_calls_per_date,
            max_total_tokens=max_tokens_per_date,
            max_latency_seconds=max_latency_seconds_per_date,
            max_monetary_cost=max_monetary_cost_per_date,
        )
        self.budget_controller = InferenceBudgetController(budget_policy)
        self.use_inference_budget = any([
            max_agent_calls_per_date,
            max_debate_calls_per_date,
            max_tokens_per_date,
            max_latency_seconds_per_date,
            max_monetary_cost_per_date,
        ])
        self.agent_profiles = AgentProfileStore()
        self.router = DiversityCostAwareRouter(
            top_k=router_top_k,
            min_agents=router_min_k,
            max_agents=router_max_k,
            diversity_lambda=router_diversity_lambda,
            cost_lambda=router_cost_lambda,
            profile_store=self.agent_profiles,
            exploration_lambda=router_exploration_lambda,
            family_diversity_lambda=router_family_diversity_lambda,
        )
        self.disagreement_diagnoser = StructuredDisagreementDiagnoser(
            abstention_threshold=abstention_threshold,
        )
        self.contextual_aggregator = ContextualAggregator(
            profile_store=self.agent_profiles,
            performance_lambda=aggregator_performance_lambda,
            confidence_lambda=aggregator_confidence_lambda,
            evidence_lambda=aggregator_evidence_lambda,
            agreement_lambda=aggregator_agreement_lambda,
            cost_lambda=aggregator_cost_lambda,
        )
        self.debate_coordinator = ConditionalDebateCoordinator(
            max_agents_per_stock=max_debate_agents_per_stock,
            max_stocks_per_date=max_debate_stocks_per_date,
        )
        try:
            self.news_info = pd.read_parquet(first_existing_data_path("wind-financial-news-info.parq", "financial-news-info.parq"))
            self.news_relationship = pd.read_parquet(first_existing_data_path("wind-financial-news-relationship.parq", "financial-news-relationship.parq"))
            self.news_info["Date"] = self.news_info["Date"].astype("int32")
            self.news_relationship["Date"] = self.news_relationship["Date"].astype("int32")
        except FileNotFoundError:
            self.news_info = pd.DataFrame(columns=["Date", "NewsId", "NewsTitle"])
            self.news_relationship = pd.DataFrame(columns=["Date", "Stock", "NewsId"])
        try:
            market_data = pd.read_parquet(first_existing_data_path("base_data.parq"))
        except FileNotFoundError:
            market_data = pd.DataFrame(columns=["Stock", "Date", "Close"])
        self.market_state_encoder = MarketStateEncoder(
            self.stock_pool,
            market_data=market_data,
            news_relationship=self.news_relationship,
            lookback_days=max(self.look_back_window, 20),
        )
        self._init_agents() 
        self.data_leakage = data_leakage
        self.optimzer = SimulatedAnnealingOptimizer(look_back_window=optimizer_look_back_window)

    @staticmethod
    def _validated_optional_alpha(value: float | None, fallback: float) -> float:
        alpha = fallback if value is None else float(value)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Horizon signal alpha must be between 0 and 1.")
        return alpha

    def _init_agents(self,) -> None:

        def cal_pe_quantile(data: pd.DataFrame, look_back_year: int = 10) -> pd.DataFrame:
            data_copy = data.copy()
            data_copy["dt"] = pd.to_datetime(data_copy["Date"].astype(str), format="%Y%m%d")
            data_copy.sort_values("dt", inplace=True)
            data_copy.reset_index(drop=True, inplace=True)  

            min_periods = look_back_year * 250
            data_copy["quantile"] = data_copy["Value"].rolling( window= min_periods, closed="both").apply(lambda x: percentileofscore(x, x.iloc[-1]), raw=False)    
            data_copy["quantile"] = data_copy["quantile"].round(1)   
            data_copy["Date"] = data_copy["dt"].dt.strftime("%Y%m%d").astype(int)  
            return data_copy[["Date", "Value", "quantile"]]  


        loan_rate = pd.read_csv(first_existing_data_path("macro_data/China_1-Year_Loan_Prime_Rate_LPR.csv"))
        cpi = pd.read_csv(first_existing_data_path("macro_data/China_CPI_YoY_Current_Month.csv"))
        csi_300_pe = pd.read_csv(first_existing_data_path("macro_data/csi_300_pe_ttm.csv"))
        csi_300_pe  = cal_pe_quantile(csi_300_pe)
        market_sentiment_index = pd.read_csv(first_existing_data_path("macro_data/Market_Sentiment_Index.csv"))
        yield_on_China_bonds = pd.read_csv(first_existing_data_path("macro_data/yield_on_China_10_year_government_bonds.csv")) 
        modalities = self._available_modalities()
        heterogeneous_modalities = self._heterogeneous_modalities(modalities)
        for num in tqdm(range(self.num_investor_type), desc="Init agents"):
            shared_result = self._sample_modality(modalities)
            for i in range(self.num_agents_per_investor):
                agent_id = len(self.agents)
                result = (
                    heterogeneous_modalities[agent_id % len(heterogeneous_modalities)]
                    if self.use_heterogeneous_agent_pool
                    else shared_result
                )
                persona = self._agent_persona(num, i) if self.use_heterogeneous_agent_pool else ""
                current_agent = StockDisagreementAgent(stock_num= self.stock_num, 
                                                   stock_pool =self.stock_pool,
                                                   stock_labels = self.stock_labels,
                                                   is_self_reflective =True,
                                                   max_reflective_times =10, 
                                                   start_date = self.start_date,
                                                   end_date =self.end_date,
                                                   modality = result,
                                                   use_prev_stock = self.use_prev_stock,
                                                   cpi = cpi,
                                                   csi_300_pe = csi_300_pe,
                                                   loan_rate = loan_rate,
                                                   market_sentiment_index = market_sentiment_index,
                                                   yield_on_China_bonds = yield_on_China_bonds,
                                                   use_self_reflection = self.use_self_reflection,
                                                   use_macro_data = self.use_macro_data,
                                                   agent_id=agent_id if self.use_heterogeneous_agent_pool else num,
                                                   investor_type=num,
                                                   persona=persona) 
                current_agent.random_seed = self.random_seed + agent_id
                current_agent.prepare_data_source(self.news_info, self.news_relationship)
                current_agent.estimated_cost = 1.0 + 0.1 * max(int(result).bit_count(), 1)
                # if not self.use_macro_data:
                #     current_agent.generate_strategy_and_stock_selector()
                # else:
                #     current_agent.generate_strategy_and_stock_selector(self.start_date)
                self.agent_distributions[current_agent.agent_type] = 1.0
                self.agents.append(current_agent) 

    def _sample_modality(self, modalities: list[Modality]) -> Modality:
        selected = self.random.sample(
            modalities,
            k=self.random.randint(1, min(3, len(modalities))),
        )
        result = Modality(0)
        for modality in selected:
            result |= modality
        return result

    def _heterogeneous_modalities(self, modalities: list[Modality]) -> list[Modality]:
        combinations = []
        for size in range(1, min(3, len(modalities)) + 1):
            for selected in itertools.combinations(modalities, size):
                result = Modality(0)
                for modality in selected:
                    result |= modality
                combinations.append(result)
        self.random.shuffle(combinations)
        return combinations

    def _agent_persona(self, investor_type: int, replica_index: int) -> str:
        base = self.INVESTOR_PERSONAS[investor_type % len(self.INVESTOR_PERSONAS)]
        return (
            f"{base}. You are independent replica {replica_index + 1}; "
            "form your own ranking from the supplied evidence and do not imitate consensus."
        )

    def _available_modalities(self) -> list[Modality]:
        modalities = []
        checks = [
            (Modality.BASE_DATA, "base_data.parq"),
            (Modality.CROSS_INDUSTRY_LABEL, "industry_ret.parq"),
            (Modality.PRICE_FEATURE, "price_feature.parq", "sub_fudamental_data.parq"),
        ]
        if not self.news_info.empty and not self.news_relationship.empty:
            modalities.append(Modality.NEWS)
        for modality, *files in checks:
            try:
                first_existing_data_path(*files)
                modalities.append(modality)
            except FileNotFoundError:
                pass
        if not modalities:
            raise FileNotFoundError("No usable modality data files were found in the dataset directory.")
        return modalities


    
    def run(self) -> pd.DataFrame:  
        def _process_agent(agent:StockDisagreementAgent, date: int, stock_selector):  
            agent.invest(date, stock_selector)

        def _selected_agents_for_date(date: int) -> list[StockDisagreementAgent]:
            if self.use_adaptive_router:
                market_state = self.market_state_encoder.encode(date)
                selected_agents = self.router.select_agents(self.agents, market_state)
                self.date_market_states[date] = market_state
            else:
                selected_agents = list(self.agents)
                market_state = self.market_state_encoder.encode(date)
            if self.use_inference_budget:
                admitted_agents = []
                for agent in selected_agents:
                    profile = self.agent_profiles.get(int(agent.agent_type))
                    estimate = self.budget_controller.estimate_for_agent(profile)
                    if self.budget_controller.admit(date, "agent", estimate):
                        admitted_agents.append(agent)
                selected_agents = admitted_agents
            self.date_selected_agents[date] = [agent.agent_type for agent in selected_agents]
            if self.use_adaptive_router:
                self.date_routing_traces[date] = self.router.selection_trace(
                    self.agents, selected_agents, market_state
                )
            return selected_agents

        def _update_mature_profiles(current_date: int) -> None:
            prediction_date = self.agent_profiles.matured_prediction_date(
                self.dates, current_date, self.profile_maturity_days
            )
            if prediction_date is None:
                return
            if prediction_date in self.agent_profiles.evaluated_dates:
                return
            analyzer = self.agents[0].investment_analyzer
            label_column = self.profile_label_column
            labels = self.stock_labels[self.stock_labels["Date"] == prediction_date]
            if label_column not in labels or prediction_date not in analyzer.data:
                return
            truth = labels.set_index("Stock")[label_column]
            for agent_type, stock_data in analyzer.data[prediction_date].items():
                scores = pd.Series({
                    stock: values["score"] / values["num_investors"]
                    if values["num_investors"] else 0.0
                    for stock, values in stock_data.items()
                }, dtype=float)
                joined = pd.concat([scores.rename("score"), truth.rename("label")], axis=1).dropna()
                if len(joined) >= 3 and joined["score"].nunique() > 1 and joined["label"].nunique() > 1:
                    rank_ic = joined["score"].rank().corr(joined["label"].rank())
                    self.agent_profiles.update_performance(int(agent_type), float(rank_ic))
            self.agent_profiles.evaluated_dates.add(prediction_date)

        def _update_cost_profiles(date: int) -> None:
            if date in self.agent_profiles.cost_dates:
                return
            analyzer = self.agents[0].investment_analyzer
            for agent_type, calls in analyzer.call_metrics.get(date, {}).items():
                for metrics in calls:
                    self.agent_profiles.update_call(int(agent_type), metrics)
                    if self.use_inference_budget:
                        self.budget_controller.record_actual(date, "agent", metrics)
            self.agent_profiles.cost_dates.add(date)
        
        if not self.use_agent_distribution_modification:
            if self.use_adaptive_router or self.use_inference_budget:
                for date in self.dates:
                    if self.use_adaptive_router:
                        _update_mature_profiles(date)
                    selected_agents = _selected_agents_for_date(date)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(len(selected_agents), 1))) as executor:
                        futures = [
                            executor.submit(_process_agent, agent, date, self.stock_selector_for_per_investor)
                            for agent in selected_agents
                        ]
                        for future in tqdm(
                            concurrent.futures.as_completed(futures),
                            total=len(futures),
                            desc=f"Processing routed agents on {date}"
                        ):
                            future.result()
                    _update_cost_profiles(date)
                    # Freeze the online information set at this date. Signal
                    # calculation happens in a later pass and must not read a
                    # profile containing observations from future dates.
                    self.date_agent_profile_stores[date] = copy.deepcopy(self.agent_profiles)
                    self.date_agent_distributions[date] = self.router.active_distributions(
                        selected_agents, self.agent_distributions
                    )
            else:
                total_agent_tasks = len(self.dates) * len(self.agents) 
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  
                    futures = [  
                        executor.submit(_process_agent, agent, date, self.stock_selector_for_per_investor)  
                        for date in self.dates  
                        for agent in self.agents  
                    ]    
                    for future in tqdm(  
                        concurrent.futures.as_completed(futures),  
                        total=total_agent_tasks,  
                        desc="Processing agents"  
                    ):   
                        future.result() 
        else:
            for date in self.dates:
                if self.use_adaptive_router:
                    _update_mature_profiles(date)
                selected_agents = _selected_agents_for_date(date)
                total_tasks = len(selected_agents)
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(total_tasks, 1))) as executor:
                    futures = [
                        executor.submit(_process_agent, agent, date, self.stock_selector_for_per_investor)
                        for agent in selected_agents
                    ]
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=total_tasks,
                        desc= f"Processing agent on {date}"
                    ):
                        future.result()
                if self.use_adaptive_router or self.use_inference_budget:
                    _update_cost_profiles(date)
                    self.date_agent_profile_stores[date] = copy.deepcopy(self.agent_profiles)
                res = self.optimzer.is_adjusted(self.dates, date)
                if res > 0:
                    self.date_agent_distributions[date] = self.agent_distributions.copy()
                    best_distributions = self.optimzer.optimize(investment_analyzer=self.agents[0].investment_analyzer,
                                                                dates = self.dates,
                                                                start_date=res,
                                                                current_date= date,
                                                                stock_labels= self.stock_labels,
                                                                stock_pool= self.stock_pool,
                                                                distribution=self.agent_distributions)
                    self.agent_distributions = best_distributions
                    if self.data_leakage:
                        self.date_agent_distributions[date] = best_distributions
                else:
                    self.date_agent_distributions[date] = self.agent_distributions
                if self.use_adaptive_router:
                    self.date_agent_distributions[date] = self.router.active_distributions(
                        selected_agents, self.date_agent_distributions[date]
                    )
                
        investment_analyzer = self.agents[0].investment_analyzer  
        investment_res = self.stock_pool[(self.stock_pool["Date"] >= self.start_date) & (self.stock_pool["Date"] <= self.end_date) ].copy()  


        def _calc_signal(date: int, agent_distributions: dict[int, float]): 
            current_pool = self.stock_pool[self.stock_pool["Date"] == date]["Stock"].tolist()  
            diagnostics = None
            contextual_weights = None
            raw_opinions = None
            market_state = None
            if self.use_disagreement_diagnosis or self.use_contextual_aggregator or self.use_conditional_debate:
                market_state = self.date_market_states.get(date) or self.market_state_encoder.encode(date)
                raw_opinions = investment_analyzer.collect_stock_opinions(date, current_pool, agent_distributions)
            if self.use_disagreement_diagnosis:
                diagnostics = self.disagreement_diagnoser.diagnose(raw_opinions, market_state)
                self.date_diagnostics[date] = diagnostics
            if self.use_conditional_debate:
                if diagnostics is None:
                    diagnostics = self.disagreement_diagnoser.diagnose(raw_opinions, market_state)
                    self.date_diagnostics[date] = diagnostics
                debate_plan = self.debate_coordinator.build_plan(raw_opinions, diagnostics)
                debate_trace = []
                debate_tasks = []
                for item in debate_plan:
                    stock = item["stock"]
                    values = raw_opinions[stock]
                    for agent_type, opinion_index in zip(item["agent_types"], item["opinion_indices"]):
                        representative = next(
                            (agent for agent in self.agents if int(agent.agent_type) == int(agent_type)),
                            None,
                        )
                        if representative is None:
                            continue
                        prior_opinion = {
                            key: values[key][opinion_index]
                            for key in ["scores", "confidences", "horizons", "evidence", "reasoning"]
                            if key in values and opinion_index < len(values[key])
                        }
                        profile = self.agent_profiles.get(int(agent_type))
                        estimate = self.budget_controller.estimate_for_agent(profile)
                        # Admission remains serial so concurrent workers cannot race the
                        # daily ledger. Network-bound debate calls are executed below.
                        if self.use_inference_budget and not self.budget_controller.admit(date, "debate", estimate):
                            debate_trace.append({
                                "stock": stock,
                                "agent_type": int(agent_type),
                                "success": False,
                                "pre_severity": diagnostics[stock]["severity"],
                                "error_type": "BudgetRejected",
                            })
                            continue
                        debate_tasks.append((
                            len(debate_tasks), stock, int(agent_type), representative,
                            diagnostics[stock], prior_opinion,
                        ))

                def _run_debate(task):
                    task_index, stock, agent_type, representative, stock_diagnostics, prior_opinion = task
                    revision, metrics = representative.debate_stock(
                        stock, stock_diagnostics, prior_opinion
                    )
                    return task_index, stock, agent_type, revision, metrics

                debate_results = []
                if debate_tasks:
                    configured_limit = self.budget_controller.policy.max_debate_calls
                    concurrency_limit = configured_limit if configured_limit > 0 else len(debate_tasks)
                    max_workers = min(len(debate_tasks), concurrency_limit, 8)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_tasks = {
                            executor.submit(_run_debate, task): task for task in debate_tasks
                        }
                        for future in concurrent.futures.as_completed(future_tasks):
                            task = future_tasks[future]
                            task_index, stock, agent_type = task[:3]
                            try:
                                debate_results.append(future.result())
                            except Exception as exc:
                                debate_trace.append({
                                    "_task_index": task_index,
                                    "stock": stock,
                                    "agent_type": agent_type,
                                    "success": False,
                                    "pre_severity": diagnostics[stock]["severity"],
                                    "error_type": type(exc).__name__,
                                })

                # Apply revisions on the main thread and in plan order. This keeps
                # aggregation reproducible even though requests finish out of order.
                for task_index, stock, agent_type, revision, metrics in sorted(debate_results):
                    if self.use_inference_budget and metrics:
                        self.budget_controller.record_actual(date, "debate", metrics)
                    investment_analyzer.apply_debate_revision(
                        date, agent_type, stock, revision.to_dict(), call_metrics=metrics,
                    )
                    debate_trace.append({
                        "_task_index": task_index,
                        "stock": stock,
                        "agent_type": agent_type,
                        "success": True,
                        "pre_severity": diagnostics[stock]["severity"],
                        **revision.to_dict(),
                    })
                debate_trace.sort(key=lambda entry: entry.get("_task_index", -1))
                for entry in debate_trace:
                    entry.pop("_task_index", None)
                if debate_trace:
                    raw_opinions = investment_analyzer.collect_stock_opinions(
                        date, current_pool, agent_distributions
                    )
                    diagnostics = self.disagreement_diagnoser.diagnose(raw_opinions, market_state)
                    for entry in debate_trace:
                        entry["post_severity"] = diagnostics[entry["stock"]]["severity"]
                    self.date_diagnostics[date] = diagnostics
                self.date_debate_traces[date] = debate_trace
            if self.use_contextual_aggregator:
                contextual_weights, aggregation_trace = self.contextual_aggregator.calculate_weights(
                    raw_opinions,
                    market_state,
                    profile_store=self.date_agent_profile_stores.get(date),
                )
                self.date_aggregation_traces[date] = aggregation_trace
            res = investment_analyzer.calculate_stock_disagreement_score(
                date,
                current_pool,
                agent_distributions=agent_distributions,
                alpha=self.signal_alpha,
                disagreement_diagnostics=diagnostics,
                contextual_weights=contextual_weights,
            )  
            return date, res   
        
        date_signal_results = {}  
        if not self.use_agent_distribution_modification and not self.use_adaptive_router and not self.use_conditional_debate and not self.use_inference_budget:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  
                date_futures = {executor.submit(_calc_signal, date, self.agent_distributions): date for date in self.dates}  
                for future in tqdm(  
                    concurrent.futures.as_completed(date_futures),  
                    total=len(self.dates),  
                    desc="Calculating signals"  
                    ):  
                    date, result = future.result()  
                    date_signal_results[date] = result 
        else:
            for date in tqdm(self.dates, desc = "calc investment res in optimzing mode."):
                distributions = self.date_agent_distributions.get(date, self.agent_distributions)
                dt, res = _calc_signal(date, distributions)
                date_signal_results[dt] = res

        data_list = []
        for date, res in sorted(date_signal_results.items()):
            for stock, values in res.items():
                data_list.append([date, stock, values[0], values[1], values[2], values[3], values[4], values[5]])
        result_df = pd.DataFrame(
            data_list,
            columns=[
                'Date',
                'Stock',
                'Signal',
                'Signal_mean',
                'Signal_std',
                'ExposureControl',
                'DisagreementSeverity',
                'DisagreementType',
            ],
        )
        if self.use_adaptive_signal_alpha:
            calibrated_frames = []
            label_columns = ["1_15_labelB", "5_15_labelB", "10_15_labelB"]
            for date, date_frame in result_df.groupby("Date", sort=True):
                alphas = self.signal_alpha_calibrator.select(int(date))
                calibrated_frames.append(add_horizon_signals(date_frame, alphas))
                calibration_frame = date_frame.merge(
                    self.stock_labels[self.stock_labels["Date"] == date][["Stock"] + label_columns],
                    on="Stock",
                    how="left",
                )
                self.signal_alpha_calibrator.observe(int(date), calibration_frame)
            result_df = pd.concat(calibrated_frames, ignore_index=True)
        else:
            result_df = add_horizon_signals(result_df, self.horizon_signal_alphas)
        investment_res = pd.merge(investment_res, result_df, on=['Date', 'Stock'], how='left')
        for date, res in date_signal_results.items():  
            for stock, values in res.items():  
                indexer = (investment_res["Date"] == date) & (investment_res["Stock"] == stock)  
                investment_res.loc[indexer, "Signal"] = values[0]  
                investment_res.loc[indexer, "Signal_mean"] = values[1]  
                investment_res.loc[indexer, "Signal_std"] = values[2]  
                investment_res.loc[indexer, "ExposureControl"] = values[3]  
                investment_res.loc[indexer, "DisagreementSeverity"] = values[4]  
                investment_res.loc[indexer, "DisagreementType"] = values[5]  
        return investment_res

    def audit_frames(self) -> dict[str, pd.DataFrame]:
        """Return reproducible audit tables without writing files implicitly."""
        market_states = [asdict(state) for _, state in sorted(self.date_market_states.items())]
        routing = [
            {"date": date, **item}
            for date, items in sorted(self.date_routing_traces.items())
            for item in items
        ]
        diagnostics = [
            {"date": date, "stock": stock, **detail}
            for date, stocks in sorted(self.date_diagnostics.items())
            for stock, detail in stocks.items()
        ]
        llm_calls = [
            {"date": date, "agent_type": int(agent_type), **metrics}
            for date, agent_metrics in sorted(self.agents[0].investment_analyzer.call_metrics.items())
            for agent_type, calls in agent_metrics.items()
            for metrics in calls
        ] if self.agents else []
        aggregation = [
            {"date": date, **item}
            for date, items in sorted(self.date_aggregation_traces.items())
            for item in items
        ]
        debates = [
            {"date": date, **item}
            for date, items in sorted(self.date_debate_traces.items())
            for item in items
        ]
        return {
            "market_states": pd.DataFrame(market_states),
            "routing": pd.DataFrame(routing),
            "diagnostics": pd.DataFrame(diagnostics),
            "llm_calls": pd.DataFrame(llm_calls),
            "agent_profiles": pd.DataFrame(self.agent_profiles.records()),
            "aggregation": pd.DataFrame(aggregation),
            "debates": pd.DataFrame(debates),
            "signal_alphas": pd.DataFrame(self.signal_alpha_calibrator.audit_records),
            "budgets": pd.DataFrame(self.budget_controller.records()),
        }

    # def run(self) -> pd.DataFrame: 
    #     def _process_agent(agent:StockDisagreementAgent, date: int, stock_selector):  
    #         agent.stock_selector.get_next()  
    #         agent.invest(date, stock_selector)  
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  
    #         for date in self.dates:  
    #             futures = []  
    #             for agent in self.agents:  
    #                 future = executor.submit(  
    #                 _process_agent,  
    #                 agent,  
    #                 date,  
    #                 self.stock_selector_for_per_investor)  
    #                 futures.append(future)
    #             for future in concurrent.futures.as_completed(futures):  
    #                 future.result() 
    #     investment_analyzer = self.agents[0].investment_analyzer
    #     investment_res = self.stock_pool.copy()
    #     investment_res["Signal"] = 0
    #     investment_res["Signal_mean"] = 0
    #     investment_res["Signal_std"] = 0
    #     for date in self.dates:
    #         current_pool = self.stock_pool[self.stock_pool["Date"] == date]["Stock"].tolist()
    #         res = investment_analyzer.calculate_stock_disagreement_score(date, current_pool)
    #         for stock in res: 
    #             investment_res.loc[
    #                 (investment_res["Date"] == date) &   
    #                 (investment_res["Stock"] == stock),   
    #                 "Signal"  
    #             ] = res[stock][0]
    #             investment_res.loc[
    #                 (investment_res["Date"] == date) &   
    #                 (investment_res["Stock"] == stock),   
    #                 "Signal_mean"  
    #             ] = res[stock][1]
    #             investment_res.loc[
    #                 (investment_res["Date"] == date) &   
    #                 (investment_res["Stock"] == stock),   
    #                 "Signal_std"  
    #             ] = res[stock][2]
    #     return investment_res

    # def run(self, ) -> InvestmentAnalyzer:
    #     for date in self.dates:
    #         for agent in self.agents:
    #             agent.stock_selector.get_next()
    #             agent.invest(date, self.stock_selector_for_per_investor)
        
    #     return self.agents[0].investment_analyzer      
