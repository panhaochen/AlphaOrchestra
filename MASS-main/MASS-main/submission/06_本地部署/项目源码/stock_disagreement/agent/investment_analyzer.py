import threading
import numpy as np
from typing import Any

class InvestmentAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.data = {}
        return cls._instance
    
    def __init__(self):  
        if not hasattr(self, 'initialized'):  
            self.initialized = True 
            self.data = {}  
            self.call_metrics = {}
    
    def record_score(self, date:int, investor_type: int, investor_id: int, stocks: dict[str, int], decisions=None, call_metrics=None):
        decisions = decisions or {}
        with self._lock:
            if date not in self.data:
                self.data[date] = {}
            if investor_type not in self.data[date]:
                self.data[date][investor_type] = {}
            for (key, value) in stocks.items():
                if key not in self.data[date][investor_type]:
                    self.data[date][investor_type][key] = {
                        "score": 0,
                        "num_investors": 0,
                        "confidences": [],
                        "horizons": [],
                        "evidence": [],
                        "reasoning": [],
                    }
                self.data[date][investor_type][key]["score"] += value
                self.data[date][investor_type][key]["num_investors"] += 1
                detail = decisions.get(key, {})
                self.data[date][investor_type][key]["confidences"].append(float(detail.get("confidence", 0.5)))
                self.data[date][investor_type][key]["horizons"].append(int(detail.get("horizon_days", 20)))
                self.data[date][investor_type][key]["evidence"].append(detail.get("evidence", ()))
                self.data[date][investor_type][key]["reasoning"].append(str(detail.get("reasoning_summary", "")))
            if call_metrics:
                self.call_metrics.setdefault(date, {}).setdefault(investor_type, []).append(call_metrics.copy())

    def record_call_metrics(self, date: int, investor_type: int, call_metrics: dict[str, Any]) -> None:
        """Audit a failed/abstained call even when it produced no usable opinion."""
        if not call_metrics:
            return
        with self._lock:
            self.call_metrics.setdefault(date, {}).setdefault(investor_type, []).append(call_metrics.copy())
    
    def calculate_stock_disagreement_score(self,
                                           date: int,
                                           stock_pool: list[str], 
                                           agent_distributions:dict[int, float] ,
                                           alpha:float = 0.5,
                                           disagreement_diagnostics: dict[str, dict[str, Any]] | None = None,
                                           contextual_weights: dict[str, list[float]] | None = None):
        res = self.collect_stock_opinions(date, stock_pool, agent_distributions)
        for stock_code, values in res.items():
            scores = np.array(values["scores"], dtype=float)
            weights = np.array(values["weights"], dtype=float)
            if contextual_weights and stock_code in contextual_weights:
                candidate_weights = np.array(contextual_weights[stock_code], dtype=float)
                if len(candidate_weights) == len(scores):
                    weights = candidate_weights
            if weights.sum() <= 0:
                mean_value = 0.0
                std = 0.0
            else:
                mean_value = float(np.sum(scores * weights) / np.sum(weights))
                std = float(np.sqrt(np.average((scores - mean_value) ** 2, weights=weights)))
            signal = alpha * mean_value - (1 - alpha) * std
            diagnostics = disagreement_diagnostics.get(stock_code, {}) if disagreement_diagnostics else {}
            exposure_control = float(diagnostics.get("exposure_control", 1.0))
            final_signal = exposure_control * signal
            res[stock_code] = [
                final_signal,
                mean_value,
                -std,
                exposure_control,
                float(diagnostics.get("severity", std)),
                str(diagnostics.get("type", "aggregate")),
            ]
        return res

    def collect_stock_opinions(self,
                               date: int,
                               stock_pool: list[str],
                               agent_distributions: dict[int, float]) -> dict[str, dict[str, list[float]]]:
        res: dict[str, dict[str, list[float]]] = {}
        with self._lock:
            if date not in self.data:
                return res
            # Concurrent agent calls may finish in different orders.  A stable
            # traversal keeps aggregation and its floating-point reductions
            # reproducible across live and cached runs.
            for investor_type in sorted(self.data[date]):
                if investor_type in agent_distributions:
                    distribution = agent_distributions[investor_type]
                else:
                    distribution = 0.0
                    raise ValueError(f"agent type {investor_type} does not exist!")
                for stock in stock_pool:
                    if stock not in self.data[date][investor_type]:
                        self.data[date][investor_type][stock] = {
                            "score": 0,
                            "num_investors": 0,
                            "confidences": [],
                            "horizons": [],
                            "evidence": [],
                            "reasoning": [],
                        }
                for stock_code in self.data[date][investor_type]:
                    stock_data = self.data[date][investor_type][stock_code]
                    total_scores = stock_data["score"]
                    total_investors = stock_data["num_investors"]
                    if stock_code not in res:
                        res[stock_code] = {"scores": [], "weights": [], "agent_types": [], "confidences": [], "horizons": [], "evidence": [], "reasoning": []}
                    res[stock_code]["scores"].append((total_scores / total_investors) if total_investors != 0 else 0)
                    res[stock_code]["weights"].append(distribution)
                    res[stock_code]["agent_types"].append(investor_type)
                    res[stock_code]["confidences"].append(
                        self._finite_mean(stock_data.get("confidences"), 0.5)
                    )
                    res[stock_code]["horizons"].append(
                        self._finite_mean(stock_data.get("horizons"), 20.0)
                    )
                    flattened_evidence = [item for group in stock_data.get("evidence", []) for item in group]
                    res[stock_code]["evidence"].append(flattened_evidence)
                    res[stock_code]["reasoning"].append(" ".join(stock_data.get("reasoning", [])))
        return res

    @staticmethod
    def _finite_mean(values: Any, default: float) -> float:
        if values is None:
            return default
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(finite.mean()) if len(finite) else default

    def apply_debate_revision(self, date: int, investor_type: int, stock: str, revision: dict[str, Any], call_metrics=None) -> None:
        with self._lock:
            stock_data = self.data[date][investor_type][stock]
            investor_count = max(int(stock_data.get("num_investors", 0)), 1)
            selection_probability = (float(revision.get("prediction", 0.0)) + 1.0) / 2.0
            stock_data["score"] = min(max(selection_probability, 0.0), 1.0) * investor_count
            stock_data["confidences"] = [float(revision.get("confidence", 0.5))]
            stock_data["horizons"] = [int(revision.get("horizon_days", 20))]
            stock_data["evidence"] = [revision.get("evidence", ())]
            stock_data["reasoning"] = [str(revision.get("reasoning_summary", ""))]
            if call_metrics:
                self.call_metrics.setdefault(date, {}).setdefault(investor_type, []).append(call_metrics.copy())
