import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any


def add_horizon_signals(frame: pd.DataFrame, alphas: dict[str, float]) -> pd.DataFrame:
    """Add horizon-specific signals without mutating the input frame."""
    result = frame.copy()
    for horizon, alpha in alphas.items():
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Horizon signal alpha must be between 0 and 1.")
        result[f"Signal_{horizon}"] = result["ExposureControl"] * (
            alpha * result["Signal_mean"]
            + (1.0 - alpha) * result["Signal_std"]
        )
    return result


@dataclass(frozen=True)
class HorizonTarget:
    horizon: str
    label_column: str
    maturity_days: int = 15


class RollingHorizonAlphaCalibrator:
    """Causally calibrate horizon alphas from matured cross-sectional outcomes."""

    def __init__(
        self,
        targets: list[HorizonTarget],
        fallback_alphas: dict[str, float],
        grid_step: float = 0.1,
        min_observations: int = 20,
        lookback_observations: int = 60,
        shrinkage: float = 0.5,
        metric: str = "rank_ic",
    ) -> None:
        if not 0.0 < grid_step <= 1.0:
            raise ValueError("grid_step must be in (0, 1].")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between 0 and 1.")
        if metric not in {"ic", "rank_ic"}:
            raise ValueError("metric must be 'ic' or 'rank_ic'.")
        self.targets = {target.horizon: target for target in targets}
        self.fallback_alphas = {
            horizon: float(fallback_alphas[horizon]) for horizon in self.targets
        }
        self.grid = np.unique(np.append(np.arange(0.0, 1.0 + grid_step / 2.0, grid_step), 1.0))
        self.min_observations = max(int(min_observations), 1)
        self.lookback_observations = max(int(lookback_observations), 1)
        self.shrinkage = float(shrinkage)
        self.metric = metric
        self.history: list[tuple[int, pd.DataFrame]] = []
        self.daily_scores: dict[str, list[pd.Series]] = {
            horizon: [] for horizon in self.targets
        }
        self.audit_records: list[dict[str, Any]] = []

    def select(self, date: int) -> dict[str, float]:
        selected: dict[str, float] = {}
        for horizon, target in self.targets.items():
            maturity = max(int(target.maturity_days), 1)
            eligible_count = max(len(self.history) - maturity + 1, 0)
            eligible = self.daily_scores[horizon][:eligible_count][-self.lookback_observations:]
            scores = self._grid_scores(eligible)
            valid_scores = scores.dropna()
            fallback = self.fallback_alphas[horizon]
            if len(eligible) < self.min_observations or valid_scores.empty:
                best_alpha = fallback
                alpha = fallback
                status = "fallback"
            else:
                best_alpha = float(valid_scores.idxmax())
                alpha = self.shrinkage * fallback + (1.0 - self.shrinkage) * best_alpha
                status = "calibrated"
            selected[horizon] = float(alpha)
            self.audit_records.append({
                "date": int(date),
                "horizon": horizon,
                "alpha": float(alpha),
                "best_grid_alpha": float(best_alpha),
                "fallback_alpha": fallback,
                "matured_observations": len(eligible),
                "calibration_score": float(valid_scores.get(best_alpha, np.nan)),
                "status": status,
                "metric": self.metric,
            })
        return selected

    def observe(self, date: int, frame: pd.DataFrame) -> None:
        required = {"Signal_mean", "Signal_std", "ExposureControl"}
        required.update(target.label_column for target in self.targets.values())
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Calibration frame is missing columns: {sorted(missing)}")
        observation = frame[list(required)].copy()
        self.history.append((int(date), observation))
        for horizon, target in self.targets.items():
            self.daily_scores[horizon].append(
                self._daily_grid_scores(observation, target.label_column)
            )

    def _grid_scores(
        self,
        eligible: list[pd.Series],
    ) -> pd.Series:
        if not eligible:
            return pd.Series({float(alpha): np.nan for alpha in self.grid}, dtype=float)
        return pd.DataFrame(eligible).mean(axis=0, skipna=True).reindex(self.grid)

    def _daily_grid_scores(self, frame: pd.DataFrame, label_column: str) -> pd.Series:
        scores: dict[float, float] = {}
        for alpha in self.grid:
            signal = frame["ExposureControl"] * (
                alpha * frame["Signal_mean"] + (1.0 - alpha) * frame["Signal_std"]
            )
            pair = pd.DataFrame({"signal": signal, "label": frame[label_column]}).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if len(pair) < 3 or pair.signal.nunique() < 2 or pair.label.nunique() < 2:
                scores[float(alpha)] = np.nan
                continue
            if self.metric == "rank_ic":
                correlation = pair.signal.rank().corr(pair.label.rank())
            else:
                correlation = pair.signal.corr(pair.label)
            scores[float(alpha)] = float(correlation) if np.isfinite(correlation) else np.nan
        return pd.Series(scores, dtype=float)
