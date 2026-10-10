"""Train and honestly evaluate the forecast model (MD-03, MD-04).

Evaluation uses **rolling-origin** validation: train on everything up to a date,
test on the month that follows, advance, repeat. A random train/test split would
put future readings in the training set, which makes the scores excellent and
meaningless — that is the second classic way these projects fool themselves.

Nothing is reported without the persistence baseline beside it. "MAE 18" means
nothing on its own; "MAE 18 against a baseline of 27" is a result.

Three quantile models are trained per fold (p10, p50, p90) so the service can
publish a forecast band rather than a false point estimate, and the report
checks that the band actually contains the truth about 80% of the time.

Writes:
    <data>/models/<target>_h<H>/model_p{10,50,90}.txt
    <reports>/model_<target>_h<H>.md

Usage:
    python pipeline/train.py --target pm2_5 --horizon 24 --folds 3   # quick
    python pipeline/train.py --target pm2_5 --horizon 24             # full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.baselines import all_baselines  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Columns that are not features.
NON_FEATURES = {"station_code", "hour", "target", "target_hour", "baseline_persistence"}

# One fold tests this many days, then the window expands.
TEST_DAYS = 30

# A fold needs a reasonable amount of history before it can teach anything.
MIN_TRAIN_DAYS = 365

QUANTILES = (0.1, 0.5, 0.9)

# Pollution above this is the episode people actually need a forecast for.
# 120 ug/m3 PM2.5 is the bottom of CPCB's "Very Poor" sub-index band.
EPISODE_THRESHOLD = 120.0

LGB_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "num_threads": 0,
}
NUM_ROUNDS = 400
EARLY_STOPPING = 40

log = logging.getLogger("train")


@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    interval_coverage: float = np.nan
    season_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    episode_scores: dict[str, float] = field(default_factory=dict)


def summarise_stations(codes: pd.Series) -> str:
    """Short, unambiguous description of which stations a run used."""
    unique = sorted(codes.unique())
    if len(unique) <= 8:
        return ", ".join(unique)
    return f"{unique[0]}..{unique[-1]} ({len(unique)} stations)"


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE and RMSE over the rows where both exist."""
    usable = ~(np.isnan(actual) | np.isnan(predicted))
    if not usable.any():
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    error = actual[usable] - predicted[usable]
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "n": int(usable.sum()),
    }


def skill(model_mae: float, baseline_mae: float) -> float:
    """Percent of the baseline's error removed. Negative means worse than doing nothing."""
    if not np.isfinite(baseline_mae) or baseline_mae == 0:
        return float("nan")
    return 100 * (1 - model_mae / baseline_mae)


def season_of(month: int) -> str:
    """Indian seasons, which drive air quality far more than calendar quarters."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post-monsoon"


def make_folds(frame: pd.DataFrame, folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Expanding-window fold boundaries, newest last.

    Each fold trains on everything before `train_end` and tests the month after.
    """
    hours = pd.to_datetime(frame["hour"])
    first, last = hours.min(), hours.max()
    earliest_train_end = first + pd.Timedelta(days=MIN_TRAIN_DAYS)

    boundaries = []
    test_end = last
    while len(boundaries) < folds:
        test_start = test_end - pd.Timedelta(days=TEST_DAYS)
        if test_start <= earliest_train_end:
            break
        boundaries.append((test_start, test_start, test_end))
        test_end = test_start
    # Oldest first, so the report reads forwards in time.
    return [(train_end, test_start, test_end) for train_end, test_start, test_end in reversed(boundaries)]


def train_quantile_models(train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str]) -> dict[float, lgb.Booster]:
    """One model per quantile, so the service can publish a band."""
    models = {}
    for quantile in QUANTILES:
        params = {**LGB_PARAMS, "alpha": quantile}
        dataset = lgb.Dataset(train[feature_names], label=train["target"])
        valid_set = lgb.Dataset(valid[feature_names], label=valid["target"], reference=dataset)
        models[quantile] = lgb.train(
            params,
            dataset,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
        )
    return models


def run_fold(frame: pd.DataFrame, feature_names: list[str], fold: int,
             train_end: pd.Timestamp, test_start: pd.Timestamp, test_end: pd.Timestamp) -> FoldResult | None:
    """Fit baselines and the model on one expanding window, then score the month after."""
    hours = pd.to_datetime(frame["hour"])
    train = frame[hours < train_end]
    test = frame[(hours >= test_start) & (hours < test_end)]
    if len(train) < 1000 or len(test) < 100:
        log.warning("  fold %d skipped: train=%d test=%d", fold, len(train), len(test))
        return None

    result = FoldResult(
        fold=fold, train_end=train_end, test_start=test_start, test_end=test_end,
        train_rows=len(train), test_rows=len(test),
    )
    actual = test["target"].to_numpy(dtype=float)

    for baseline in all_baselines():
        baseline.fit(train)
        result.scores[baseline.name] = score(actual, baseline.predict(test))

    # Hold back the last stretch of training data to stop boosting early.
    split = pd.to_datetime(train["hour"]).quantile(0.9)
    inner_train = train[pd.to_datetime(train["hour"]) < split]
    inner_valid = train[pd.to_datetime(train["hour"]) >= split]
    if len(inner_valid) < 200:
        inner_train, inner_valid = train, train

    models = train_quantile_models(inner_train, inner_valid, feature_names)
    predictions = {q: m.predict(test[feature_names]) for q, m in models.items()}

    result.scores["lightgbm"] = score(actual, predictions[0.5])

    # Does the published 80% band actually contain the truth 80% of the time?
    inside = (actual >= predictions[0.1]) & (actual <= predictions[0.9])
    result.interval_coverage = float(np.nanmean(inside[~np.isnan(actual)]))

    # Overall averages hide failure exactly where the service matters most.
    months = pd.to_datetime(test["hour"]).dt.month.map(season_of).to_numpy()
    for season in pd.unique(months):
        mask = months == season
        result.season_scores[season] = {
            "lightgbm": score(actual[mask], predictions[0.5][mask])["mae"],
            "persistence": score(actual[mask], test["baseline_persistence"].to_numpy(dtype=float)[mask])["mae"],
            "n": int(mask.sum()),
        }

    episode = actual >= EPISODE_THRESHOLD
    if episode.any():
        result.episode_scores = {
            "lightgbm": score(actual[episode], predictions[0.5][episode])["mae"],
            "persistence": score(actual[episode], test["baseline_persistence"].to_numpy(dtype=float)[episode])["mae"],
            "n": int(episode.sum()),
        }

    result.models = models  # type: ignore[attr-defined]
    return result


def build_summary(results: list[FoldResult], target: str, horizon: int, feature_names: list[str],
                  stations: int, importance: pd.DataFrame, elapsed_min: float,
                  station_codes: str = "") -> str:
    def mean_of(name: str, key: str = "mae") -> float:
        values = [r.scores[name][key] for r in results if name in r.scores and np.isfinite(r.scores[name][key])]
        return float(np.mean(values)) if values else float("nan")

    model_mae = mean_of("lightgbm")
    persistence_mae = mean_of("persistence")
    coverage = float(np.nanmean([r.interval_coverage for r in results]))

    lines = [
        f"# Forecast model — {target}, +{horizon}h (MD-03, MD-04)",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/train.py`.",
        "",
        f"- Target: **{target}**, **{horizon}** hours ahead",
        f"- Stations: **{stations}** · features: **{len(feature_names)}**",
        f"- Station codes: `{station_codes}`",
        f"- Rolling-origin folds: **{len(results)}**, {TEST_DAYS} test days each",
        f"- Runtime: **{elapsed_min:.1f} min**",
        "",
        "## Headline",
        "",
        "| Model | MAE (µg/m³) | RMSE | Skill vs persistence |",
        "|---|---:|---:|---:|",
    ]
    for name in ("persistence", "climatology", "persistence+climatology", "lightgbm"):
        mae, rmse = mean_of(name), mean_of(name, "rmse")
        if not np.isfinite(mae):
            continue
        label = f"**{name}**" if name == "lightgbm" else name
        skill_text = "—" if name == "persistence" else f"{skill(mae, persistence_mae):+.1f}%"
        lines.append(f"| {label} | {mae:.1f} | {rmse:.1f} | {skill_text} |")
    lines.append("")

    verdict = "beats" if model_mae < persistence_mae else "**does not beat**"
    lines += [
        f"The model {verdict} persistence by **{skill(model_mae, persistence_mae):+.1f}%** MAE.",
        "",
        "## Prediction interval",
        "",
        f"- 80% band (p10–p90) contains the truth **{100 * coverage:.1f}%** of the time",
        f"- Target is 80%. {'Well calibrated.' if 0.75 <= coverage <= 0.85 else 'Not calibrated — the published band is misleading.'}",
        "",
        "## Per fold",
        "",
        "| Fold | Test period | Train rows | Test rows | Persistence MAE | Model MAE | Skill |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        p = r.scores.get("persistence", {}).get("mae", float("nan"))
        m = r.scores.get("lightgbm", {}).get("mae", float("nan"))
        lines.append(
            f"| {r.fold} | {r.test_start:%Y-%m-%d} → {r.test_end:%Y-%m-%d} | {r.train_rows:,} "
            f"| {r.test_rows:,} | {p:.1f} | {m:.1f} | {skill(m, p):+.1f}% |"
        )
    lines.append("")

    lines += [
        "## By season",
        "",
        "Winter is when north India's air is worst and a forecast matters most.",
        "",
        "| Season | Rows | Persistence MAE | Model MAE | Skill |",
        "|---|---:|---:|---:|---:|",
    ]
    seasons: dict[str, list[tuple[float, float, int]]] = {}
    for r in results:
        for season, s in r.season_scores.items():
            seasons.setdefault(season, []).append((s["persistence"], s["lightgbm"], s["n"]))
    for season, entries in sorted(seasons.items()):
        p = float(np.nanmean([e[0] for e in entries]))
        m = float(np.nanmean([e[1] for e in entries]))
        n = sum(e[2] for e in entries)
        lines.append(f"| {season} | {n:,} | {p:.1f} | {m:.1f} | {skill(m, p):+.1f}% |")
    lines.append("")

    episodes = [r.episode_scores for r in results if r.episode_scores]
    lines += [
        f"## High-pollution episodes (actual ≥ {EPISODE_THRESHOLD:.0f} µg/m³)",
        "",
        "The hours the service exists for. An overall average hides failure here.",
        "",
    ]
    if episodes:
        p = float(np.nanmean([e["persistence"] for e in episodes]))
        m = float(np.nanmean([e["lightgbm"] for e in episodes]))
        n = sum(e["n"] for e in episodes)
        lines += [
            f"- Episode hours tested: **{n:,}**",
            f"- Persistence MAE: **{p:.1f}** · model MAE: **{m:.1f}** · skill **{skill(m, p):+.1f}%**",
            "",
            "Errors are much larger here than on the overall average. That is honest and expected:"
            " spikes are driven by stubble burning and festivals, which nothing in these features sees.",
        ]
    else:
        lines.append("- No episode hours in the test folds.")
    lines.append("")

    lines += [
        "## What the model leans on",
        "",
        "| Feature | Gain |",
        "|---|---:|",
    ]
    for row in importance.head(20).itertuples():
        lines.append(f"| {row.feature} | {row.gain:,.0f} |")
    lines += [
        "",
        "## Limitations",
        "",
        "- Trained on ERA5 reanalysis weather but will run on *forecast* weather. That mismatch",
        "  costs accuracy in production and is not measured here.",
        "- Stubble burning and Diwali drive the worst episodes and are invisible to these features.",
        "  NASA FIRMS fire counts are the fix.",
        f"- Evaluated on {stations} stations; national behaviour may differ.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--target", default="pm2_5")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--folds", type=int, default=12, help="Rolling-origin folds (default 12)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    features_path = args.data / "features" / f"target={args.target}" / f"horizon={args.horizon}" / "features.parquet"
    if not features_path.exists():
        log.error("No features at %s — run pipeline/features.py --target %s --horizon %d first",
                  features_path, args.target, args.horizon)
        return 2

    frame = pd.read_parquet(features_path).sort_values(["station_code", "hour"]).reset_index(drop=True)
    feature_names = [c for c in frame.columns if c not in NON_FEATURES]

    # Every feature column must be float32, not just the float ones. A single
    # int column makes pandas upcast the whole matrix to float64 on conversion,
    # which is what exhausted memory on the 43-station Delhi build.
    for column in feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    log.info("Feature matrix: %.1f GB", frame[feature_names].memory_usage(deep=True).sum() / 2**30)
    log.info("Loaded %s rows, %d features, %d stations",
             f"{len(frame):,}", len(feature_names), frame["station_code"].nunique())

    folds = make_folds(frame, args.folds)
    if not folds:
        log.error("Not enough history for even one fold — need more than %d days", MIN_TRAIN_DAYS)
        return 1
    log.info("Running %d rolling-origin folds", len(folds))

    start = time.time()
    results: list[FoldResult] = []
    for n, (train_end, test_start, test_end) in enumerate(folds, 1):
        result = run_fold(frame, feature_names, n, train_end, test_start, test_end)
        if result is None:
            continue
        results.append(result)
        log.info("  fold %d/%d  %s→%s  persistence %.1f  model %.1f  skill %+.1f%%",
                 n, len(folds), test_start.date(), test_end.date(),
                 result.scores["persistence"]["mae"], result.scores["lightgbm"]["mae"],
                 skill(result.scores["lightgbm"]["mae"], result.scores["persistence"]["mae"]))
    elapsed_min = (time.time() - start) / 60

    if not results:
        log.error("Every fold was skipped")
        return 1

    # Keep the newest fold's models: they saw the most data.
    newest = results[-1].models  # type: ignore[attr-defined]
    model_dir = args.data / "models" / f"{args.target}_h{args.horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    for quantile, booster in newest.items():
        booster.save_model(str(model_dir / f"model_p{int(quantile * 100)}.txt"))
    (model_dir / "features.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    importance = pd.DataFrame({
        "feature": newest[0.5].feature_name(),
        "gain": newest[0.5].feature_importance("gain"),
    }).sort_values("gain", ascending=False)

    args.reports.mkdir(parents=True, exist_ok=True)
    name = f"model_{args.target}_h{args.horizon}.md"
    (args.reports / name).write_text(
        build_summary(results, args.target, args.horizon, feature_names,
                      frame["station_code"].nunique(), importance, elapsed_min,
                      station_codes=summarise_stations(frame["station_code"])),
        encoding="utf-8",
    )

    model_mae = float(np.mean([r.scores["lightgbm"]["mae"] for r in results]))
    persistence_mae = float(np.mean([r.scores["persistence"]["mae"] for r in results]))
    log.info("Done in %.1f min: model MAE %.1f vs persistence %.1f (skill %+.1f%%)",
             elapsed_min, model_mae, persistence_mae, skill(model_mae, persistence_mae))
    log.info("Models: %s", model_dir)
    log.info("Report: %s", args.reports / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
