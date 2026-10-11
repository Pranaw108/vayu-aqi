"""Compare two trained models through one shared evaluation path.

Every mis-comparison in this project came from measuring two things with two
different pieces of code: an ad-hoc script against a pipeline metric, a
December-only figure against a three-fold average, a stale report against a
fresh one. This exists so a comparison can only be made one way.

Both models are scored on the same rows with the same function, and onset
detection is reported as a curve rather than at a single threshold — recall and
precision trade off against each other, so one operating point cannot say
whether a model detects better or is merely tuned differently.

Writes:
    <reports>/comparison_<a>_vs_<b>.md

Usage:
    python pipeline/compare_models.py --a no_fires --b core_fires
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

EPISODE_THRESHOLD = 120.0
THRESHOLD_SWEEP = (80, 90, 100, 110, 120, 130, 140, 160)


@dataclass(frozen=True)
class Scored:
    """One model's predictions on the shared evaluation rows."""
    tag: str
    features: int
    trees: int
    prediction: np.ndarray
    actual: np.ndarray
    current: np.ndarray


def load_and_score(data_dir: Path, tag: str, target: str, horizon: int,
                   start: str, end: str) -> Scored:
    """Predict with one tagged model over the given window."""
    model_dir = data_dir / "models" / f"{target}_h{horizon}_{tag}"
    booster = lgb.Booster(model_file=str(model_dir / "model_p50.txt"))
    names = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))

    features_path = data_dir / "features" / f"target={target}" / f"horizon={horizon}" / "features.parquet"
    frame = pd.read_parquet(features_path)
    hours = pd.to_datetime(frame["hour"])
    window = frame[(hours >= start) & (hours < end)]

    missing = [n for n in names if n not in window.columns]
    if missing:
        raise ValueError(
            f"Model {tag!r} needs {len(missing)} feature(s) absent from the current feature "
            f"table, e.g. {missing[:3]}. Rebuild features for the configuration this model "
            f"was trained on, or score each model against its own table."
        )

    matrix = window[names].apply(pd.to_numeric, errors="coerce").astype("float32")
    return Scored(
        tag=tag,
        features=len(names),
        trees=booster.num_trees(),
        prediction=booster.predict(matrix),
        actual=window["target"].to_numpy(dtype=float),
        current=window["baseline_persistence"].to_numpy(dtype=float),
    )


def usable(scored: Scored) -> np.ndarray:
    return ~(np.isnan(scored.prediction) | np.isnan(scored.actual) | np.isnan(scored.current))


def accuracy(scored: Scored) -> dict[str, float]:
    """MAE and skill against persistence, conditioned only on the present."""
    keep = usable(scored)
    prediction, actual, current = scored.prediction[keep], scored.actual[keep], scored.current[keep]

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).mean()) if a.size else float("nan")

    calm = current < EPISODE_THRESHOLD
    persistence_mae = mae(current, actual)
    model_mae = mae(prediction, actual)
    # Persistence can be exactly right on a short or synthetic window, and
    # dividing by its error would raise rather than report.
    skill = 100 * (1 - model_mae / persistence_mae) if persistence_mae else float("nan")
    return {
        "rows": int(keep.sum()),
        "mae": model_mae,
        "persistence_mae": persistence_mae,
        "skill": skill,
        "mae_calm_now": mae(prediction[calm], actual[calm]),
        "mae_elevated_now": mae(prediction[~calm], actual[~calm]),
    }


def detection(scored: Scored) -> dict:
    """Onset detection as a curve, plus threshold-free summaries.

    Only hours that are calm now can contain an onset, so the curve is measured
    on those. The outcome is the label here, which is a legitimate use of it.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    keep = usable(scored)
    prediction, actual, current = scored.prediction[keep], scored.actual[keep], scored.current[keep]
    calm = current < EPISODE_THRESHOLD

    label = (actual[calm] >= EPISODE_THRESHOLD).astype(int)
    score = prediction[calm]
    if label.sum() == 0 or label.sum() == label.size:
        return {"onsets": int(label.sum()), "average_precision": float("nan"), "roc_auc": float("nan"), "curve": []}

    curve = []
    for threshold in THRESHOLD_SWEEP:
        called = score >= threshold
        hits = int((called & (label == 1)).sum())
        curve.append({
            "threshold": threshold,
            "recall": hits / max(int(label.sum()), 1),
            "precision": hits / max(int(called.sum()), 1),
        })
    return {
        "calm_hours": int(calm.sum()),
        "onsets": int(label.sum()),
        "average_precision": float(average_precision_score(label, score)),
        "roc_auc": float(roc_auc_score(label, score)),
        "curve": curve,
    }


def build_summary(a: Scored, b: Scored, start: str, end: str) -> str:
    acc_a, acc_b = accuracy(a), accuracy(b)
    det_a, det_b = detection(a), detection(b)

    def delta(x: float, y: float) -> str:
        return f"{y - x:+.2f}"

    lines = [
        f"# {a.tag} vs {b.tag}",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/compare_models.py`.",
        "",
        "Both models scored on the **same rows** with the **same code**. That is the whole",
        "point of this script: every mis-comparison in this project came from two numbers",
        "produced by two different pieces of code.",
        "",
        f"- Window: **{start}** to **{end}**",
        f"- Rows scored: **{acc_a['rows']:,}** (identical for both)",
        f"- `{a.tag}`: {a.features} features, {a.trees} trees",
        f"- `{b.tag}`: {b.features} features, {b.trees} trees",
        "",
        "## Accuracy",
        "",
        f"| Metric | {a.tag} | {b.tag} | Δ |",
        "|---|---:|---:|---:|",
        f"| MAE (µg/m³) | {acc_a['mae']:.2f} | {acc_b['mae']:.2f} | {delta(acc_a['mae'], acc_b['mae'])} |",
        f"| Skill vs persistence | {acc_a['skill']:.2f}% | {acc_b['skill']:.2f}% | {delta(acc_a['skill'], acc_b['skill'])} |",
        f"| MAE, calm now | {acc_a['mae_calm_now']:.2f} | {acc_b['mae_calm_now']:.2f} | {delta(acc_a['mae_calm_now'], acc_b['mae_calm_now'])} |",
        f"| MAE, elevated now | {acc_a['mae_elevated_now']:.2f} | {acc_b['mae_elevated_now']:.2f} | {delta(acc_a['mae_elevated_now'], acc_b['mae_elevated_now'])} |",
        "",
        "## Onset detection, free of any threshold choice",
        "",
        "Recall and precision trade off against each other, so a single operating point",
        "cannot say which model detects better. These two summaries do not depend on where",
        "the threshold is set.",
        "",
        f"| Measure | {a.tag} | {b.tag} | Δ |",
        "|---|---:|---:|---:|",
        f"| Average precision | {det_a['average_precision']:.4f} | {det_b['average_precision']:.4f} "
        f"| {det_b['average_precision'] - det_a['average_precision']:+.4f} |",
        f"| ROC AUC | {det_a['roc_auc']:.4f} | {det_b['roc_auc']:.4f} "
        f"| {det_b['roc_auc'] - det_a['roc_auc']:+.4f} |",
        "",
        f"Measured on {det_a['calm_hours']:,} calm hours containing {det_a['onsets']:,} onsets.",
        "",
        "### The full curve",
        "",
        f"| Threshold | {a.tag} recall | {a.tag} precision | {b.tag} recall | {b.tag} precision |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point_a, point_b in zip(det_a["curve"], det_b["curve"]):
        lines.append(
            f"| {point_a['threshold']} | {100 * point_a['recall']:.1f}% | {100 * point_a['precision']:.1f}% "
            f"| {100 * point_b['recall']:.1f}% | {100 * point_b['precision']:.1f}% |"
        )
    lines += [
        "",
        "The 120 µg/m³ threshold the reports quote is a choice, not a fact. A health warning",
        "may well prefer a lower one: more false alarms, far fewer missed episodes.",
        "",
    ]

    gain = det_b["average_precision"] - det_a["average_precision"]
    verdict = ("detects better" if gain > 0.01 else
               "detects worse" if gain < -0.01 else
               "does not detect measurably better or worse")
    lines += [
        "## Verdict",
        "",
        f"`{b.tag}` {verdict} than `{a.tag}`: average precision moves by {gain:+.4f}.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--a", required=True, help="First model tag, the baseline")
    p.add_argument("--b", required=True, help="Second model tag, the treatment")
    p.add_argument("--target", default="pm2_5")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--start", default="2024-10-01")
    p.add_argument("--end", default="2024-12-30")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    log = logging.getLogger("compare_models")

    try:
        a = load_and_score(args.data, args.a, args.target, args.horizon, args.start, args.end)
        b = load_and_score(args.data, args.b, args.target, args.horizon, args.start, args.end)
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        return 2

    args.reports.mkdir(parents=True, exist_ok=True)
    name = f"comparison_{args.a}_vs_{args.b}.md"
    (args.reports / name).write_text(build_summary(a, b, args.start, args.end), encoding="utf-8")

    det_a, det_b = detection(a), detection(b)
    log.info("%s: AP %.4f  AUC %.4f", a.tag, det_a["average_precision"], det_a["roc_auc"])
    log.info("%s: AP %.4f  AUC %.4f", b.tag, det_b["average_precision"], det_b["roc_auc"])
    log.info("Report: %s", args.reports / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
