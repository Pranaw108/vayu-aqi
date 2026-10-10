"""Fetch NASA FIRMS active-fire history for the training window (LI-04).

Stubble burning drives north India's worst winter episodes. The Delhi backtest
showed the model detects about half of episode onsets but predicts 119 ug/m3
against an actual 163 — it knows something is coming and understates how bad.
Fire data is the missing cause behind that gap.

Needs FIRMS_MAP_KEY. The API covers at most 5 days per request, so seven years
is roughly 512 calls against a 5,000-per-10-minutes limit. Each chunk is cached,
so an interrupted run resumes.

Writes:
    <data>/fires/fires.parquet       one row per detection
    <data>/fire_cache/*.parquet      one file per 5-day chunk
    <reports>/fires_summary.md

Usage:
    python pipeline/fetch_fires.py
    python pipeline/fetch_fires.py --start 2024-10-01 --end 2024-12-31   # one season
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.sources.firms import (  # noqa: E402
    INDIA_BBOX,
    BoundingBox,
    FirmsClient,
    MissingMapKey,
    date_chunks,
    drop_low_confidence,
)

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2024, 12, 31)

log = logging.getLogger("fetch_fires")


def load_env_key(repo_root: Path) -> None:
    """Read FIRMS_MAP_KEY from .env if it is not already in the environment.

    Keeps the key out of the shell history and out of git — .env is ignored.
    """
    import os

    if os.environ.get("FIRMS_MAP_KEY"):
        return
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("FIRMS_MAP_KEY="):
            os.environ["FIRMS_MAP_KEY"] = line.split("=", 1)[1].strip()
            return


def fetch_all(client: FirmsClient, bbox: BoundingBox, start: date, end: date) -> pd.DataFrame:
    chunks = date_chunks(start, end)
    log.info("Fetching %d chunks of up to 5 days each, %s to %s", len(chunks), start, end)

    frames = []
    for n, (chunk_start, days) in enumerate(chunks, 1):
        try:
            frames.append(client.fetch_chunk(bbox, chunk_start, days))
        except MissingMapKey:
            raise
        except Exception as error:
            # One bad chunk must not lose the other 500.
            log.error("  %s (%dd) failed: %s: %s", chunk_start, days, type(error).__name__, error)
            continue
        if n % 25 == 0 or n == len(chunks):
            total = sum(len(f) for f in frames)
            log.info("  %d/%d chunks · %s detections · %d calls",
                     n, len(chunks), f"{total:,}", client.calls_made)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def build_summary(fires: pd.DataFrame, calls: int, elapsed_min: float,
                  start: date, end: date, dropped_low: int) -> str:
    by_month = (
        fires.assign(month=fires["acq_date"].dt.month)
        .groupby("month", as_index=False)
        .agg(detections=("frp", "size"), total_frp=("frp", "sum"))
    )
    peak = by_month.loc[by_month["detections"].idxmax()]

    lines = [
        "# Active fires — NASA FIRMS",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/fetch_fires.py`.",
        "",
        f"- Window: **{start}** to **{end}**",
        f"- Detections kept: **{len(fires):,}**",
        f"- Low-confidence detections dropped: **{dropped_low:,}**",
        f"- API calls: **{calls}** (limit 5,000 per 10 minutes)",
        f"- Runtime: **{elapsed_min:.1f} min**",
        "",
        "Source: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/), VIIRS 375m, "
        "science-quality. Free MAP_KEY required.",
        "",
        "## Detections by month",
        "",
        "If this data is any use, October and November should dominate — that is the "
        "stubble-burning season, and it is when Delhi's air collapses.",
        "",
        "| Month | Detections | Total FRP |",
        "|---|---:|---:|",
    ]
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for row in by_month.itertuples():
        lines.append(f"| {month_names[int(row.month)]} | {int(row.detections):,} | {row.total_frp:,.0f} |")
    lines += [
        "",
        f"Peak month: **{month_names[int(peak.month)]}** with {int(peak.detections):,} detections.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=DEFAULT_START)
    p.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=DEFAULT_END)
    p.add_argument("--keep-low-confidence", action="store_true",
                   help="Keep low-confidence detections (they carry many false positives)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    load_env_key(REPO_ROOT)

    client = FirmsClient(args.data / "fire_cache")
    try:
        client.require_key()
    except MissingMapKey as error:
        log.error("%s", error)
        return 2

    start_time = time.time()
    fires = fetch_all(client, BoundingBox(*INDIA_BBOX), args.start, args.end)
    elapsed_min = (time.time() - start_time) / 60

    if fires.empty:
        log.error("No detections fetched")
        return 1

    before = len(fires)
    if not args.keep_low_confidence:
        fires = drop_low_confidence(fires)
    dropped_low = before - len(fires)

    out_dir = args.data / "fires"
    out_dir.mkdir(parents=True, exist_ok=True)
    fires.to_parquet(out_dir / "fires.parquet", index=False, compression="zstd")

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "fires_summary.md").write_text(
        build_summary(fires, client.calls_made, elapsed_min, args.start, args.end, dropped_low),
        encoding="utf-8",
    )

    log.info("Done in %.1f min: %s detections kept (%s low-confidence dropped), %d calls",
             elapsed_min, f"{len(fires):,}", f"{dropped_low:,}", client.calls_made)
    log.info("Report: %s", args.reports / "fires_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
