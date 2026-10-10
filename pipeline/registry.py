"""Build a trustworthy station registry from stations.csv (DF-02).

stations.csv describes 553 monitoring stations, but it cannot be used as-is:

  * 84 stations share one coordinate pair, (22.6876, 79.3704), which is the geographic
    centre of India. That is a placeholder someone used for "location unknown".
  * ~106 stations have a pollution control board acronym (HSPCB, KSPCB, UPPCB...) in the
    city column instead of a city name, so city-level averages built on them are wrong.
  * A few rows and files do not line up: one CSV has no metadata row, two metadata rows
    have no CSV.

This script marks each problem with a flag rather than deleting rows. Stations with bad
coordinates are still fine for training a per-station model -- they are only unusable on a
map or in any feature that depends on distance between stations.

Writes:
    <out>/registry.parquet          one row per station, with is_mappable / has_real_city
    <out>/reports/registry_summary.md   what was flagged and why

Usage:
    python pipeline/registry.py --raw-root "D:/.../India datasets"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

STATIONS_FILE = "stations.csv"

# Rough bounding box for India, used to catch coordinates that are simply wrong.
INDIA_BOUNDS = {"lat_min": 6.0, "lat_max": 37.5, "lon_min": 68.0, "lon_max": 97.5}

# A coordinate shared by more than this many stations is a placeholder, not a location.
# Real stations in one city sit metres apart, never on the exact same point.
SHARED_COORD_LIMIT = 2

# City names that are really agency acronyms: 3-6 capital letters, e.g. HSPCB, MPCB, CECB.
AGENCY_NAME_PATTERN = re.compile(r"^[A-Z]{3,6}$")

log = logging.getLogger("registry")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_stations(raw_root: Path) -> pd.DataFrame:
    """Read stations.csv and drop exact duplicate station codes."""
    df = pd.read_csv(raw_root / STATIONS_FILE)
    before = len(df)
    df = df.drop_duplicates(subset=["station_code"], keep="first")
    if before != len(df):
        log.warning("Dropped %d duplicate station_code rows", before - len(df))
    return df


def list_station_codes_on_disk(raw_root: Path) -> set[str]:
    """Station codes that actually have a data file, taken from the CSV filenames."""
    return {p.stem for p in raw_root.glob("*/*.csv")}


# ---------------------------------------------------------------------------
# Flagging problems
# ---------------------------------------------------------------------------

def flag_placeholder_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Mark stations whose coordinates are missing, outside India, or shared with many others."""
    df = df.copy()
    lat, lon = df["latitude"], df["longitude"]

    df["missing_coords"] = lat.isna() | lon.isna()

    df["outside_india"] = (
        ~df["missing_coords"]
        & (
            (lat < INDIA_BOUNDS["lat_min"]) | (lat > INDIA_BOUNDS["lat_max"])
            | (lon < INDIA_BOUNDS["lon_min"]) | (lon > INDIA_BOUNDS["lon_max"])
        )
    )

    # Round before counting: two stations "sharing" a point may differ in the 9th decimal.
    point = lat.round(4).astype(str) + "," + lon.round(4).astype(str)
    stations_at_point = point.map(point.value_counts())
    df["shared_coords"] = ~df["missing_coords"] & (stations_at_point > SHARED_COORD_LIMIT)
    df["stations_at_same_point"] = stations_at_point.where(~df["missing_coords"], 0).astype(int)

    return df


def flag_agency_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows where the city column holds an agency acronym instead of a city."""
    df = df.copy()
    city = df["city"].fillna("")
    df["has_real_city"] = ~city.str.match(AGENCY_NAME_PATTERN) & (city.str.strip() != "")
    return df


def reconcile_with_files(df: pd.DataFrame, codes_on_disk: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Keep only stations that have both a metadata row and a data file.

    Returns the filtered registry and the list of codes that have a file but no metadata.
    """
    df = df.copy()
    metadata_codes = set(df["station_code"])

    files_without_metadata = sorted(codes_on_disk - metadata_codes)
    metadata_without_files = sorted(metadata_codes - codes_on_disk)

    if metadata_without_files:
        log.info("Dropping %d stations with metadata but no data file: %s",
                 len(metadata_without_files), ", ".join(metadata_without_files))
    df = df[df["station_code"].isin(codes_on_disk)]

    return df, files_without_metadata


def build_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Add the single is_mappable flag the rest of the project checks."""
    df = flag_placeholder_coordinates(df)
    df = flag_agency_city_names(df)

    # A station goes on the map only if we actually know where it is.
    df["is_mappable"] = ~(df["missing_coords"] | df["outside_india"] | df["shared_coords"])

    return df.sort_values("station_code").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame, orphan_files: list[str], out_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Station registry — DF-02")
    lines.append("")
    lines.append(f"Generated {date.today().isoformat()} by `pipeline/registry.py`.")
    lines.append("")
    lines.append(f"- Stations in registry: **{len(df):,}**")
    lines.append(f"- Usable on a map: **{int(df['is_mappable'].sum()):,}**")
    lines.append(f"- With a real city name: **{int(df['has_real_city'].sum()):,}**")
    lines.append(f"- Output: `{out_path}`")
    lines.append("")

    lines.append("## Why stations were excluded from the map")
    lines.append("")
    lines.append("| Reason | Stations |")
    lines.append("|---|---:|")
    lines.append(f"| Coordinates shared with many other stations (placeholder) "
                 f"| {int(df['shared_coords'].sum()):,} |")
    lines.append(f"| Coordinates missing | {int(df['missing_coords'].sum()):,} |")
    lines.append(f"| Coordinates outside India | {int(df['outside_india'].sum()):,} |")
    lines.append("")

    shared = df[df["shared_coords"]]
    if not shared.empty:
        lines.append("### Placeholder coordinate points")
        lines.append("")
        lines.append("| Latitude | Longitude | Stations | States |")
        lines.append("|---:|---:|---:|---|")
        grouped = shared.groupby([shared["latitude"].round(4), shared["longitude"].round(4)])
        for (lat, lon), group in sorted(grouped, key=lambda g: -len(g[1])):
            states = ", ".join(sorted(group["state_code"].dropna().unique()))
            lines.append(f"| {lat} | {lon} | {len(group)} | {states} |")
        lines.append("")

    lines.append("## Stations named after an agency, not a city")
    lines.append("")
    lines.append("City rollups must skip these until someone maps them by hand.")
    lines.append("")
    agency = df[~df["has_real_city"]]
    if agency.empty:
        lines.append("- None")
    else:
        lines.append("| Name in `city` | Stations |")
        lines.append("|---|---:|")
        for name, count in agency["city"].fillna("(blank)").value_counts().items():
            lines.append(f"| {name} | {count} |")
    lines.append("")

    lines.append("## Files with no metadata row")
    lines.append("")
    lines.append("These have readings but no location, so they cannot be registered.")
    lines.append("")
    lines.extend([f"- `{code}`" for code in orphan_files] or ["- None"])
    lines.append("")

    lines.append("## Mappable stations per state")
    lines.append("")
    lines.append("| State code | Mappable | Total |")
    lines.append("|---|---:|---:|")
    for state_code, group in df.groupby("state_code"):
        lines.append(f"| {state_code} | {int(group['is_mappable'].sum())} | {len(group)} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=os.environ.get("VAYU_RAW_ROOT"),
                   help="Folder holding stations.csv and the state subfolders "
                        "(or set VAYU_RAW_ROOT)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Where registry.parquet is written (default: data/)")
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS,
                   help="Where the summary report is written (default: reports/)")
    args = p.parse_args(argv)
    if not args.raw_root:
        p.error("--raw-root is required (or set VAYU_RAW_ROOT)")
    if not (args.raw_root / STATIONS_FILE).exists():
        p.error(f"{STATIONS_FILE} not found in {args.raw_root}")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)

    stations = load_stations(args.raw_root)
    codes_on_disk = list_station_codes_on_disk(args.raw_root)
    log.info("Loaded %d metadata rows, found %d station files", len(stations), len(codes_on_disk))

    stations, orphan_files = reconcile_with_files(stations, codes_on_disk)
    registry = build_registry(stations)

    registry_path = args.out / "registry.parquet"
    registry.to_parquet(registry_path, index=False, compression="zstd")

    summary = build_summary(registry, orphan_files, registry_path)
    (args.reports / "registry_summary.md").write_text(summary, encoding="utf-8")

    log.info("Registry: %d stations, %d mappable, %d with a real city name",
             len(registry), int(registry["is_mappable"].sum()),
             int(registry["has_real_city"].sum()))
    if orphan_files:
        log.warning("%d data files have no metadata row: %s",
                    len(orphan_files), ", ".join(orphan_files))
    log.info("Report: %s", args.reports / "registry_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
