"""DuckDB query layer over the curated data (DF-03).

Everything the service reads goes through here: station list, station detail,
history and city rollups. DuckDB queries the hourly Parquet directly, so there
is no database server to run.

AQI is never computed in SQL. This module pulls the recent readings a station
needs and hands them to `api.aqi`, so the CPCB rules have exactly one
implementation — the tested one.

Important: the archive ends in 2024, and stations stop at different dates. Every
reading carries how old it is, so nothing stale is ever presented as current.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from . import aqi as aqi_engine

REPO_ROOT = Path(__file__).resolve().parents[1]

# Enough history to fill a 24-hour window, with room for gaps.
AQI_WINDOW_HOURS = 48

# City rollups skip stations whose "city" is a pollution-control-board acronym.
MIN_STATIONS_FOR_CITY = 1

# Two stations are only comparable if both were read at about the same time.
COMPARABLE_WINDOW_HOURS = 24

log = logging.getLogger("store")


@dataclass(frozen=True)
class StoreStats:
    stations: int
    mappable: int
    hourly_rows: int
    first_reading: datetime
    last_reading: datetime


class Store:
    """Read-only access to the curated archive."""

    def __init__(self, data_dir: Path | str = REPO_ROOT / "data"):
        self.data_dir = Path(data_dir)
        self.hourly_glob = (self.data_dir / "hourly" / "**" / "*.parquet").as_posix()
        self.registry_path = (self.data_dir / "registry.parquet").as_posix()

        if not (self.data_dir / "hourly").is_dir():
            raise FileNotFoundError(f"No hourly data at {self.data_dir / 'hourly'} — run pipeline/build_hourly.py")
        if not (self.data_dir / "registry.parquet").exists():
            raise FileNotFoundError(f"No registry at {self.registry_path} — run pipeline/registry.py")

        self.con = duckdb.connect()
        self._register_views()
        # Keyed on the newest hour in the data, so it can never serve a result
        # from older data than the store now holds.
        self._station_cache: dict[datetime, pd.DataFrame] = {}

    def _register_views(self) -> None:
        """Name the two Parquet sources once, so queries stay readable."""
        self.con.execute(f"""
            CREATE OR REPLACE VIEW hourly AS
            SELECT * FROM read_parquet('{self.hourly_glob}', hive_partitioning=1)
        """)
        self.con.execute(f"""
            CREATE OR REPLACE VIEW registry AS
            SELECT * FROM read_parquet('{self.registry_path}')
        """)

    # ---------------------------------------------------------------- metadata

    def stats(self) -> StoreStats:
        stations, hourly_rows, first, last = self.con.execute("""
            SELECT count(DISTINCT station_code), count(*), min(hour), max(hour) FROM hourly
        """).fetchone()
        total, mappable = self.con.execute("""
            SELECT count(*), sum(CASE WHEN is_mappable THEN 1 ELSE 0 END) FROM registry
        """).fetchone()
        return StoreStats(
            stations=int(total),
            mappable=int(mappable),
            hourly_rows=int(hourly_rows),
            first_reading=first,
            last_reading=last,
        )

    def as_of(self) -> datetime:
        """Latest hour anywhere in the archive."""
        return self.con.execute("SELECT max(hour) FROM hourly").fetchone()[0]

    # ---------------------------------------------------------------- stations

    def _recent_readings(self, station_codes: list[str] | None = None) -> pd.DataFrame:
        """Each station's last AQI_WINDOW_HOURS of readings, for AQI computation."""
        where = ""
        params: list = []
        if station_codes:
            placeholders = ",".join("?" for _ in station_codes)
            where = f"WHERE station_code IN ({placeholders})"
            params = list(station_codes)

        return self.con.execute(f"""
            WITH last_hour AS (
                SELECT station_code, max(hour) AS latest FROM hourly {where} GROUP BY station_code
            )
            SELECT h.station_code, h.hour AS timestamp,
                   h.pm2_5, h.pm10, h.no2, h.so2, h.nh3, h.co, h.o3,
                   h."at", h.rh, h.ws, h.wd, h.readings
            FROM hourly h
            JOIN last_hour l USING (station_code)
            WHERE h.hour > l.latest - INTERVAL {AQI_WINDOW_HOURS} HOUR
            ORDER BY h.station_code, h.hour
        """, params).fetch_df()

    def _latest_aqi(self, station_codes: list[str] | None = None) -> pd.DataFrame:
        """One row per station: its most recent valid AQI, via the AQI engine."""
        readings = self._recent_readings(station_codes)
        if readings.empty:
            return pd.DataFrame(columns=["station_code", "aqi", "aqi_category", "dominant_pollutant", "last_reading"])

        # One windowing pass and one AQI pass over every station together.
        # Scoring each station separately spends almost all its time in pandas
        # overhead rather than on the arithmetic.
        averaged = aqi_engine.rolling_concentrations_by_station(readings)
        scored = aqi_engine.compute_aqi_frame(averaged)
        scored["station_code"] = averaged["station_code"].to_numpy()
        scored["timestamp"] = averaged["timestamp"].to_numpy()

        # Prefer each station's newest row that produced a valid AQI; if none
        # did, keep its newest row so the refusal reason still reaches the API.
        scored["has_aqi"] = scored["aqi"].notna()
        scored = scored.sort_values(["station_code", "has_aqi", "timestamp"])
        latest = scored.groupby("station_code", as_index=False, sort=False).last()

        last_reading = readings.groupby("station_code", as_index=False)["timestamp"].max()
        last_reading = last_reading.rename(columns={"timestamp": "last_reading"})

        out = latest.merge(last_reading, on="station_code", how="left")
        out["aqi"] = out["aqi"].astype("Int64")
        out["n_pollutants"] = out["n_pollutants"].astype("Int64")
        return out.rename(columns={"timestamp": "aqi_at"})[
            ["station_code", "aqi", "aqi_category", "dominant_pollutant", "aqi_reason",
             "n_pollutants", "last_reading", "aqi_at"]
        ]

    def stations(self, mappable_only: bool = True) -> pd.DataFrame:
        """Stations with location and their latest AQI. This is the map's data.

        Scoring every station takes a few seconds, so the result is held until
        new data arrives. Ages are recomputed on each call, never cached.
        """
        key = self.as_of()
        if key not in self._station_cache:
            self._station_cache = {key: self._build_stations()}
        table = self._station_cache[key]
        if mappable_only:
            table = table[table["is_mappable"]]
        return self._add_age(table.reset_index(drop=True))

    def _build_stations(self) -> pd.DataFrame:
        """Every station joined to its latest AQI. Cached by `stations()`."""
        meta = self.con.execute("""
            SELECT station_code, city, state_code, latitude, longitude, elevation_m,
                   is_mappable, has_real_city
            FROM registry
            ORDER BY station_code
        """).fetch_df()
        scored = self._latest_aqi(meta["station_code"].tolist())
        return meta.merge(scored, on="station_code", how="left")

    def _stations_uncached(self, mappable_only: bool = True) -> pd.DataFrame:
        """Same as `stations()` but always recomputed — used by the tests."""
        where = "WHERE is_mappable" if mappable_only else ""
        meta = self.con.execute(f"""
            SELECT station_code, city, state_code, latitude, longitude, elevation_m,
                   is_mappable, has_real_city
            FROM registry {where}
            ORDER BY station_code
        """).fetch_df()

        scored = self._latest_aqi(meta["station_code"].tolist())
        out = meta.merge(scored, on="station_code", how="left")
        return self._add_age(out)

    def station(self, station_code: str) -> dict | None:
        """One station: metadata, latest AQI and its current pollutant values."""
        meta = self.con.execute("""
            SELECT station_code, city, state_code, latitude, longitude, elevation_m,
                   is_mappable, has_real_city
            FROM registry WHERE station_code = ?
        """, [station_code]).fetch_df()
        if meta.empty:
            return None

        scored = self._latest_aqi([station_code])
        detail = self._add_age(meta.merge(scored, on="station_code", how="left")).iloc[0].to_dict()

        latest = self.con.execute("""
            SELECT * FROM hourly WHERE station_code = ? ORDER BY hour DESC LIMIT 1
        """, [station_code]).fetch_df()
        if not latest.empty:
            row = latest.iloc[0]
            detail["pollutants"] = {
                name: (None if pd.isna(row[name]) else round(float(row[name]), 2))
                for name in aqi_engine.BREAKPOINTS
                if name in latest.columns
            }
        return detail

    def history(self, station_code: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Hourly readings for a date range, oldest first."""
        return self.con.execute("""
            SELECT hour AS timestamp, pm2_5, pm10, no2, so2, nh3, co, o3, "at", rh, ws, wd, readings
            FROM hourly
            WHERE station_code = ? AND hour >= ? AND hour <= ?
            ORDER BY hour
        """, [station_code, start, end]).fetch_df()

    def cities(self, comparable_within_hours: int = COMPARABLE_WINDOW_HOURS) -> tuple[pd.DataFrame, dict]:
        """City rollup, ranked by AQI.

        Only stations whose newest reading is close to the archive's newest hour
        are included. Ranking a station last read in mid-December against one
        read on the 31st is not a comparison, and in this archive 61 of 546
        stations are more than a day behind.

        Returns the rollup and a note saying what was left out and why.
        """
        stations = self.stations(mappable_only=True)
        usable = stations[
            stations["has_real_city"]
            & stations["aqi"].notna()
            & (stations["hours_behind_latest"] <= comparable_within_hours)
        ]

        excluded = stations[
            stations["has_real_city"]
            & stations["aqi"].notna()
            & (stations["hours_behind_latest"] > comparable_within_hours)
        ]
        note = {
            "as_of": self.as_of().isoformat(),
            "comparable_within_hours": comparable_within_hours,
            "stations_used": int(len(usable)),
            "stations_excluded_as_not_comparable": int(len(excluded)),
            "basis": "Stations read more than the window behind the archive's newest hour are excluded, "
                     "because their readings describe a different period.",
        }

        columns = ["city", "state_code", "stations", "aqi", "aqi_category", "last_reading"]
        if usable.empty:
            return pd.DataFrame(columns=columns), note

        rollup = (
            usable.groupby(["city", "state_code"], as_index=False)
            .agg(stations=("station_code", "count"), aqi=("aqi", "mean"), last_reading=("last_reading", "max"))
        )
        rollup = rollup[rollup["stations"] >= MIN_STATIONS_FOR_CITY]
        rollup["aqi"] = rollup["aqi"].round().astype(int)
        rollup["aqi_category"] = [aqi_engine.category_for(v)[0] for v in rollup["aqi"]]
        return rollup.sort_values("aqi", ascending=False).reset_index(drop=True)[columns], note

    # ------------------------------------------------------------------ health

    def health(self) -> dict:
        """What the service knows about its own data. Served as-is by /health."""
        stats = self.stats()
        age = datetime.now() - stats.last_reading
        return {
            "stations": stats.stations,
            "mappable_stations": stats.mappable,
            "hourly_rows": stats.hourly_rows,
            "first_reading": stats.first_reading.isoformat(),
            "last_reading": stats.last_reading.isoformat(),
            "data_age_hours": round(age.total_seconds() / 3600, 1),
            # True once a live feed is wired up (LI-01). Today this is an archive.
            "is_live": age < timedelta(hours=6),
        }

    # ------------------------------------------------------------------ helpers

    def _add_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add two different ages. Nothing stale is ever presented as current.

        age_hours            — against the clock, so the UI can say "this is an archive"
        hours_behind_latest  — against the newest hour in the data, which is what
                               decides whether two stations can be compared
        """
        if "last_reading" not in df.columns:
            return df
        df = df.copy()
        last = pd.to_datetime(df["last_reading"])
        df["age_hours"] = ((pd.Timestamp.now() - last).dt.total_seconds() / 3600).round(1)
        df["hours_behind_latest"] = ((pd.Timestamp(self.as_of()) - last).dt.total_seconds() / 3600).round(1)
        df["is_stale"] = df["age_hours"] > 6
        return df

    def close(self) -> None:
        self.con.close()
