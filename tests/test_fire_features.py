"""Tests for the fire features (MD-01 extension).

These run on hand-built fires with known geometry, so the expected answers can
be checked against a map rather than against the code's own output. That means
the logic is proven before the FIRMS key exists.

Run:  pytest tests/test_fire_features.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.sources.firms import (  # noqa: E402
    BoundingBox,
    FirmsClient,
    MissingMapKey,
    date_chunks,
    drop_low_confidence,
    normalise_detections,
)
from pipeline.fire_features import (  # noqa: E402
    UPWIND_HALF_ANGLE_DEGREES,
    add_fire_lags,
    bearing_degrees,
    build_fire_features,
    daily_station_fires,
    daily_upwind_fires,
    haversine_km,
    station_cell_links,
    to_daily_grid,
)

# Delhi, and the Punjab burning belt roughly north-west of it.
DELHI = (28.61, 77.21)
PUNJAB = (30.90, 75.85)


# ------------------------------------------------------------------ geometry

def test_haversine_matches_a_known_distance():
    """Delhi to Punjab's burning belt is roughly 270 km."""
    km = float(haversine_km(np.array([DELHI[0]]), np.array([DELHI[1]]),
                            np.array([PUNJAB[0]]), np.array([PUNJAB[1]]))[0])
    assert 240 < km < 300


def test_haversine_of_a_point_to_itself_is_zero():
    km = haversine_km(np.array([28.6]), np.array([77.2]), np.array([28.6]), np.array([77.2]))
    assert km[0] == pytest.approx(0.0, abs=1e-6)


def test_bearing_north_is_zero_and_east_is_ninety():
    north = bearing_degrees(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    east = bearing_degrees(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert north[0] == pytest.approx(0, abs=0.5)
    assert east[0] == pytest.approx(90, abs=0.5)


def test_punjab_is_north_west_of_delhi():
    """Sanity check on the real geography the upwind logic depends on."""
    bearing = float(bearing_degrees(np.array([DELHI[0]]), np.array([DELHI[1]]),
                                    np.array([PUNJAB[0]]), np.array([PUNJAB[1]]))[0])
    assert 280 < bearing < 340, f"expected north-west, got {bearing:.0f} degrees"


# ------------------------------------------------------------------ gridding

def test_detections_collapse_onto_a_daily_grid():
    detections = pd.DataFrame({
        "latitude": [30.90, 30.91, 30.92],
        "longitude": [75.85, 75.86, 75.84],
        "acq_date": pd.to_datetime(["2024-11-05"] * 3),
        "frp": [10.0, 20.0, 5.0],
    })
    grid = to_daily_grid(detections)
    # All three sit in one 0.25 degree cell on one day.
    assert len(grid) == 1
    assert grid["fire_count"].iloc[0] == 3
    assert grid["total_frp"].iloc[0] == 35.0


def test_different_days_stay_separate():
    detections = pd.DataFrame({
        "latitude": [30.9, 30.9],
        "longitude": [75.85, 75.85],
        "acq_date": pd.to_datetime(["2024-11-05", "2024-11-06"]),
        "frp": [10.0, 10.0],
    })
    assert len(to_daily_grid(detections)) == 2


def test_empty_detections_give_an_empty_grid_not_an_error():
    assert to_daily_grid(pd.DataFrame()).empty


# ------------------------------------------------------------------ distance bands

def _stations() -> pd.DataFrame:
    return pd.DataFrame({"station_code": ["DELHI"], "latitude": [DELHI[0]], "longitude": [DELHI[1]]})


def test_far_away_fires_are_excluded():
    """A fire in Kerala must not appear in Delhi's exposure."""
    detections = pd.DataFrame({
        "latitude": [10.0], "longitude": [76.0],
        "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [500.0],
    })
    links = station_cell_links(_stations(), to_daily_grid(detections))
    assert links.empty


def test_a_punjab_fire_lands_in_a_far_band_not_a_near_one():
    """Punjab's burning belt is ~286 km from Delhi, so it must not look local."""
    detections = pd.DataFrame({
        "latitude": [PUNJAB[0]], "longitude": [PUNJAB[1]],
        "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [100.0],
    })
    cells = to_daily_grid(detections)
    daily = daily_station_fires(cells, station_cell_links(_stations(), cells))

    far = [c for c in daily.columns if c.startswith("fire_frp_") and c != "fire_frp_0_100km"]
    assert any(daily[c].iloc[0] == 100.0 for c in far)
    assert "fire_frp_0_100km" not in daily.columns or daily["fire_frp_0_100km"].iloc[0] == 0.0


def test_gridding_can_shift_a_fire_across_a_band_boundary():
    """Documents a real limitation rather than leaving it to be discovered.

    Delhi to Punjab is 286 km, inside the 100-300 km band. Snapping the fire to
    a 0.25 degree cell moves it to 301 km, so it is reported one band out. The
    bands are modelling choices rather than physical thresholds, and the model
    learns what each is worth, so a shift of this size is tolerable — but it is
    not an accident and should not be mistaken for one.
    """
    detections = pd.DataFrame({
        "latitude": [PUNJAB[0]], "longitude": [PUNJAB[1]],
        "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [100.0],
    })
    raw_km = float(haversine_km(np.array([DELHI[0]]), np.array([DELHI[1]]),
                                np.array([PUNJAB[0]]), np.array([PUNJAB[1]]))[0])
    links = station_cell_links(_stations(), to_daily_grid(detections))
    gridded_km = float(links["distance_km"].iloc[0])

    assert abs(gridded_km - raw_km) < 25, "gridding moved the fire further than one cell could explain"
    assert raw_km < 300 <= gridded_km


def test_a_nearby_fire_lands_in_the_inner_band():
    detections = pd.DataFrame({
        "latitude": [29.0], "longitude": [77.3],
        "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [42.0],
    })
    cells = to_daily_grid(detections)
    daily = daily_station_fires(cells, station_cell_links(_stations(), cells))
    assert daily["fire_frp_0_100km"].iloc[0] == 42.0


# ------------------------------------------------------------------ upwind

def _punjab_fire_cells() -> pd.DataFrame:
    return to_daily_grid(pd.DataFrame({
        "latitude": [PUNJAB[0]], "longitude": [PUNJAB[1]],
        "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [100.0],
    }))


def test_a_fire_the_wind_comes_from_counts():
    """North-westerly wind carries Punjab's smoke to Delhi. This is the real case."""
    cells = _punjab_fire_cells()
    links = station_cell_links(_stations(), cells)
    wind = pd.DataFrame({
        "station_code": ["DELHI"], "acq_date": pd.to_datetime(["2024-11-05"]),
        "wind_from_deg": [315.0],   # from the north-west
    })
    upwind = daily_upwind_fires(cells, links, wind)
    assert not upwind.empty
    assert upwind["fire_frp_upwind"].iloc[0] > 0


def test_the_same_fire_downwind_does_not_count():
    """The discriminating test: identical fire, opposite wind, no exposure."""
    cells = _punjab_fire_cells()
    links = station_cell_links(_stations(), cells)
    wind = pd.DataFrame({
        "station_code": ["DELHI"], "acq_date": pd.to_datetime(["2024-11-05"]),
        "wind_from_deg": [135.0],   # from the south-east, so Punjab is downwind
    })
    assert daily_upwind_fires(cells, links, wind).empty


def test_upwind_angle_is_a_cone_not_a_ray():
    """Plumes spread, so a fire slightly off the wind axis still counts."""
    cells = _punjab_fire_cells()
    links = station_cell_links(_stations(), cells)
    bearing = float(links["bearing_deg"].iloc[0])
    just_inside = (bearing + UPWIND_HALF_ANGLE_DEGREES - 5) % 360
    wind = pd.DataFrame({
        "station_code": ["DELHI"], "acq_date": pd.to_datetime(["2024-11-05"]),
        "wind_from_deg": [just_inside],
    })
    assert not daily_upwind_fires(cells, links, wind).empty


def test_upwind_fires_are_reported_per_distance_band():
    """No inverse-distance weighting, deliberately.

    An earlier version weighted upwind fires by FRP / distance. Measured against
    Delhi PM2.5 in October-November, fires within 100km correlate at about
    r=-0.01 while fires at 100-300km reach r=+0.43 — so that weighting gave a
    10km fire thirty times the influence of the Punjab fires that actually drive
    the smoke. Bands let the model learn the distance response instead of having
    my guess baked in.
    """
    near = pd.DataFrame({
        "latitude": [29.0], "longitude": [77.3],
        "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [100.0],
    })
    cells = to_daily_grid(near)
    links = station_cell_links(_stations(), cells)
    bearing = float(links["bearing_deg"].iloc[0])
    wind = pd.DataFrame({"station_code": ["DELHI"], "acq_date": pd.to_datetime(["2024-11-05"]),
                         "wind_from_deg": [bearing]})
    out = daily_upwind_fires(cells, links, wind)

    # A nearby fire lands in the near band and nowhere else.
    assert out["fire_frp_upwind_0_100km"].iloc[0] == 100.0
    assert out.get("fire_frp_upwind_100_300km", pd.Series([0.0])).iloc[0] == 0.0
    # The unbanded summary is still provided, unweighted.
    assert out["fire_frp_upwind"].iloc[0] == 100.0


def test_upwind_totals_are_raw_frp_not_distance_weighted():
    """Two identical fires at different distances contribute the same FRP total."""
    def total(lat, lon):
        cells = to_daily_grid(pd.DataFrame({
            "latitude": [lat], "longitude": [lon],
            "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [100.0],
        }))
        links = station_cell_links(_stations(), cells)
        bearing = float(links["bearing_deg"].iloc[0])
        wind = pd.DataFrame({"station_code": ["DELHI"], "acq_date": pd.to_datetime(["2024-11-05"]),
                             "wind_from_deg": [bearing]})
        return float(daily_upwind_fires(cells, links, wind)["fire_frp_upwind"].iloc[0])

    assert total(29.0, 76.9) == total(PUNJAB[0], PUNJAB[1]) == 100.0


# ------------------------------------------------------------------ lags

def test_fire_lags_look_backwards_only():
    daily = pd.DataFrame({
        "station_code": ["X"] * 4,
        "acq_date": pd.to_datetime(["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04"]),
        "fire_frp_0_100km": [10.0, 20.0, 30.0, 40.0],
    })
    out = add_fire_lags(daily)
    assert out["fire_frp_0_100km_d1"].iloc[2] == 20.0
    assert pd.isna(out["fire_frp_0_100km_d1"].iloc[0])


def test_fire_lags_do_not_cross_stations():
    daily = pd.DataFrame({
        "station_code": ["A", "A", "B", "B"],
        "acq_date": pd.to_datetime(["2024-11-01", "2024-11-02"] * 2),
        "fire_frp_0_100km": [1.0, 2.0, 100.0, 200.0],
    })
    out = add_fire_lags(daily)
    first_b = out[(out["station_code"] == "B") & (out["acq_date"] == pd.Timestamp("2024-11-01"))]
    assert pd.isna(first_b["fire_frp_0_100km_d1"].iloc[0])


# ------------------------------------------------------------------ end to end

def test_build_fire_features_produces_joinable_rows():
    detections = pd.DataFrame({
        "latitude": [PUNJAB[0], 29.0],
        "longitude": [PUNJAB[1], 77.3],
        "acq_date": pd.to_datetime(["2024-11-05", "2024-11-05"]),
        "frp": [100.0, 40.0],
    })
    wind = pd.DataFrame({"station_code": ["DELHI"], "acq_date": pd.to_datetime(["2024-11-05"]),
                         "wind_from_deg": [315.0]})
    out = build_fire_features(detections, _stations(), wind)
    assert {"station_code", "acq_date"} <= set(out.columns)
    assert any(c.startswith("fire_") for c in out.columns)


def test_no_fires_anywhere_is_not_an_error():
    assert build_fire_features(pd.DataFrame(), _stations()).empty


# ------------------------------------------------------------------ client

def test_missing_map_key_fails_with_instructions():
    client = FirmsClient("/tmp/does-not-matter", map_key="")
    with pytest.raises(MissingMapKey, match="firms.modaps.eosdis.nasa.gov"):
        client.require_key()


def test_date_chunks_respect_the_five_day_limit():
    from datetime import date as d
    chunks = date_chunks(d(2024, 1, 1), d(2024, 1, 31))
    assert all(days <= 5 for _, days in chunks)
    assert sum(days for _, days in chunks) == 31
    assert chunks[0][0] == d(2024, 1, 1)


def test_date_chunks_cover_a_single_day():
    from datetime import date as d
    assert date_chunks(d(2024, 1, 1), d(2024, 1, 1)) == [(d(2024, 1, 1), 1)]


def test_bounding_box_formats_west_south_east_north():
    assert BoundingBox(67.0, 5.0, 99.0, 38.5).as_param() == "67.0,5.0,99.0,38.5"


def test_normalise_handles_a_missing_column():
    raw = pd.DataFrame({"latitude": [30.9], "longitude": [75.8], "acq_date": ["2024-11-05"]})
    out = normalise_detections(raw)
    assert out["frp"].iloc[0] == 0.0
    assert pd.notna(out["acq_date"].iloc[0])


def test_low_confidence_detections_are_dropped():
    frame = pd.DataFrame({"confidence": ["l", "n", "h"], "frp": [1.0, 2.0, 3.0]})
    assert len(drop_low_confidence(frame)) == 2


def test_cache_path_is_stable_for_the_same_chunk():
    """fetch_all needs to find a cached chunk without refetching it."""
    from datetime import date as d
    client = FirmsClient("/tmp/firms-test", map_key="x")
    first = client.cache_path(d(2024, 11, 5), 5)
    second = client.cache_path(d(2024, 11, 5), 5)
    assert first == second
    assert client.cache_path(d(2024, 11, 10), 5) != first


def test_combine_chunks_drops_duplicate_detections(tmp_path):
    """Chunk edges overlap, so the same fire can be written twice."""
    from pipeline.fetch_fires import combine_chunks

    row = {"latitude": [30.9], "longitude": [75.8], "acq_date": pd.to_datetime(["2024-11-05"]),
           "frp": [10.0], "confidence": ["n"], "daynight": ["D"]}
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    pd.DataFrame(row).to_parquet(first, index=False)
    pd.DataFrame(row).to_parquet(second, index=False)

    assert len(combine_chunks([first, second])) == 1


def test_combine_chunks_skips_files_that_are_not_there(tmp_path):
    """A failed chunk leaves no file, and that must not stop the rest combining."""
    from pipeline.fetch_fires import combine_chunks

    present = tmp_path / "present.parquet"
    pd.DataFrame({"latitude": [30.9], "longitude": [75.8],
                  "acq_date": pd.to_datetime(["2024-11-05"]), "frp": [10.0]}).to_parquet(present, index=False)
    out = combine_chunks([present, tmp_path / "missing.parquet"])
    assert len(out) == 1
