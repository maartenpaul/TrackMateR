from __future__ import annotations

from pathlib import Path

import pytest

import trackmater as tm


EXAMPLE_XML = Path(__file__).resolve().parent.parent / "trackmater" / "data" / "ExampleTrackMateData.xml"


@pytest.fixture(scope="session")
def example_xml_path() -> Path:
    assert EXAMPLE_XML.exists(), f"Example XML not found at {EXAMPLE_XML}"
    return EXAMPLE_XML


@pytest.fixture(scope="session")
def trackmate_data(example_xml_path) -> tm.TrackMateData:
    data = tm.read_trackmate_xml(example_xml_path)
    assert data is not None
    return data


@pytest.fixture(scope="session")
def msd_result(trackmate_data) -> tm.MSDResult:
    result = tm.calculate_msd(trackmate_data.tracks, n=3, short=8)
    assert result is not None
    return result


@pytest.fixture(scope="session")
def jd_result(trackmate_data) -> tm.JDResult:
    result = tm.calculate_jd(trackmate_data, delta_t=1, n_pop=2)
    assert result is not None
    return result


@pytest.fixture(scope="session")
def fd_result(trackmate_data) -> tm.FDResult:
    result = tm.calculate_fd(trackmate_data)
    assert result is not None
    return result


@pytest.fixture(scope="session")
def td_result(trackmate_data) -> tm.TrackDensityResult:
    result = tm.calculate_track_density(trackmate_data, radius=1.5)
    assert result is not None
    return result
