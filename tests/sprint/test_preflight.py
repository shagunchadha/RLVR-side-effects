"""The host RAM/disk preflight is bypassed in unit tests (conftest `_no_host_preflight`, F0a) but stays real for jobs."""
import pytest

from exp import resources as R


def test_preflight_is_bypassed_in_unit_tests(tmp_path):
    assert R.preflight(10 ** 9, dirs=[tmp_path], min_ram_gb=10 ** 9) == {}      # would abort if it were real


@pytest.mark.real_preflight
def test_real_preflight_still_aborts(tmp_path):
    with pytest.raises(SystemExit, match="PREFLIGHT ABORT"):
        R.preflight(10 ** 9, dirs=[tmp_path], gpu=False)                        # 1e9 GB of free disk: impossible
