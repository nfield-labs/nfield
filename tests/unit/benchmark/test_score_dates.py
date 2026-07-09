"""Calendar-date equality rules in the benchmark scorer."""

from __future__ import annotations

from benchmark.scoring.score import _dates_match, _parse_dates


def _match(gold: str, predicted: str) -> bool:
    return _dates_match(_parse_dates(gold), _parse_dates(predicted))


class TestTimezoneShiftedTimestamps:
    def test_evening_utc_reads_as_next_local_day(self):
        assert _match("2019-02-28t23:00:00.000z", "march 2019")

    def test_morning_utc_stays_same_day(self):
        assert _match("2019-03-01t04:00:00z", "march 2019")
        assert not _match("2019-03-01t04:00:00z", "february 2019")

    def test_month_boundary_shift(self):
        assert _match("2019-03-31t23:00:00z", "april 2019")


class TestMonthYearGranularity:
    def test_iso_date_matches_its_month(self):
        assert _match("2019-03-01", "march 2019")

    def test_different_month_rejected(self):
        assert not _match("2019-02-10", "march 2019")

    def test_day_granular_pairs_need_the_same_day(self):
        assert not _match("2019-03-01", "2019-03-02")
        assert _match("2019-03-01", "1 march 2019")


class TestNonDatesUntouched:
    def test_period_labels_do_not_parse_as_dates(self):
        assert _parse_dates("fy2025 q2") == set()

    def test_plain_words_do_not_parse(self):
        assert _parse_dates("march of progress") == set()
