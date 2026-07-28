from utils import format_time


def test_format_time_seconds_only():
    assert format_time(45) == "45s"


def test_format_time_zero():
    assert format_time(0) == "0s"


def test_format_time_minutes_and_seconds():
    assert format_time(65) == "1m 05s"


def test_format_time_hours_minutes_seconds():
    assert format_time(3661) == "1h 01m 01s"


def test_format_time_clamps_negative_to_zero():
    assert format_time(-5) == "0s"
