import datetime

from server.visitors import VisitorCounter


def test_counts_unique_visitors_and_readers(tmp_path, capsys):
    day = [datetime.date(2026, 11, 13)]
    counter = VisitorCounter(str(tmp_path), today=lambda: day[0])
    assert counter.record("1.2.3.4", "Firefox", False) == (1, 0)
    assert counter.record("1.2.3.4", "Firefox", True) == (1, 1)
    assert counter.record("1.2.3.4", "Firefox", True) == (1, 1)
    assert counter.record("5.6.7.8", "Firefox", False) == (2, 1)
    # Another worker shares the day's file
    other = VisitorCounter(str(tmp_path), today=lambda: day[0])
    assert other.record("1.2.3.4", "Firefox", False) == (2, 1)
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "visitors 2026-11-13: 2 (1 read something)"
    # Only counts are printed, never addresses
    assert "1.2.3.4" not in out
    # A new day starts again, and yesterday's file is deleted
    day[0] = datetime.date(2026, 11, 14)
    assert counter.record("1.2.3.4", "Firefox", False) == (1, 0)
    assert [path.name for path in tmp_path.iterdir()] == [
        "dots-and-circles-visitors-2026-11-14.txt"]
    assert "1.2.3.4" not in (tmp_path / "dots-and-circles-visitors-"
                             "2026-11-14.txt").read_text()
