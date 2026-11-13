import pytest

pytest.importorskip("PIL")  # ocr.training_data imports it

from ocr.training_data.add_norman import verb_notes

TEXT = """DOSIMBI (-ka) 1. to enter, to advance
JEMBI (1) (imperative jefu, -ke, -tere) to eat
JEMBI (2) (-nggke, -ndere, -mpi) to bear, to tolerate
JIMBI (imperative: jio, -he, -dere) to come, see also jidere
DARIMBI (-ka/ha) to touch lightly
ILIMBAHAMBI (-baha) to get used to
DUMBI (for tūmbi) to strike
GENEMBI to go
"""


def test_verb_notes(tmp_path):
    path = tmp_path / "norman.txt"
    path.write_text(TEXT, encoding="utf-8")
    notes = verb_notes(str(path))
    assert notes["dosimbi"] == {"perfective": ["-ka"],
                                "forms": [["dosika", ["-ka"]]]}
    # Numbered headwords are merged, and the OCR's -nggke mended
    assert notes["jembi"] == {"perfective": ["-ke", "-ngke"], "forms": [
        ["jefu", []], ["jeke", ["-ke"]], ["jetere", ["-tere"]],
        ["jengke", ["-ngke"]], ["jendere", ["-ndere"]], ["jempi", ["-mpi"]]]}
    assert notes["jimbi"]["forms"][0] == ["jio", []]
    # -ka/ha is both
    assert notes["darimbi"]["perfective"] == ["-ka", "-ha"]
    # An ending the stem already ends in
    assert notes["ilimbahambi"]["forms"] == [["ilimbaha", ["-baha"]]]
    # Notes saying nothing, and verbs without notes, are left out
    assert "dumbi" not in notes and "genembi" not in notes
