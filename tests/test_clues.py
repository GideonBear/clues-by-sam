from __future__ import annotations

import pytest

from clues_by_sam.clues import (
    CRIMINAL,
    INNOCENT,
    Above,
    Below,
    Between,
    Clue,
    Column,
    Combined,
    Connected,
    Edges,
    Exact,
    Neighboring,
    Not,
    OnlyOne,
    OnlyOnePerson,
    Overlap,
    Parity,
    Person,
    RegionClue,
    Row,
)


@pytest.mark.parametrize(
    ("clue", "expected"),
    [
        (
            "Only 1 of the 2 criminals neighboring Kay is in column A",
            Combined(
                RegionClue(Neighboring(Person("Kay")), Exact(CRIMINAL, 2)),
                RegionClue(
                    Overlap(Neighboring(Person("Kay")), Column(0)), Exact(CRIMINAL, 1)
                ),
            ),
        ),
        (
            "There are exactly 2 innocents below Martin",
            RegionClue(Below(Person("Martin")), Exact(INNOCENT, 2)),
        ),
        (
            "There's an odd number of criminals in column D",
            RegionClue(Column(3), Parity(CRIMINAL, 1)),
        ),
        (
            "All innocents below Eve are connected",
            RegionClue(Below(Person("Eve")), Connected(INNOCENT)),
        ),
        (
            "Only 1 of the 3 criminals neighboring Martin is on the edges",
            Combined(
                RegionClue(Neighboring(Person("Martin")), Exact(CRIMINAL, 3)),
                RegionClue(
                    Overlap(Neighboring(Person("Martin")), Edges()),
                    Exact(CRIMINAL, 1),
                ),
            ),
        ),
        (
            "There's an odd number of criminals in column A",
            RegionClue(Column(0), Parity(CRIMINAL, 1)),
        ),
        (
            "There's an odd number of innocents below Betty",
            RegionClue(Below(Person("Betty")), Parity(INNOCENT, 1)),
        ),
        (
            "An odd number of innocents above Vera neighbor Martin",
            RegionClue(
                Overlap(Above(Person("Vera")), Neighboring(Person("Martin"))),
                Parity(INNOCENT, 1),
            ),
        ),
        (
            "Only one row has exactly 2 innocents",
            OnlyOne(
                RegionClue(Row(0), Exact(INNOCENT, 2)),
                RegionClue(Row(1), Exact(INNOCENT, 2)),
                RegionClue(Row(2), Exact(INNOCENT, 2)),
                RegionClue(Row(3), Exact(INNOCENT, 2)),
            ),
        ),
        (
            "Only one person in row 3 has exactly 3 criminal neighbors",
            OnlyOnePerson(Row(2), Neighboring, Exact(CRIMINAL, 3)),
        ),
        (
            "Column B is the only column with exactly 2 innocents",
            Combined(
                RegionClue(Column(1), Exact(INNOCENT, 2)),
                RegionClue(Column(0), Not(Exact(INNOCENT, 2))),
                RegionClue(Column(2), Not(Exact(INNOCENT, 2))),
                RegionClue(Column(3), Not(Exact(INNOCENT, 2))),
            ),
        ),
        (
            "Exactly 1 innocent in between Betty and Vera is neighboring Kay",
            RegionClue(
                Overlap(
                    Between(Person("Betty"), Person("Vera")),
                    Neighboring(Person("Kay")),
                ),
                Exact(INNOCENT, 1),
            ),
        ),
        (
            "Exactly 2 of the 4 innocents neighboring Will are in row 4",
            Combined(
                RegionClue(Neighboring(Person("Will")), Exact(INNOCENT, 4)),
                RegionClue(
                    Overlap(Neighboring(Person("Will")), Row(3)), Exact(INNOCENT, 2)
                ),
            ),
        ),
    ],
)
def test_parse_clues(clue: str, expected: Clue) -> None:
    assert Clue.parse(clue) == expected
