from __future__ import annotations

import pytest

from clues_by_sam.clues import (
    CRIMINAL,
    INNOCENT,
    Above,
    AtLeast,
    Below,
    Between,
    Clue,
    Column,
    Combined,
    Connected,
    ConnectedRegionClue,
    DirectlyBelow,
    Edges,
    Equal,
    Exact,
    Known,
    Left,
    More,
    Neighboring,
    Not,
    OnlyOne,
    OnlyOnePerson,
    Overlap,
    Parity,
    ProfessionRegion,
    RegionClue,
    Right,
    Row,
)
from clues_by_sam.game import Person, Profession


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
            ConnectedRegionClue(Below(Person("Eve")), Connected(INNOCENT)),
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
                RegionClue(Row(4), Exact(INNOCENT, 2)),
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
        (
            "Kumar and Will have an equal number of innocent neighbors",
            Equal(
                Neighboring(Person("Kumar")),
                INNOCENT,
                Neighboring(Person("Will")),
                INNOCENT,
            ),
        ),
        (
            "Chris has more innocent neighbors than Xavi",
            More(
                Neighboring(Person("Chris")),
                INNOCENT,
                Neighboring(Person("Xavi")),
                INNOCENT,
            ),
        ),
        (
            "Each row has at least 3 innocents",
            Combined(
                RegionClue(Row(0), AtLeast(INNOCENT, 3)),
                RegionClue(Row(1), AtLeast(INNOCENT, 3)),
                RegionClue(Row(2), AtLeast(INNOCENT, 3)),
                RegionClue(Row(3), AtLeast(INNOCENT, 3)),
                RegionClue(Row(4), AtLeast(INNOCENT, 3)),
            ),
        ),
        (
            "John has exactly 7 innocent neighbors",
            RegionClue(Neighboring(Person("John")), Exact(INNOCENT, 7)),
        ),
        (
            "There's an equal number of innocents in rows 2 and 3",
            Equal(Row(1), INNOCENT, Row(2), INNOCENT),
        ),
        (
            "Quita is one of Mark's 4 criminal neighbors",
            Combined(
                Known(Person("Quita"), CRIMINAL),
                RegionClue(Neighboring(Person("Mark")), Exact(CRIMINAL, 4)),
            ),
        ),
        (
            "Gary and Mark have 3 innocent neighbors in common",
            RegionClue(
                Overlap(Neighboring(Person("Gary")), Neighboring(Person("Mark"))),
                Exact(INNOCENT, 3),
            ),
        ),
        (
            "There is only one innocent to the left of Carol",
            RegionClue(Left(Person("Carol")), Exact(INNOCENT, 1)),
        ),
        (
            "Ollie is one of 2 criminals in column C",
            Combined(
                Known(Person("Ollie"), CRIMINAL),
                RegionClue(Column(2), Exact(CRIMINAL, 2)),
            ),
        ),
        (
            "1 of the 2 criminals in column B is in between Mary and Quita",
            Combined(
                RegionClue(Column(1), Exact(CRIMINAL, 2)),
                RegionClue(
                    Overlap(Column(1), Between(Person("Mary"), Person("Quita"))),
                    Exact(CRIMINAL, 1),
                ),
            ),
        ),
        (
            "There are more innocents in column A than column B",
            More(Column(0), INNOCENT, Column(1), INNOCENT),
        ),
        (
            "I'm the only innocent to the right of Quita",
            Combined(
                Known(Person("Me"), INNOCENT),
                RegionClue(Right(Person("Quita")), Exact(INNOCENT, 1)),
            ),
        ),
        (
            "Both innocents above Wanda are connected",
            Combined(
                RegionClue(Above(Person("Wanda")), Exact(INNOCENT, 2)),
                ConnectedRegionClue(Above(Person("Wanda")), Connected(INNOCENT)),
            ),
        ),
        (
            "2 mechs have a criminal directly below them",
            RegionClue(
                DirectlyBelow(ProfessionRegion(Profession("mech"))), Exact(CRIMINAL, 2)
            ),
        ),
        (
            "There are as many criminal builders as there are criminal coders",
            Equal(
                ProfessionRegion(Profession("builder")),
                CRIMINAL,
                ProfessionRegion(Profession("coder")),
                CRIMINAL,
            ),
        ),
    ],
)
def test_parse_clues(clue: str, expected: Clue) -> None:
    assert Clue.parse(clue, Person("Me")) == expected
