from __future__ import annotations

import re
from pathlib import Path
from string import ascii_uppercase
from typing import TYPE_CHECKING, cast

from lark import Discard, Lark, Transformer

from clues_by_sam.clues import (
    CRIMINAL,
    INNOCENT,
    Above,
    All,
    Amount,
    AtLeast,
    Below,
    Between,
    Clue,
    Column,
    ColumnOf,
    Combined,
    ConditionalPersonConstraint,
    Connected,
    ConnectedRegion,
    ConnectedRegionClue,
    Constraint,
    Corners,
    Count,
    DirectlyAbove,
    DirectlyBelow,
    DirectlyLeft,
    DirectlyRight,
    Edges,
    Equal,
    Exact,
    ForEvery,
    ForEveryProfession,
    Known,
    Left,
    More,
    Neighboring,
    Not,
    OnlyX,
    OnlyXPeople,
    Overlap,
    Parity,
    PersonalRegion,
    ProfessionRegion,
    Region,
    RegionClue,
    Right,
    Row,
    RowOf,
    SimplePersonConstraint,
    SinglePerson,
    Verdict,
)
from clues_by_sam.game import COLUMNS, ROWS, Person, Profession


if TYPE_CHECKING:
    from lark.visitors import _DiscardType


grammar = Path(__file__).parent / "clues.lark"
parser = Lark(grammar.read_text(), g_regex_flags=re.IGNORECASE)


NUMBERS = {
    "no": 0,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


# ruff: disable[no-self-use, invalid-function-name]
class ToClue(Transformer):  # type: ignore[type-arg]  # ruff: ignore[too-many-public-methods]
    def __init__(self, me: Person) -> None:
        self.me_ = me
        super().__init__()

    def start(self, c: tuple[Clue]) -> Clue:
        (clue,) = c
        return clue

    # clue

    def known(self, c: tuple[Person, Verdict]) -> Clue:
        person, verdict = c
        return Known(person, verdict)

    def region_exact(self, c: tuple[Amount, Verdict, Region]) -> Clue:
        amount, verdict, region = c
        return RegionClue(region, Count(verdict, amount))

    def region_exact_2(self, c: tuple[Amount, Verdict, Region, Region]) -> Clue:
        amount, verdict, region_a, region_b = c
        return RegionClue(Overlap(region_a, region_b), Count(verdict, amount))

    def both_region_region(self, c: tuple[Verdict, Region, Region]) -> Clue:
        verdict, region_a, region_b = c
        return Combined(
            RegionClue(region_a, Count(verdict, Exact(2))),
            RegionClue(
                Overlap(region_a, region_b),
                Count(verdict, Exact(2)),
            ),
        )

    def parity(self, c: tuple[Parity, Verdict, Region]) -> Clue:
        parity, verdict, region = c
        return RegionClue(region, Count(verdict, parity))

    def parity_rev(self, c: tuple[Parity, Region, Verdict]) -> Clue:
        parity, region, verdict = c
        return self.parity((parity, verdict, region))

    def parity_2(self, c: tuple[Parity, Verdict, Region, Region]) -> Clue:
        parity, verdict, region, region_2 = c
        return RegionClue(Overlap(region, region_2), Count(verdict, parity))

    def x_regions_have_constraint(
        self,
        c: tuple[Amount, type[Row | Column], Constraint],
    ) -> Clue:
        amount, typ, constraint = c
        num = ROWS if typ == Row else COLUMNS
        return OnlyX((RegionClue(typ(i), constraint) for i in range(num)), amount)

    def one_region_has_constraint(
        self, c: tuple[Region, type[Row | Column], Constraint]
    ) -> Clue:
        region, typ, constraint = c
        if not isinstance(region, typ):
            msg = "Mismatch of row/column"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] the value is the clue

        num = ROWS if typ == Row else COLUMNS

        return Combined(
            *(
                RegionClue(typ(i), constraint)
                if typ(i) == region
                else RegionClue(typ(i), Not(constraint))
                for i in range(num)
            )
        )

    def row_col_more_than_any_other(
        self, c: tuple[Region, Verdict, type[Row | Column]]
    ) -> Clue:
        region, verdict, typ = c
        if not isinstance(region, typ):
            msg = "Mismatch of row/column"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] the value is the clue

        num = ROWS if typ == Row else COLUMNS
        return Combined(
            *(
                More(region, verdict, typ(i), verdict)
                for i in range(num)
                if typ(i) != region
            )
        )

    def each_row_col(self, c: tuple[type[Row | Column], Constraint]) -> Clue:
        typ, constraint = c
        num = ROWS if typ == Row else COLUMNS
        return Combined(*(RegionClue(typ(i), constraint) for i in range(num)))

    def connected(self, c: tuple[Verdict, ConnectedRegion]) -> Clue:
        verdict, region = c
        return ConnectedRegionClue(
            region,
            Connected(verdict),
        )

    def both_connected(self, c: tuple[Verdict, ConnectedRegion]) -> Clue:
        verdict, region = c
        return Combined(
            RegionClue(region, Count(verdict, Exact(2))),
            ConnectedRegionClue(region, Connected(verdict)),
        )

    def of_the(self, c: tuple[Amount, Amount, Verdict, Region, Region]) -> Clue:
        spec_amount, total_amount, verdict, total_region, spec_region = c
        return Combined(
            RegionClue(
                total_region,
                Count(verdict, total_amount),
            ),
            RegionClue(
                Overlap(total_region, spec_region),
                Count(verdict, spec_amount),
            ),
        )

    def of_the_profession(
        self, c: tuple[Amount, Amount, Profession, Region, Verdict]
    ) -> Clue:
        spec_amount, _total_amount, profession, region, verdict = c
        return RegionClue(
            Overlap(ProfessionRegion(profession), region), Count(verdict, spec_amount)
        )

    def of_the_neighbors(
        self, c: tuple[Amount, Person, Amount, Verdict, Region]
    ) -> Clue:
        spec_amount, total_person, total_amount, verdict, spec_region = c
        total_region = Neighboring(total_person)
        return Combined(
            RegionClue(
                total_region,
                Count(verdict, total_amount),
            ),
            RegionClue(
                Overlap(total_region, spec_region),
                Count(verdict, spec_amount),
            ),
        )

    def is_one_of(self, c: tuple[Person, Amount, Verdict, Region]) -> Clue:
        person, amount, verdict, region = c
        return Combined(
            Known(person, verdict),
            RegionClue(region, Count(verdict, amount)),
        )

    def is_one_of_neighbors(self, c: tuple[Person, Person, Amount, Verdict]) -> Clue:
        person_a, person_b, amount, verdict = c
        return Combined(
            Known(person_a, verdict),
            RegionClue(Neighboring(person_b), Count(verdict, amount)),
        )

    def everyone_exact_neighbors(self, c: tuple[Amount, Verdict]) -> Clue:
        amount, verdict = c
        return ForEvery(
            All(),
            SimplePersonConstraint(Neighboring, Count(verdict, amount)),
        )

    def exact_neighbors(self, c: tuple[Person, Amount, Verdict]) -> Clue:
        person, amount, verdict = c
        return RegionClue(Neighboring(person), Count(verdict, amount))

    def region_neighbors(self, c: tuple[Amount, Region, Constraint]) -> Clue:
        amount_a, region, constraint = c
        return OnlyXPeople(
            amount_a,
            region,
            SimplePersonConstraint(Neighboring, constraint),
        )

    def profession_neighbors(self, c: tuple[Amount, Profession, Constraint]) -> Clue:
        amount_a, profession, constraint = c
        return self.region_neighbors((
            amount_a,
            ProfessionRegion(profession),
            constraint,
        ))

    def only_in_region(self, c: tuple[Person, Verdict, Region]) -> Clue:
        person, verdict, region = c
        return Combined(
            Known(person, verdict),
            RegionClue(region, Count(verdict, Exact(1))),
        )

    def only_with_neighbors(self, c: tuple[Person, Constraint]) -> Clue:
        person, constraint = c
        return Combined(
            RegionClue(Neighboring(person), constraint),
            OnlyXPeople(
                Exact(1), All(), SimplePersonConstraint(Neighboring, constraint)
            ),
        )

    def equal_neighbors(self, c: tuple[Person, Person, Verdict]) -> Clue:
        person_a, person_b, verdict = c
        return Equal(
            Neighboring(person_a),
            verdict,
            Neighboring(person_b),
            verdict,
        )

    def equal_region_and_region(self, c: tuple[Verdict, tuple[Region, Region]]) -> Clue:
        verdict, (region_a, region_b) = c
        return Equal(region_a, verdict, region_b, verdict)

    def equal_verdict_region_and_verdict_region(
        self, c: tuple[Verdict, Region, Verdict, Region]
    ) -> Clue:
        verdict_a, region_a, verdict_b, region_b = c
        return Equal(region_a, verdict_a, region_b, verdict_b)

    def equal_verdicts(self, c: tuple[Region]) -> Clue:
        (region,) = c
        return Equal(region, INNOCENT, region, CRIMINAL)

    def more_verdict_than_region(self, c: tuple[Verdict, Region, Region]) -> Clue:
        verdict, region_a, region_b = c
        return More(
            region_a,
            verdict,
            region_b,
            verdict,
        )

    def more_verdict_than_verdict(self, c: tuple[Verdict, Verdict, Region]) -> Clue:
        verdict_a, verdict_b, region = c
        return More(
            region,
            verdict_a,
            region,
            verdict_b,
        )

    def more_verdict_region_than_verdict_region(
        self, c: tuple[Verdict, Region, Verdict, Region]
    ) -> Clue:
        verdict_a, region_a, verdict_b, region_b = c
        return More(
            region_a,
            verdict_a,
            region_b,
            verdict_b,
        )

    def personal_equal_verdicts(self, c: tuple[Person, PersonalRegion]) -> Clue:
        person, personal_region = c
        region = personal_region(person)
        return Equal(region, INNOCENT, region, CRIMINAL)

    def personal_more(self, c: tuple[Person, Verdict, Verdict, PersonalRegion]) -> Clue:
        person, verdict_a, verdict_b, personal_region = c
        region = personal_region(person)
        return More(region, verdict_a, region, verdict_b)

    def amount_personal_singular_region(
        self, c: tuple[Amount, Region, Verdict, PersonalRegion]
    ) -> Clue:
        amount, region, verdict, personal_region = c
        return OnlyXPeople(
            amount,
            region,
            SimplePersonConstraint(
                personal_region,
                Count(verdict, Exact(1)),
            ),
        )

    def profession_amount_personal_singular_region(
        self, c: tuple[Amount, Profession, Verdict, PersonalRegion]
    ) -> Clue:
        amount, profession, verdict, personal_region = c
        return self.amount_personal_singular_region((
            amount,
            ProfessionRegion(profession),
            verdict,
            personal_region,
        ))

    def amount_personal_singular_region_conditional(
        self, c: tuple[Amount, Verdict, Region, Verdict, PersonalRegion]
    ) -> Clue:
        amount, condition_verdict, region, verdict, personal_region = c
        return OnlyXPeople(
            amount,
            region,
            ConditionalPersonConstraint(
                personal_region,
                Count(verdict, Exact(1)),
                condition_verdict,
            ),
        )

    def common_neighbors(self, c: tuple[Person, Person, Amount, Verdict]) -> Clue:
        person_a, person_b, amount, verdict = c
        return RegionClue(
            Overlap(
                Neighboring(person_a),
                Neighboring(person_b),
            ),
            Count(verdict, amount),
        )

    def more_neighbors(self, c: tuple[Person, Verdict, Person]) -> Clue:
        person_a, verdict, person_b = c
        return More(
            Neighboring(person_a),
            verdict,
            Neighboring(person_b),
            verdict,
        )

    def among_all_professions(self, c: tuple[Constraint]) -> Clue:
        (constraint,) = c
        return ForEveryProfession(constraint)

    # region / region_2

    def all(self, c: tuple[()]) -> Region:
        () = c
        return All()

    def edges(self, c: tuple[()]) -> Region:
        () = c
        return Edges()

    def corners(self, c: tuple[()]) -> Region:
        () = c
        return Corners()

    def row_col_region(self, c: tuple[Region]) -> Region:
        (region,) = c
        return region

    def above(self, c: tuple[Person]) -> Region:
        (person,) = c
        return Above(person)

    def below(self, c: tuple[Person]) -> Region:
        (person,) = c
        return Below(person)

    def left(self, c: tuple[Person]) -> Region:
        (person,) = c
        return Left(person)

    def right(self, c: tuple[Person]) -> Region:
        (person,) = c
        return Right(person)

    def between(self, c: tuple[Person, Person]) -> Region:
        person_a, person_b = c
        return Between(person_a, person_b)

    def neighboring(self, c: tuple[Person]) -> Region:
        (person,) = c
        return Neighboring(person)

    def region_2(self, c: tuple[Region]) -> Region:
        (region,) = c
        return region

    # region_and_region

    def region_and_region(self, c: tuple[Region, Region]) -> tuple[Region, Region]:
        return c

    def row_and_row(self, c: tuple[int, int]) -> tuple[Region, Region]:
        a, b = c
        return Row(a), Row(b)

    def col_and_col(self, c: tuple[int, int]) -> tuple[Region, Region]:
        a, b = c
        return Column(a), Column(b)

    # personal_region_*

    def personal_neighboring(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return Neighboring

    def personal_above(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return Above

    def personal_below(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return Below

    def personal_left(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return Left

    def personal_right(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return Right

    def personal_between(self, c: tuple[Person]) -> PersonalRegion:
        (other,) = c
        return lambda person: Between(person, other)

    def personal_directly_above(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return lambda x: DirectlyAbove(SinglePerson(x))

    def personal_directly_below(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return lambda x: DirectlyBelow(SinglePerson(x))

    def personal_directly_left(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return lambda x: DirectlyLeft(SinglePerson(x))

    def personal_directly_right(self, c: tuple[()]) -> PersonalRegion:
        () = c
        return lambda x: DirectlyRight(SinglePerson(x))

    # row_col

    def row(self, c: tuple[int]) -> Region:
        (row,) = c
        return Row(row)

    def col(self, c: tuple[int]) -> Region:
        (col,) = c
        return Column(col)

    def my_row_col(self, c: tuple[type[Row | Column]]) -> Region:
        (typ,) = c
        typ_of = RowOf if typ == Row else ColumnOf
        return typ_of(self.me_)

    def ROW_COL_TYPE(self, c: str) -> type[Row | Column]:
        match c:
            case "row":
                return Row
            case "column":
                return Column
            case _:
                msg = "Unreachable"
                raise AssertionError(msg)

    # constraint

    def count(self, c: tuple[Amount, Verdict]) -> Constraint:
        amount, verdict = c
        return Count(verdict, amount)

    # amount

    def exact(self, c: tuple[int]) -> Amount:
        (amount,) = c
        return Exact(amount)

    def at_least(self, c: tuple[int]) -> Amount:
        (amount,) = c
        return AtLeast(amount)

    def more_than(self, c: tuple[int]) -> Amount:
        (amount,) = c
        return AtLeast(amount + 1)

    # x_people

    def zero(self, c: tuple[()]) -> Amount:
        () = c
        return Exact(0)

    def x_people(self, c: tuple[Amount]) -> Amount:
        (amount,) = c
        return amount

    # person / person_2 / person_is

    def me(self, c: tuple[()]) -> Person:
        () = c
        return self.me_

    def person(self, c: tuple[str]) -> Person:
        (person,) = c
        return Person(person)

    def person_2(self, c: tuple[str]) -> Person:
        (person,) = c
        return Person(person)

    def person_is(self, c: tuple[Person]) -> Person:
        (person,) = c
        return person

    def profession_p_region(self, c: tuple[Profession]) -> Region:
        (profession,) = c
        return ProfessionRegion(profession)

    def profession_p(self, c: tuple[Profession]) -> Profession:
        (profession,) = c
        return profession

    def PROFESSION(self, profession: str) -> Profession:
        return Profession(profession)

    def NUMBER(self, n: str) -> int:
        if n.lower() in NUMBERS:
            return NUMBERS[n.lower()]
        return int(n)

    def PARITY(self, p: str) -> Parity:
        match p:
            case "even":
                return Parity(0)
            case "odd":
                return Parity(1)
            case _:
                msg = "Unreachable"
                raise AssertionError(msg)

    def VERDICT(self, v: str) -> Verdict:
        match v:
            case "innocent":
                return INNOCENT
            case "criminal":
                return CRIMINAL
            case _:
                msg = "Unreachable"
                raise AssertionError(msg)

    def ROW(self, row: str) -> int:
        return int(row) - 1

    def COLUMN(self, column: str) -> int:
        return ascii_uppercase.index(column)

    def int_discard(self, _c: tuple[int]) -> _DiscardType:
        return Discard


# ruff: enable[no-self-use, invalid-function-name]


def parse_clue(s: str, me: Person) -> Clue:
    s = s.replace("\u00a0", " ")
    t = parser.parse(s)
    return cast("Clue", ToClue(me).transform(t))
