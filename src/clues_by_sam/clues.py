from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from string import ascii_uppercase
from typing import TYPE_CHECKING, Self, override

import z3  # type: ignore[import-untyped]
from z3 import And, BoolRef, If, PbEq, Sum

from clues_by_sam.game import COLUMNS, ROWS, Field, Person
from clues_by_sam.utils import splitlist


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


class Verdict(Enum):
    INNOCENT = False
    CRIMINAL = True

    def __invert__(self) -> Self:
        return self.__class__(not self.value)

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def parse(cls, verdict: str) -> Self:
        verdict = verdict.removesuffix("s")
        return cls[verdict.upper()]


INNOCENT = Verdict.INNOCENT
CRIMINAL = Verdict.CRIMINAL


class Region(ABC):
    @abstractmethod
    def people(self, field: Field) -> Iterable[Person]: ...

    @classmethod
    def parse_region(cls, region: Sequence[str]) -> Region:  # ruff: ignore[complex-structure, too-many-branches]
        for i in range(1, len(region) + 1):
            found: Region | None = None
            match region[:i]:
                case ["on", "the", "edges"]:
                    found = Edges()
                case ["in", "column", column] | ["column", column]:
                    found = Column.parse(column)
                case ["in", "row", row] | ["row", row]:
                    found = Row.parse(row)
                case ["above", person]:
                    found = Above(Person(person))
                case ["below", person]:
                    found = Below(Person(person))
                case ["to", "the", "left", "of", person]:
                    found = Left(Person(person))
                case ["to", "the", "right", "of", person]:
                    found = Right(Person(person))
                case ["in", "between", a, "and", b]:
                    found = Between(Person(a), Person(b))
                case ["neighbor" | "neighbors" | "neighboring", person]:
                    found = Neighboring(Person(person))
                case _:
                    pass

            if found is not None:
                rest = region[i:]
                if len(rest) == 0:
                    return found
                if rest[0] == "is":
                    rest = rest[1:]
                return Overlap(found, cls.parse_region(rest))

        msg = f"Unknown region: '{" ".join(region)}'"
        raise ValueError(msg)


class ConnectedRegion(Region, ABC):
    @abstractmethod
    def people(self, field: Field) -> Sequence[Person]: ...

    @classmethod
    def parse_region(cls, region: Sequence[str]) -> ConnectedRegion:
        region_p = Region.parse_region(region)
        if not isinstance(region_p, ConnectedRegion):
            msg = f"Expected a connected region, got: '{" ".join(region)}'"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
        return region_p


@dataclass(frozen=True, init=False)
class Overlap(Region):
    regions: frozenset[Region]

    def __init__(self, *regions: Region) -> None:
        object.__setattr__(self, "regions", frozenset(regions))

    def people(self, field: Field) -> Iterable[Person]:
        a: set[Person] = reduce(
            set.intersection, (set(region.people(field)) for region in self.regions)
        )
        return a


@dataclass(frozen=True)
class AllExcept(Region):
    exception: Region

    def people(self, field: Field) -> Iterable[Person]:
        return set(field.all()) - set(self.exception.people(field))


@dataclass(frozen=True)
class Above(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field[i][:j]


@dataclass(frozen=True)
class Below(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field[i][j + 1 :]


@dataclass(frozen=True)
class Left(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field.column(j)[i + 1 :]


@dataclass(frozen=True)
class Right(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field.column(j)[:i]


@dataclass(frozen=True)
class Between(ConnectedRegion):
    a: Person
    b: Person

    def people(self, field: Field) -> Sequence[Person]:
        ai, aj = field.find(self.a)
        bi, bj = field.find(self.b)
        if ai == bi:
            return field[ai][aj + 1 : bj]
        if aj == bj:
            return field.column(aj)[ai + 1 : bi]

        msg = (
            f"Invalid 'between' clue: '{self.a}' and '{self.b}' "
            f"are not in the same row or column"
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class Neighboring(Region):
    person: Person

    def people(self, field: Field) -> Iterable[Person]:
        i, j = field.find(self.person)
        indexes = [
            (i - 1, j - 1),
            (i - 1, j),
            (i - 1, j + 1),
            (i, j - 1),
            (i, j + 1),
            (i + 1, j - 1),
            (i + 1, j),
            (i + 1, j + 1),
        ]
        return (
            field[i][j]
            for (i, j) in indexes
            if 0 <= i < field.rows and 0 <= j < field.columns
        )


@dataclass(frozen=True)
class Edges(Region):
    @override
    def people(self, field: Field) -> Iterable[Person]:
        return (
            {field[i][0] for i in range(field.rows)}
            | {field[i][field.columns - 1] for i in range(field.rows)}
            | {field[0][j] for j in range(field.columns)}
            | {field[field.rows - 1][j] for j in range(field.columns)}
        )


@dataclass(frozen=True)
class Row(ConnectedRegion):
    row: int

    def people(self, field: Field) -> Sequence[Person]:
        return field[self.row]

    @classmethod
    def parse(cls, row: str) -> Self:
        return cls(int(row) - 1)


@dataclass(frozen=True)
class Column(ConnectedRegion):
    column: int

    def people(self, field: Field) -> Sequence[Person]:
        return field.column(self.column)

    @classmethod
    def parse(cls, column: str) -> Self:
        return cls(ascii_uppercase.index(column))


class ConnectedConstraint(ABC):
    @abstractmethod
    def z3(
        self, people: Sequence[Person], people_m: Mapping[Person, BoolRef]
    ) -> BoolRef: ...


class Constraint(ConnectedConstraint, ABC):
    @abstractmethod
    def z3(
        self, people: Iterable[Person], people_m: Mapping[Person, BoolRef]
    ) -> BoolRef: ...


@dataclass(frozen=True)
class Exact(Constraint):
    typ: Verdict
    amount: int

    def z3(
        self, people: Iterable[Person], people_m: Mapping[Person, BoolRef]
    ) -> BoolRef:
        people = tuple(people)
        count = Sum([If(people_m[person], 1, 0) for person in people])
        if self.typ == INNOCENT:
            count = len(people) - count
        return count == self.amount


@dataclass(frozen=True)
class Parity(Constraint):
    typ: Verdict
    parity: int

    @classmethod
    def parse(cls, verdict: str, parity_s: str) -> Self:
        if parity_s == "even":
            parity = 0
        elif parity_s == "odd":
            parity = 1
        else:
            msg = f"Parity should be one of 'even' or 'odd', got '{parity_s}'"
            raise ValueError(msg)
        return cls(Verdict.parse(verdict), parity)

    def z3(
        self, people: Iterable[Person], people_m: Mapping[Person, BoolRef]
    ) -> BoolRef:
        people = tuple(people)
        count = Sum([If(people_m[person], 1, 0) for person in people])
        if self.typ == INNOCENT:
            count = len(people) - count
        return count % 2 == self.parity


@dataclass(frozen=True)
class Connected(ConnectedConstraint):
    typ: Verdict

    def z3(
        self, people: Sequence[Person], people_m: Mapping[Person, BoolRef]
    ) -> BoolRef:
        constraints = []
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                for k in range(j + 1, len(people)):
                    if self.typ == CRIMINAL:
                        constraints.append(
                            Not(
                                And(
                                    people_m[people[i]],
                                    z3.Not(people_m[people[j]]),
                                    people_m[people[k]],
                                )
                            )
                        )
                    else:
                        constraints.append(
                            Not(
                                And(
                                    z3.Not(people_m[people[i]]),
                                    people_m[people[j]],
                                    z3.Not(people_m[people[k]]),
                                )
                            )
                        )
        return And(constraints)


@dataclass(frozen=True)
class Not(Constraint):
    constraint: Constraint

    def z3(
        self, people: Iterable[Person], people_m: Mapping[Person, BoolRef]
    ) -> BoolRef:
        return z3.Not(self.constraint.z3(people, people_m))


class Clue(ABC):
    @abstractmethod
    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef: ...

    @classmethod
    def parse(cls, clue_s: str) -> Clue:  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-return-statements]
        clue = clue_s.split()
        # Note that it does not matter what regions apply to:
        #  "An odd number of innocents above Vera neighbor Martin" means the same as
        #  "The number of innocents that are above Vera and neighbor Martin is odd"
        # Additionally, a clue such as:
        #  "Only 1 of the 2 criminals neighboring Kay is in column A" means the same as
        #  "There are 2 criminals neighboring Kay and there is 1 criminal that is
        #  neighboring Kay and in column A".
        #  The "only" is implied by "there is (exactly) 1 criminal".
        match clue:
            case [person, "is", "innocent" | "criminal" as verdict] | [
                person,
                "is",
                "a",
                "innocent" | "criminal" as verdict,
            ]:
                return Known(Person(person), Verdict.parse(verdict))

            case [
                "There",
                "is" | "are",
                "exactly",
                amount,
                "innocent" | "innocents" | "criminal" | "criminals" as verdict,
                *region,
            ]:
                return RegionClue(
                    Region.parse_region(region),
                    Exact(Verdict.parse(verdict), int(amount)),
                )

            case [
                "Exactly",
                amount,
                "innocent" | "innocents" | "criminal" | "criminals" as verdict,
                *region,
            ]:
                return RegionClue(
                    Region.parse_region(region),
                    Exact(Verdict.parse(verdict), int(amount)),
                )

            case [
                "An",
                "odd" | "even" as parity,
                "number",
                "of",
                "innocents" | "criminals" as verdict,
                *region,
            ] | [
                "There's",
                "an",
                "odd" | "even" as parity,
                "number",
                "of",
                "innocents" | "criminals" as verdict,
                *region,
            ]:
                return RegionClue(
                    Region.parse_region(region), Parity.parse(verdict, parity)
                )

            case [
                "All",
                "innocents" | "criminals" as verdict,
                *region,
                "are",
                "connected",
            ]:
                return ConnectedRegionClue(
                    ConnectedRegion.parse_region(region),
                    Connected(Verdict.parse(verdict)),
                )

            case [
                "Only" | "Exactly",
                spec_amount,
                "of",
                "the",
                total_amount,
                "innocents" | "criminals" as verdict,
                *region_is_region,
            ]:
                if "is" in region_is_region:
                    total_region_s, spec_region_s = splitlist(region_is_region, "is")
                elif "are" in region_is_region:
                    total_region_s, spec_region_s = splitlist(region_is_region, "are")
                else:
                    msg = (
                        f"Expected to find 'is' or 'are' in this combined clue: "
                        f"'{clue_s}'"
                    )
                    raise ValueError(msg)
                total_region = Region.parse_region(total_region_s)
                spec_region = Region.parse_region(spec_region_s)
                verdict_p = Verdict.parse(verdict)
                return Combined(
                    RegionClue(total_region, Exact(verdict_p, int(total_amount))),
                    RegionClue(
                        Overlap(total_region, spec_region),
                        Exact(verdict_p, int(spec_amount)),
                    ),
                )

            case [
                "Only",
                "one",
                "row" | "column" as typ,
                "has",
                "exactly",
                amount,
                "innocents" | "criminals" as verdict,
            ]:
                region_type = Row if typ == "row" else Column
                num = ROWS if typ == "row" else COLUMNS
                return OnlyOne(
                    *(
                        RegionClue(
                            region_type(i), Exact(Verdict.parse(verdict), int(amount))
                        )
                        for i in range(num)
                    )
                )

            case [
                "Column" | "Row" as typ_1,
                row_or_column,
                "is",
                "the",
                "only",
                typ_2,
                "with",
                "exactly",
                amount,
                "innocents" | "criminals" as verdict,
            ] if typ_1.lower() == typ_2:
                region_type = Row if typ_2 == "row" else Column
                num = ROWS if typ_2 == "row" else COLUMNS
                verdict_p = Verdict.parse(verdict)
                row_or_column_p = region_type.parse(row_or_column)
                return Combined(
                    *(
                        RegionClue(
                            region_type(i),
                            Exact(verdict_p, int(amount)),
                        )
                        if region_type(i) == row_or_column_p
                        else RegionClue(
                            region_type(i), Not(Exact(verdict_p, int(amount)))
                        )
                        for i in range(num)
                    )
                )

            case [
                "Only",
                "one",
                "person",
                *region,
                "has",
                "exactly",
                amount,
                "innocent" | "criminal" as verdict,
                "neighbors",
            ]:
                return OnlyOnePerson(
                    Region.parse_region(region),
                    Neighboring,
                    Exact(Verdict.parse(verdict), int(amount)),
                )

            case _:
                msg = f"Unknown clue: '{clue_s}'"
                raise ValueError(msg)


@dataclass(frozen=True, init=False)
class Combined(Clue):
    clues: frozenset[Clue]

    def __init__(self, *clues: Clue) -> None:
        object.__setattr__(self, "clues", frozenset(clues))

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return And([clue.z3(field, people) for clue in self.clues])


@dataclass(frozen=True)
class Known(Clue):
    person: Person
    verdict: Verdict

    @override
    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return people[self.person] == self.verdict.value


@dataclass(frozen=True)
class RegionClue(Clue):
    region: Region
    constraint: Constraint

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return self.constraint.z3(self.region.people(field), people)


@dataclass(frozen=True)
class ConnectedRegionClue(Clue):
    region: ConnectedRegion
    constraint: ConnectedConstraint

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return self.constraint.z3(self.region.people(field), people)


@dataclass(frozen=True)
class OnlyOnePerson(Clue):
    region: Region
    personal_region: Callable[[Person], Region]
    constraint: Constraint

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return PbEq(
            [
                RegionClue(self.personal_region(person), self.constraint).z3(
                    field, people
                )
                for person in self.region.people(field)
            ],
            1,
        )


@dataclass(frozen=True, init=False)
class OnlyOne(Clue):
    clues: frozenset[Clue]

    def __init__(self, *clues: Clue) -> None:
        object.__setattr__(self, "clues", frozenset(clues))

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return PbEq([(clue.z3(field, people), 1) for clue in self.clues], 1)
