from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from string import ascii_uppercase
from typing import TYPE_CHECKING, Self, override

from clues_by_sam.utils import splitlist, splitlist_by_subseq


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Person:
    name: str

    def __post_init__(self) -> None:
        if not self.name[0].isupper() and self.name[1:].islower():
            msg = f"Invalid person name: '{self.name}'"
            raise ValueError(msg)

    def __str__(self) -> str:
        return self.name


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


ROWS_AND_COLUMNS = 4


class Field(UserList[list[Person]]):
    def find(self, target: Person) -> tuple[int, int]:
        for i, row in enumerate(self):
            for j, person in enumerate(row):
                if person == target:
                    return i, j

        msg = f"Person '{target}' not in field"
        raise ValueError(msg)

    def column(self, j: int) -> list[Person]:
        return [row[j] for row in self]

    def get_columns(self) -> list[list[Person]]:
        columns: list[list[Person]] = [[]] * self.columns
        for row in self:
            for j, person in enumerate(row):
                columns[j].append(person)
        return columns

    @property
    def rows(self) -> int:
        return len(self)

    @property
    def columns(self) -> int:
        return len(self[0])

    def all(self) -> Iterator[Person]:
        for row in self:
            yield from row


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


@dataclass(frozen=True, init=False)
class Overlap(Region):
    regions: frozenset[Region]

    def __init__(self, *regions: Region) -> None:
        object.__setattr__(self, "regions", frozenset(regions))

    def people(self, field: Field) -> Iterable[Person]:
        a: set[Person] = reduce(
            set.union, (set(region.people(field)) for region in self.regions)
        )
        return a


@dataclass(frozen=True)
class AllExcept(Region):
    exception: Region

    def people(self, field: Field) -> Iterable[Person]:
        return set(field.all()) - set(self.exception.people(field))


@dataclass(frozen=True)
class Above(Region):
    person: Person

    def people(self, field: Field) -> Iterable[Person]:
        i, j = field.find(self.person)
        return field[i][:j]


@dataclass(frozen=True)
class Below(Region):
    person: Person

    def people(self, field: Field) -> Iterable[Person]:
        i, j = field.find(self.person)
        return field[i][j + 1 :]


@dataclass(frozen=True)
class Left(Region):
    person: Person

    def people(self, field: Field) -> Iterable[Person]:
        i, j = field.find(self.person)
        return field.column(j)[i + 1 :]


@dataclass(frozen=True)
class Right(Region):
    person: Person

    def people(self, field: Field) -> Iterable[Person]:
        i, j = field.find(self.person)
        return field.column(j)[:i]


@dataclass(frozen=True)
class Between(Region):
    a: Person
    b: Person

    def people(self, field: Field) -> Iterable[Person]:
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
class Row(Region):
    row: int

    def people(self, field: Field) -> Iterable[Person]:
        return field[self.row]

    @classmethod
    def parse(cls, row: str) -> Self:
        return cls(int(row) - 1)


@dataclass(frozen=True)
class Column(Region):
    column: int

    def people(self, field: Field) -> Iterable[Person]:
        return field.column(self.column)

    @classmethod
    def parse(cls, column: str) -> Self:
        return cls(ascii_uppercase.index(column))


class Constraint(ABC): ...  # ruff: ignore[abstract-base-class-without-abstract-method] TODO


@dataclass(frozen=True)
class Exact(Constraint):
    typ: Verdict
    amount: int


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


@dataclass(frozen=True)
class Connected(Constraint):
    typ: Verdict


@dataclass(frozen=True)
class Not(Constraint):
    constraint: Constraint


class Clue(ABC):  # ruff: ignore[abstract-base-class-without-abstract-method] TODO
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
                return RegionClue(
                    Region.parse_region(region), Connected(Verdict.parse(verdict))
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
                elif "are" in region_is_region and "in" in region_is_region:
                    total_region_s, spec_region_s = splitlist_by_subseq(
                        region_is_region, ("are", "in")
                    )
                else:
                    msg = (
                        f"Expected to find 'is' or 'are in' in this combined clue: "
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
                return OnlyOne(
                    *(
                        RegionClue(
                            region_type(i), Exact(Verdict.parse(verdict), int(amount))
                        )
                        for i in range(ROWS_AND_COLUMNS)
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
                        for i in range(ROWS_AND_COLUMNS)
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


@dataclass(frozen=True)
class Known(Clue):
    person: Person
    verdict: Verdict


@dataclass(frozen=True)
class RegionClue(Clue):
    region: Region
    constraint: Constraint


@dataclass(frozen=True)
class OnlyOnePerson(Clue):
    region: Region
    personal_region: type[Region]
    constraint: Constraint


@dataclass(frozen=True, init=False)
class OnlyOne(Clue):
    clues: frozenset[Clue]

    def __init__(self, *clues: Clue) -> None:
        object.__setattr__(self, "clues", frozenset(clues))
