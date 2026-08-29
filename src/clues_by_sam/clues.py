from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Self, override

import z3  # type: ignore[import-not-found]
from z3 import And, BoolRef, If, IntNumRef, PbEq, PbGe, Sum

from clues_by_sam.game import Field, Person, Profession


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


class Verdict(Enum):
    INNOCENT = False
    CRIMINAL = True

    def __invert__(self) -> Self:
        return self.__class__(not self.value)

    def __repr__(self) -> str:
        return self.name


INNOCENT = Verdict.INNOCENT
CRIMINAL = Verdict.CRIMINAL


class Region(ABC):
    @abstractmethod
    def people(self, field: Field) -> Iterable[Person]: ...


class ConnectedRegion(Region, ABC):
    @abstractmethod
    def people(self, field: Field) -> Sequence[Person]: ...


@dataclass(frozen=True)
class All(Region):
    @override
    def people(self, field: Field) -> Iterable[Person]:
        return field.all()


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
class SinglePerson(Region):
    person: Person

    @override
    def people(self, field: Field) -> Iterable[Person]:
        yield self.person


@dataclass(frozen=True)
class ProfessionRegion(Region):
    profession: Profession

    def people(self, field: Field) -> Iterable[Person]:
        return (
            person
            for person in field.all()
            if field.professions[person] == self.profession
        )


@dataclass(frozen=True)
class Left(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field[i][:j]


@dataclass(frozen=True)
class Right(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field[i][j + 1 :]


@dataclass(frozen=True)
class Above(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field.column(j)[:i]


@dataclass(frozen=True)
class Below(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, j = field.find(self.person)
        return field.column(j)[i + 1 :]


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
class Corners(Region):
    @override
    def people(self, field: Field) -> Iterable[Person]:
        return (
            field[0][0],
            field[0][field.columns - 1],
            field[field.rows - 1][0],
            field[field.rows - 1][field.columns - 1],
        )


@dataclass(frozen=True)
class Row(ConnectedRegion):
    row: int

    def people(self, field: Field) -> Sequence[Person]:
        return field[self.row]


@dataclass(frozen=True)
class Column(ConnectedRegion):
    column: int

    def people(self, field: Field) -> Sequence[Person]:
        return field.column(self.column)


@dataclass(frozen=True)
class RowOf(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        i, _j = field.find(self.person)
        return Row(i).people(field)


@dataclass(frozen=True)
class ColumnOf(ConnectedRegion):
    person: Person

    def people(self, field: Field) -> Sequence[Person]:
        _i, j = field.find(self.person)
        return Column(j).people(field)


@dataclass(frozen=True)
class DirectlyLeft(Region):
    region: Region

    def people(self, field: Field) -> Iterable[Person]:
        for person in self.region.people(field):
            i, j = field.find(person)
            if j == 0:
                continue
            yield field[i][j - 1]


@dataclass(frozen=True)
class DirectlyRight(Region):
    region: Region

    def people(self, field: Field) -> Iterable[Person]:
        for person in self.region.people(field):
            i, j = field.find(person)
            if j == field.columns - 1:
                continue
            yield field[i][j + 1]


@dataclass(frozen=True)
class DirectlyAbove(Region):
    region: Region

    def people(self, field: Field) -> Iterable[Person]:
        for person in self.region.people(field):
            i, j = field.find(person)
            if i == 0:
                continue
            yield field[i - 1][j]


@dataclass(frozen=True)
class DirectlyBelow(Region):
    region: Region

    def people(self, field: Field) -> Iterable[Person]:
        for person in self.region.people(field):
            i, j = field.find(person)
            if i == field.rows - 1:
                continue
            yield field[i + 1][j]


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


class Amount(ABC):
    @abstractmethod
    def cmp(self, actual: IntNumRef) -> BoolRef: ...
    @abstractmethod
    def pb(self, args: Iterable[tuple[BoolRef, int]]) -> BoolRef: ...


@dataclass(frozen=True)
class Exact(Amount):
    amount: int

    def cmp(self, actual: IntNumRef) -> BoolRef:
        return actual == self.amount

    def pb(self, args: Iterable[tuple[BoolRef, int]]) -> BoolRef:
        return PbEq(args, self.amount)


@dataclass(frozen=True)
class AtLeast(Amount):
    amount: int

    def cmp(self, actual: IntNumRef) -> BoolRef:
        return actual >= self.amount

    def pb(self, args: Iterable[tuple[BoolRef, int]]) -> BoolRef:
        return PbGe(args, self.amount)


@dataclass(frozen=True)
class Parity(Amount):
    parity: int

    def cmp(self, actual: IntNumRef) -> BoolRef:
        return actual % 2 == self.parity

    def pb(self, args: Iterable[tuple[BoolRef, int]]) -> BoolRef:
        raise NotImplementedError


def count(
    people: Iterable[Person], people_m: Mapping[Person, BoolRef], verdict: Verdict
) -> IntNumRef:
    people = tuple(people)
    c = Sum([If(people_m[person], 1, 0) for person in people])
    if verdict == INNOCENT:
        c = len(people) - c
    return c


@dataclass(frozen=True)
class Count(Constraint):
    typ: Verdict
    amount: Amount

    def z3(
        self, people: Iterable[Person], people_m: Mapping[Person, BoolRef]
    ) -> BoolRef:
        return self.amount.cmp(count(people, people_m, self.typ))


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
                            z3.Not(
                                And(
                                    people_m[people[i]],
                                    z3.Not(people_m[people[j]]),
                                    people_m[people[k]],
                                )
                            )
                        )
                    else:
                        constraints.append(
                            z3.Not(
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
class OnlyXPeople(Clue):
    amount: Amount
    region: Region
    constraint: PersonConstraint

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return self.amount.pb([
            (
                self.constraint.z3(field, people, person),
                1,
            )
            for person in self.region.people(field)
        ])


@dataclass(frozen=True)
class ForEvery(Clue):
    region: Region
    constraint: PersonConstraint

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return And(
            self.constraint.z3(field, people, person)
            for person in self.region.people(field)
        )


class PersonConstraint(ABC):
    @abstractmethod
    def z3(
        self, field: Field, people: Mapping[Person, BoolRef], person: Person
    ) -> BoolRef: ...


type PersonalRegion = Callable[[Person], Region]


@dataclass(frozen=True)
class SimplePersonConstraint(PersonConstraint):
    personal_region: PersonalRegion
    constraint: Constraint

    def z3(
        self, field: Field, people: Mapping[Person, BoolRef], person: Person
    ) -> BoolRef:
        return RegionClue(self.personal_region(person), self.constraint).z3(
            field, people
        )


@dataclass(frozen=True)
class ConditionalPersonConstraint(PersonConstraint):
    personal_region: PersonalRegion
    constraint: Constraint
    condition: Verdict

    def z3(
        self, field: Field, people: Mapping[Person, BoolRef], person: Person
    ) -> BoolRef:
        return And(
            people[person] == self.condition.value,
            RegionClue(self.personal_region(person), self.constraint).z3(field, people),
        )


@dataclass(frozen=True, init=False)
class OnlyX(Clue):
    clues: frozenset[Clue]
    amount: Amount

    def __init__(self, clues: Iterable[Clue], amount: Amount) -> None:
        object.__setattr__(self, "clues", frozenset(clues))
        object.__setattr__(self, "amount", amount)

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return self.amount.pb([(clue.z3(field, people), 1) for clue in self.clues])


@dataclass(frozen=True)
class Equal(Clue):
    a: Region
    a_verdict: Verdict
    b: Region
    b_verdict: Verdict

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return count(self.a.people(field), people, self.a_verdict) == count(
            self.b.people(field), people, self.b_verdict
        )


@dataclass(frozen=True)
class More(Clue):
    a: Region
    a_verdict: Verdict
    b: Region
    b_verdict: Verdict

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return count(self.a.people(field), people, self.a_verdict) > count(
            self.b.people(field), people, self.b_verdict
        )


@dataclass(frozen=True)
class ForEveryProfession(Clue):
    constraint: Constraint

    def z3(self, field: Field, people: Mapping[Person, BoolRef]) -> BoolRef:
        return And(
            RegionClue(ProfessionRegion(profession), self.constraint).z3(field, people)
            for profession in set(field.professions.values())
        )
