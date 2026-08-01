from __future__ import annotations

from collections import UserList
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self


if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Person:
    name: str

    def __post_init__(self) -> None:
        if not self.name[0].isupper() and self.name[1:].islower():
            msg = f"Invalid person name: '{self.name}'"
            raise ValueError(msg)

    def __str__(self) -> str:
        return self.name


ROWS = 5
COLUMNS = 4


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

    @classmethod
    def from_list(cls, people: list[Person]) -> Self:
        if len(people) != ROWS * COLUMNS:
            msg = "Field is not 5x4"
            raise ValueError(msg)
        return cls(people[i * COLUMNS : (i + 1) * COLUMNS] for i in range(ROWS))
