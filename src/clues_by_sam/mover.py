from __future__ import annotations

from typing import TYPE_CHECKING

from z3 import Bool, Solver, sat, unsat  # type: ignore[import-untyped]

from clues_by_sam.clues import Clue, Known, Verdict


if TYPE_CHECKING:
    from collections.abc import Iterable

    from clues_by_sam.game import Field


class Mover:
    def __init__(self, field: Field, knowns: list[Known]) -> None:
        self.field = field
        self.solver = Solver()
        self.people = {person: Bool(person.name) for person in field.all()}
        self.unknowns = set(field.all())
        for known in knowns:
            self.unknowns.remove(known.person)
            self.solver.add(self.people[known.person] == known.verdict.value)

    def add_clues(self, clues: Iterable[Clue]) -> None:
        for clue in clues:
            self.solver.add(clue.z3(self.field, self.people))

        if self.solver.check() != sat:
            msg = "Unsatisfiable"
            raise ValueError(msg)

    def get_move(self) -> Known | None:
        for unknown in self.unknowns:
            for verdict in [False, True]:
                self.solver.push()
                self.solver.add(self.people[unknown] == verdict)
                if self.solver.check() == unsat:
                    self.solver.pop()
                    self.solver.add(self.people[unknown] == (not verdict))
                    return Known(unknown, Verdict(not verdict))
                self.solver.pop()

        return None
