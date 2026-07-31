from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


def splitlist[T](lst: Sequence[T], v: T) -> tuple[Sequence[T], Sequence[T]]:
    i = lst.index(v)
    return lst[:i], lst[i + 1 :]
