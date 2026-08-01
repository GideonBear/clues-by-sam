from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


def splitlist[T](lst: Sequence[T], v: T) -> tuple[Sequence[T], Sequence[T]]:
    i = lst.index(v)
    return lst[:i], lst[i + 1 :]


def splitlist_by_subseq[T](
    lst: Sequence[T], v: Sequence[T]
) -> tuple[Sequence[T], Sequence[T]]:
    i = lst.index(v[0])
    for j, subv in enumerate(v[1:]):
        if lst[i + j + 1] != subv:
            msg = (
                f"First occurrence of '{v[0]}' did not match full subsequence {v}. "
                f"Searched in: {lst}"
            )
            raise ValueError(msg)
    return lst[:i], lst[i + len(v) :]
