from __future__ import annotations

import argparse
from argparse import ArgumentParser

from clues_by_sam.interaction import run


class Args(argparse.Namespace):
    url: str


def parse_args() -> Args:
    parser = ArgumentParser()

    parser.add_argument("url", type=str, nargs="?", default="https://cluesbysam.com")
    parser.add_argument("--no-wait", action="store_true")

    return parser.parse_args(namespace=Args())


def main() -> int:
    args = parse_args()
    run(args.url, wait=not args.no_wait)
    return 0
