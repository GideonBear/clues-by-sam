# ruff: file-ignore[assert]  type checker

from __future__ import annotations

import contextlib

from playwright.sync_api import (
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from clues_by_sam.clues import CRIMINAL, INNOCENT, Clue, Known
from clues_by_sam.game import Field, Person
from clues_by_sam.mover import Mover


def run(url: str) -> None:
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()
        page.goto(url)

        game, people, knowns, clues = load_game(page)
        play_game(page, game, people, knowns, clues)

        input("Press enter to continue...")

        browser.close()


def load_game(
    page: Page,
) -> tuple[Field, dict[Person, Locator], list[Known], list[Clue]]:
    # When on the homepage, there is a start button
    start_button = page.locator(".btn.start")
    with contextlib.suppress(PlaywrightTimeoutError):
        start_button.click(timeout=1000)

    grid = page.locator("#grid").first
    cards = grid.locator("> *")
    people = {}
    knowns = []
    clues = []
    for i in range(cards.count()):
        card = cards.nth(i).locator(".card").first
        person, card_knowns, card_clues = process_card(card)
        knowns.extend(card_knowns)
        clues.extend(card_clues)
        people[person] = card

    field = Field.from_list(list(people.keys()))

    return field, people, knowns, clues


def play_game(
    page: Page,
    field: Field,
    people: dict[Person, Locator],
    knowns: list[Known],
    clues: list[Clue],
) -> None:
    # TODO: implement
    mover = Mover(field, knowns)
    mover.add_clues(clues)
    while True:
        move = mover.get_move()
        if move is None:
            break

        card = people[move.person]
        card.click()

        if move.verdict == INNOCENT:
            page.locator(".btn-innocent").click()
        else:
            page.locator(".btn-criminal").click()

        card_person, card_knowns, card_clues = process_card(card)
        if card_person != move.person:
            raise AssertionError
        if card_knowns != [move]:
            raise AssertionError
        mover.add_clues(card_clues)


def process_card(card: Locator) -> tuple[Person, list[Known], list[Clue]]:
    classes_s = card.get_attribute("class")
    assert classes_s is not None
    classes = classes_s.split()
    name = card.locator("h3.name").text_content()
    assert name is not None
    name = name.capitalize()
    profession = card.locator("p.profession").text_content()
    # TODO: profession
    assert profession is not None

    person = Person(name)

    knowns = []
    if "innocent" in classes:
        knowns.append(Known(person, INNOCENT))
    elif "criminal" in classes:
        knowns.append(Known(person, CRIMINAL))

    clues = []
    if "has-hint" in classes:
        hint = card.locator(".hint").text_content()
        assert hint is not None
        clues.append(Clue.parse(hint))

    return person, knowns, clues
