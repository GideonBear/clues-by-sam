# ruff: file-ignore[assert]  type checker

from __future__ import annotations

from playwright.sync_api import Page, sync_playwright

from clues_by_sam.clues import CRIMINAL, INNOCENT, Clue, Known
from clues_by_sam.game import Field, Game, Person


def run(url: str = "https://cluesbysam.com") -> None:
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()
        page.goto(url)
        game = load_game(page)
        play_game(page, game)

        browser.close()


def load_game(page: Page) -> Game:
    start_button = page.locator(".btn.start")
    start_button.click()

    grid = page.locator("#grid").first
    cards = grid.locator("> *")
    people = []
    knowns = []
    clues = []
    for i in range(cards.count()):
        card = cards.nth(i).locator(".card").first
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
        people.append(person)

        if "innocent" in classes:
            knowns.append(Known(person, INNOCENT))
        elif "criminal" in classes:
            knowns.append(Known(person, CRIMINAL))

        if "has-hint" in classes:
            hint = card.locator(".hint").text_content()
            assert hint is not None
            clues.append(Clue.parse(hint))

    game = Game(Field.from_list(people))
    for known in knowns:
        game.add_known(known)
    for clue in clues:
        game.add_clue(clue)

    return game


def play_game(page: Page, game: Game) -> None: ...  # TODO: implement
