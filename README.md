# clues-by-sam

Solver for [Clues By Sam](https://cluesbysam.com/)
* Reads grid and clicks tiles using Playwright
* Parses clues using a [lark](https://github.com/lark-parser/lark) parser
* Converts the clues into Z3 expressions and uses Z3 to generate valid moves

## Installation

```bash
# Using uv (will automatically install Python if necessary):
uv tool install git+https://github.com/GideonBear/clues-by-sam
# Using pipx:
pipx install git+https://github.com/GideonBear/clues-by-sam
# Using pip:
pip install git+https://github.com/GideonBear/clues-by-sam
```

## Python version support

This project will only ever support the latest released minor version of python, but will most likely work on older
versions as well. Change `requires-python` manually in `pyproject.toml` if necessary.
