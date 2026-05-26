# Contributing to n-SBC

Thank you for your interest in contributing to n-SBC. This document provides guidelines for contributing to the project.

## Reporting bugs

If you find a bug, please open an issue on [GitHub Issues](https://github.com/valdolab/n-sbc/issues) with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your Python version and OS

## Suggesting features

Feature requests are welcome. Please open an issue describing:

- The use case your feature would address
- How you envision the API working
- Any relevant references or examples

## Development setup

1. Clone the repository:

```bash
git clone https://github.com/valdolab/n-sbc.git
cd n-sbc
```

2. Install in development mode:

```bash
pip install -e .[dev]
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

## Running tests

```bash
python -m pytest tests/ -v
```

To run a specific test:

```bash
python -m pytest tests/test_cryotherapy.py::test_single_prediction -v
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
python -m ruff check .
python -m ruff format .
```

Key conventions:
- Line length: 88 characters
- Python 3.10+ syntax
- Type hints encouraged for public APIs
- Variable naming: lowercase `x`, `x_train`, `y`

Pre-commit hooks enforce these rules automatically on each commit.

## Pull request process

1. Create a feature branch from `dev`:

```bash
git checkout -b feature/your-feature dev
```

2. Make your changes and ensure tests pass
3. Run the full pre-commit suite:

```bash
pre-commit run --all-files
```

4. Push and open a pull request against `dev`
5. Describe what your PR does and why

## Code of conduct

Be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.
