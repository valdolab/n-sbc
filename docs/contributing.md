# Contributing

See [CONTRIBUTING.md](https://github.com/valdolab/n-sbc/blob/main/CONTRIBUTING.md) for full guidelines.

## Quick reference

```bash
# Clone and install
git clone https://github.com/valdolab/n-sbc.git
cd n-sbc
pip install -e .[dev]
pre-commit install

# Run tests
python -m pytest tests/ -v

# Lint and format
python -m ruff check .
python -m ruff format .
```

Open pull requests against the `dev` branch.
