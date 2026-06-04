# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.3] - 2026-06-04

### Added
- Zenodo archival for software releases (DOI minted automatically per version)
- This `CHANGELOG.md`

## [1.2.2] - 2026-06-04

### Added
- JOSS submission paper (`paper.md`, `paper.bib`)
- `CONTRIBUTING.md` with development guidelines
- MkDocs documentation site (deployed to GitHub Pages via `docs.yml`)
- Draft PDF GitHub Action (`draft-pdf.yml`) for paper compilation
- ORCID and software-level author info in `CITATION.cff`

### Fixed
- `pyproject.toml` version now matches `__version__.py`

## [1.2.1] - 2026-04-XX

### Added
- Explainability features: `predict_explain()` returning a `ZMatrix` with similarity matrix, class scores, and per-feature importances
- Plotting utilities: `plot_z_scores`, `plot_feature_importances`, `plot_similarity_heatmap`, `plot_chunk_similarity`
- Example notebooks in `examples/`

## [1.0.0] - 2025-XX-XX

### Added
- Initial release
- `NSBCClassifier` with scikit-learn compatible API
- Gray-coded binary encoding and Hamming similarity computation
- Validation tests against original MATLAB implementation
