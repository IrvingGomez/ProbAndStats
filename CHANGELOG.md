# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog.
This project follows Semantic Versioning (MAJOR.MINOR.PATCH).

---

## [Unreleased]

### Planned (v0.2.x direction)

- Build out Probability tab (distributions, simulation overlays, intuition-first visuals).
- Add statistical correctness tests (CI coverage checks, estimator selection contracts).
- Improve input validation and error messaging across controllers.

---

## [0.1.2] – 2026-02-22

### Changed
- Reorganized documentation structure
- Renamed docs/architecture/ to docs/project/
- Recreated architecture.md as structural overview
- Consolidated architectural authority in constitution.md
- Fixed internal documentation links

## [0.1.1] - 2026-02-18

### Added
- LICENSE (open-source licensing formalized).
- CITATION.cff (academic citation metadata).

### Notes
- Patch release.
- No functional or API changes.

---

## [0.1.0] - 2026-02-18

### Added
- First structured public release under Himmapan Lab.
- Layered architecture (Core / Controllers / UI / State) with strict dependency rules.
- Developer documentation:
  - `docs/developers/architecture.md`
  - `docs/developers/adding_new_feature.md`
  - `docs/developers/coding_rules.md`
- Practice datasets under `datasets/practice/`.
- “Home” tab explaining Himmapan Lab and the product ecosystem.
- Minimal test suite (`tests/`) as foundation for CI and regression protection.
- Core statistical modules:
  - Descriptive statistics
  - Inference utilities
  - Hypothesis testing
  - Linear regression
- Modular Gradio UI with tabs:
  - Data
  - Probability (scaffolding)
  - Estimation
  - Hypothesis Testing
  - Linear Regression

### Known Limitations
- Probability tab is scaffolding and under active development.
- Test coverage is minimal; statistical correctness tests planned.

