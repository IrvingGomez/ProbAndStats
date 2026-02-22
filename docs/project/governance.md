# Thotsakan Statistics
# Governance & Lifecycle Policy
## Himmapan Lab

This document defines repository governance, versioning, and release discipline.

---

# 1. Repository as Computational Artifact

The repository must be reproducible.

Cloning the project and installing dependencies must produce identical behavior across machines.

---

# 2. Dependency Discipline

- Runtime dependencies → requirements.txt
- Development dependencies → requirements-dev.txt
- No virtual environments committed
- No cache directories committed
- .gitignore must prevent environment leakage

---

# 3. Testing Requirements

Before merging changes:

- All tests must pass
- Import integrity must hold
- No architectural boundary violations
- No undocumented statistical changes

Testing levels:

1. Core mathematical tests
2. Controller contract tests
3. Exportability tests
4. Structural import tests

---

# 4. Pull Request Rules

Every PR must:

- Describe the change clearly
- Classify version impact (PATCH/MINOR/MAJOR)
- Update tests if statistical logic changes
- Update documentation if behavior changes

No undocumented statistical change is permitted.

---

# 5. Versioning Policy

Semantic Versioning: MAJOR.MINOR.PATCH

MAJOR:
- Changes statistical behavior
- Changes public controller return contracts
- Changes estimator definitions

MINOR:
- Adds new estimator
- Adds new method
- Adds new visualization
- Adds new tab

PATCH:
- Fixes bug without changing math
- Fixes formatting
- Fixes UI layout

---

# 6. Release Procedure

1. Run full test suite
2. Confirm no invariant violations
3. Update version in ui/version.py
4. Update CHANGELOG.md
5. Commit with semantic message
6. Create annotated Git tag
7. Publish GitHub release

---

# 7. Documentation Discipline

Any change in statistical logic requires:

- Documentation update
- Test update
- Version classification update

Mathematical transparency is mandatory.

---

End of Governance Policy.