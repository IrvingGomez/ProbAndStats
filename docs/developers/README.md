# Developer Documentation  
## Thotsakan Statistics – Himmapan Lab

This section contains the technical documentation for contributors to **Thotsakan Statistics**.

It is intended for:

- Undergraduate students extending the project
- Teaching assistants
- Future maintainers
- Himmapan Lab collaborators

This documentation defines the architectural rules and extension workflow that must be followed to preserve long-term stability.

---

# Philosophy

Thotsakan Statistics is designed to be:

- Statistically correct  
- Architecturally disciplined  
- Pedagogically clear  
- Safe for collaborative student development  

Clarity and correctness are prioritized over shortcuts and quick fixes.

Architectural rules are strictly enforced.

---

# Reading Order

If you are new to the project, read the following in order

## 1. Constitution (must read)

[Constitution (Structural & Mathematical Rules)](../project/constitution.md)

Defines the non-negotiable structural and mathematical rules of the project.

---

## 2. Architecture Overview

[Architecture Guide](../project/architecture.md)

Explains the layered architecture, dependency rules, and responsibilities of each layer.

---

## 3. Governance & Lifecycle

Describes versioning policy, tests, releases, and repository governance.

[Governance & Lifecycle Policy](../project/governance.md)

---

## 4. Feature Workflow

[Feature Workflow](docs/developers/feature_workflow.md)

Explains how to add new features correctly (Core → Controller → UI), and where new code belongs.

---

# Core Architectural Rule

Dependencies always flow downward:

    UI  →  Controllers  →  Core

- Core must never depend on Controllers, UI, or State.
- Controllers may depend on Core.
- UI may depend on Controllers and State.
- State contains data only.

Violating this rule is considered a design error.

---

# Contribution Expectations

Before submitting code:

- Ensure Core contains no UI logic.
- Ensure Controllers contain no statistical formulas.
- Ensure UI does not call Core directly.
- Ensure no architectural rule violations exist.
- Test features incrementally (Core → Controller → UI).

If you are unsure where code belongs, ask before implementing.

---

# Testing

A minimal test suite exists in:

`tests/`

To run tests locally:

1. Activate your virtual environment.
2. Install development dependencies:

```
pip install -r requirements-dev.txt
```

3. Run:

```
pytest -q
```

Tests must pass before submitting changes.

The goal of testing in this project is to:

- Catch structural regressions
- Prevent broken imports
- Protect statistical correctness
- Encourage disciplined development habits

---

# Himmapan Lab Ecosystem

This project is part of the broader **Himmapan Lab** initiative.

Future projects (e.g., Differential Equations and other advanced engineering mathematics tools) will follow the same architectural template.

Maintaining architectural discipline here ensures:

- Reusability across projects
- Consistent onboarding
- Sustainable long-term growth

---

# Final Reminder

This is an educational and collaborative project.

The goal is not only to build software, but to build:

- Good engineering habits  
- Clean architectural thinking  
- Reproducible scientific computation  

When in doubt, favor simplicity and clarity.
