# Himmapan Lab — Thotsakan Statistics
## Architectural and Mathematical Principles

This document defines the non-negotiable design and mathematical rules of the project.

Any refactor, feature addition, or optimization must comply with these principles.

---

# 1. Foundational Philosophy

> Architecture follows the math, not the other way around.

The statistical model and estimator logic define the structure of the system.
UI convenience must never dictate mathematical behavior.

---

# 2. Layered Architecture (Strict Separation)

The project follows a three-layer architecture:

    core (stats)  ←  controllers  ←  ui

## 2.1 Core / Stats Layer

Responsibilities:
- All statistical formulas
- All estimators
- All probability distributions
- All analytic CI/PI formulas
- All bootstrap logic
- Deterministic computational routines

Rules:
- No UI imports.
- No formatting.
- No rounding for display.
- No Gradio components.
- No user interface assumptions.
- No silent estimator decisions.

Core functions must be:
- Pure (no side effects)
- Explicit in parameters
- Deterministic when seed is provided

---

## 2.2 Controllers Layer

Responsibilities:
- Parse UI inputs
- Validate arguments
- Convert strings to typed parameters
- Choose estimator strategy
- Choose analytic vs bootstrap
- Handle edge cases
- Apply rounding for presentation
- Return structured outputs to UI

Rules:
- No statistical formulas.
- No probability derivations.
- No estimator definitions.
- No direct computation of statistics beyond trivial preprocessing.

Controllers orchestrate; they do not compute mathematics.

---

## 2.3 UI Layer (Gradio Tabs)

Responsibilities:
- Layout
- Widgets
- Event wiring
- Visibility logic
- Passing user selections to controllers

Rules:
- No statistical formulas.
- No estimator logic.
- No silent assumptions.
- No hardcoded statistical constants.

UI must be a thin shell.

---

# 3. Estimator Governance

Estimator choice is always user-controlled.

The system must never silently fix:

- Variance ddof
- Biased vs unbiased estimators
- Population vs sample sigma
- Bootstrap method
- Percentile vs other CI strategy

If a function needs an estimator:
It must accept it explicitly as a parameter.

Example (Correct):

    def ci_mean(data, variance_method="unbiased", ...):

Example (Incorrect):

    sigma_hat = np.std(data, ddof=1)  # silent assumption

All estimator decisions must originate in the controller layer.

---

# 4. Confidence and Prediction Intervals

Rules:

- If σ unknown → use t distribution.
- If σ provided → use z distribution.
- Bootstrap method must be explicit.
- CI method (analytic / percentile / BCa) must be explicit.
- All assumptions must be encoded in parameters.

No hidden statistical behavior.

---

# 5. Determinism & Reproducibility

- All random procedures must accept a seed parameter.
- Bootstrap must allow deterministic runs.
- Simulations must be reproducible.

---

# 6. Testing Requirements

Minimum invariants:

- PMFs sum to 1 (within tolerance).
- CDF monotonicity.
- CI endpoints ordered.
- Bootstrap reproducible with fixed seed.
- Estimator toggles change numerical results appropriately.

Tests encode mathematical contracts.

---

# 7. Documentation Discipline

Any change in statistical behavior requires:

1. Update to documentation.
2. Update to tests.
3. Clear commit message describing the change.

---

# 8. Code Style Rules

- Explicit parameter names.
- No magic numbers.
- No duplicated formulas.
- Prefer small pure functions.
- Mathematical clarity > cleverness.

---

# 9. Versioning Policy

- Patch: bug fixes, no behavior change.
- Minor: feature additions, no breaking math changes.
- Major: mathematical behavior changes.

---

# 10. Non-Negotiables

The following are violations:

- UI performing statistics.
- Controllers computing formulas.
- Core making UI decisions.
- Silent estimator assumptions.
- Undocumented statistical changes.

If in doubt: preserve mathematical transparency.

---

End of principles.