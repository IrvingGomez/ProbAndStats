# Thotsakan Statistics
# Architectural & Mathematical Constitution
## Himmapan Lab

This document defines the permanent structural and mathematical rules of the project.

These rules are non-negotiable.

Any feature addition, modification, or optimization must comply with this constitution.

---

# 1. Foundational Principle

> Architecture follows the mathematics.

Statistical definitions, estimator behavior, and probabilistic assumptions determine system structure.

UI convenience, coding shortcuts, or implementation ease must never override mathematical correctness.

---

# 2. Layered Architecture (Strict Separation)

The project follows a strict layered structure:

    UI  →  Controllers  →  Core

Dependencies flow downward only.

## 2.1 Core Layer (`core/`)

Purpose:
- Implement all statistical, mathematical, and analytical logic.

Responsibilities:
- Estimators
- Probability distributions
- Confidence and prediction intervals
- Hypothesis tests
- Regression models
- Bootstrap procedures
- Plot construction (matplotlib)

Rules:
- No UI imports
- No Gradio usage
- No formatting for display
- No rounding for presentation
- No access to AppState
- No silent estimator selection
- Deterministic when seed provided
- Raise explicit Python exceptions on invalid input

Core answers:

> “What is the mathematically correct result?”

---

## 2.2 Controller Layer (`controllers/`)

Purpose:
- Translate user intent into explicit statistical parameters.

Responsibilities:
- Parse UI inputs
- Validate arguments
- Select estimators
- Select analytic vs bootstrap strategy
- Select distributional assumptions
- Apply rounding and formatting
- Return structured outputs for UI rendering

Rules:
- No statistical formulas
- No probability derivations
- No estimator definitions
- No UI layout logic
- No direct imports from `ui.tabs`

Controllers orchestrate. They do not compute mathematics.

Controllers answer:

> “Given these user choices, what must be computed?”

---

## 2.3 UI Layer (`ui/`)

Purpose:
- Define interaction and presentation.

Responsibilities:
- Layout
- Widgets
- Event wiring
- Visibility logic
- Passing parameters to controllers
- Rendering outputs

Rules:
- No statistical formulas
- No estimator logic
- No distribution logic
- No silent mathematical constants
- No direct imports from `core`

UI must remain a thin interaction shell.

UI answers:

> “How does the user interact with the system?”

---

## 2.4 State Layer (`state/`)

Purpose:
- Store shared application data.

Responsibilities:
- Dataset storage
- Filtered dataset
- Metadata

Rules:
- No statistical logic
- No UI logic
- No controller logic

State is data-only.

---

# 3. Estimator Governance

Estimator choice always originates from user intent.

The system must never silently choose:

- ddof
- Biased vs unbiased variance
- Robust vs classical estimators
- Bootstrap method
- CI construction strategy

Then:

- The UI exposes those options explicitly.
- The Controller interprets those options.
- The Core receives explicit parameters and does not override them.

If a function requires an estimator:

It must receive it explicitly.

Incorrect pattern:

    sigma_hat = np.std(data, ddof=1)

Correct pattern:

    def ci_mean(data, variance_method, ...)

Estimator authority flows:

UI → Controller → Core

Never the reverse.

### Exception — Likelihood-Based Inference

For likelihood-based confidence regions or intervals:

- The estimator is the **Maximum Likelihood Estimator (MLE)** by definition.
- The user does not override the estimator in this context.
- The estimator is determined by the statistical model, not by user preference.

In such cases, estimator choice is not a configurable parameter but a structural property of the method.

Outside of likelihood-based methods, no silent estimator selection is permitted.

---

# 4. Interval Construction Rules

- When the estimator for μ is the **sample mean** and the estimator for σ is the **sample standard deviation with 1 degree of freedom (ddof = 1)**, interval construction uses the **t-distribution** (finite-sample Student–t theory).

- For **all other estimator combinations** (robust location estimators, alternative scale estimators, bias-corrected dispersion measures, transformed parameters, etc.), interval construction is based on **asymptotic normal distributions** (or another explicitly specified large-sample distribution).

- Bootstrap-based intervals are used **only when explicitly requested**, and the bootstrap method (e.g. percentile, BCa, etc.) must be encoded as a parameter.

- The CI/PI strategy (**analytic t-based, asymptotic normal, bootstrap, etc.**) must always be explicit.

- All assumptions (estimator choice, distributional approximation, bootstrap method) must be represented in parameters; none may be silently inferred.

No hidden statistical behavior.

---

# 5. Determinism & Reproducibility

- All random procedures accept a seed.
- Bootstrap must be reproducible.
- Simulation must be reproducible.

Reproducibility is structural, not optional.

---

# 6. Mathematical Integrity Contracts

At minimum, the system must guarantee:

- PMFs sum to 1 (within tolerance)
- CDF monotonicity
- Ordered CI/PI endpoints
- Bootstrap reproducibility under fixed seed
- Estimator selection changes numerical output appropriately

Tests encode these contracts.

---

# 7. Architectural Anti-Patterns (Forbidden)

The following are violations:

- UI performing statistical computation
- Controllers computing formulas
- Core importing UI modules
- Silent estimator overrides
- Hard-coded α
- Undocumented statistical changes
- Hidden global state

Violations must be corrected immediately.

---

# 8. Versioning as Mathematical Contract

The version number communicates behavioral guarantees.

- PATCH → bug fixes, no statistical behavior change
- MINOR → new features, no breaking math changes
- MAJOR → changes to statistical behavior or public API

Statistical behavior changes require MAJOR version increment.

---

End of Constitution.