# Coding Rules  
## Thotsakan Statistics – Himmapan Lab

This document defines coding conventions and non-negotiable rules for contributors.

These rules exist to:

- Preserve architectural discipline
- Prevent technical debt
- Maintain clarity for student collaboration
- Ensure long-term scalability

Follow these rules strictly.

---

# 1. Architectural Discipline

## 1.1 Layer Boundaries

The project follows a strict layered structure:

```
UI → Controllers → Core
```

### Core
- Contains statistical logic only.
- Must not import from `controllers`, `ui`, or `state`.
- Must not format output for presentation.
- Must not access application state.

### Controllers
- May import from `core`.
- Must not implement statistical formulas.
- Must not import UI modules.
- Responsible for parsing, validation, and formatting.

### UI
- Must call controllers only.
- Must not import from `core`.
- Must not perform statistical computation.

### State
- Contains data only.
- No statistical logic.
- No UI logic.
- No controller logic.

Violations of these rules are considered design errors.

---

# 2. Naming Conventions

## 2.1 File Names

- Use lowercase with underscores.
- Be descriptive.
- Examples:
  - `ci_mean.py`
  - `inference_controller.py`
  - `graphical_analysis.py`

Avoid vague names like:
- `utils2.py`
- `misc.py`
- `helpers.py`

---

## 2.2 Function Names

- Use clear, explicit names.
- Prefer `ci_mean_bootstrap` over `compute_ci`.
- Avoid abbreviations unless standard (e.g., `ci`, `pi`, `df`).

---

## 2.3 Variables

- Use descriptive variable names.
- Avoid single-letter variables except in mathematical contexts.
- Avoid shadowing built-ins (`list`, `dict`, `sum`, etc.).

---

# 3. Estimator Selection Contract

This is critical for this project.

## Rule:

**The Core must not silently choose estimators if the user has options.**

If the UI allows selection of:
- Mean estimator
- Deviation estimator
- Bootstrap vs analytic
- Robust vs classical

Then:

- The Controller decides which estimator to use.
- The Core receives that decision explicitly.
- The Core does not override user choices.

Never hard-code classical defaults when configurability exists.

---

# 4. Formatting and Rounding

- Core must return raw numerical results.
- Controllers apply rounding.
- UI displays formatted values.

Do NOT round inside statistical functions.

Bad:
```python
return round(ci_lower, 2), round(ci_upper, 2)
```

Correct:

```python
return ci_lower, ci_upper
```

Formatting belongs in controllers.

---
# 5. Error Handling

In Core:
- Raise meaningful Python exceptions.
- Do not suppress errors silently.

In Controllers:
- Catch predictable errors.
- Convert them into user-friendly messages if needed.

# 6. Imports

Import order:
1. Standard library
2. Third-party libraries
3. Local project modules

Example:
```python
import numpy as np
import pandas as pd

from core.estimation.inference import ci_mean
```

Avoid circular imports.

---

# 7. Avoid Quick Fixes

Do NOT:
- Patch logic inside UI because “it works”
- Add conditional hacks to bypass architecture
- Duplicate statistical logic in multiple places

If something feels like a shortcut, it probably violates architecture.

---

# 8. Testing Mindset

When adding features:
- Test the Core function directly.
- Test the Controller separately.
- Then test the UI integration.
Debug from bottom to top.

---

# 9. Code Clarity

This is an educational project.

Prefer:
- Readability over cleverness
- Explicitness over abstraction
- Simplicity over premature optimization

Students should be able to understand the code.

---

# 10. When Unsure

If unsure about:
- Layer placement
- Estimator logic
- Architectural implications
Ask before implementing.

Architectural integrity is more important than speed of development.