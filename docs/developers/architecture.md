# Architecture Overview
## Layered Statistical Application Architecture

---

## 1. Purpose of this document

This document describes the architectural design of the **Thotsakan Statistics** project.

Its goals are to:

- Make the structure of the codebase explicit and unambiguous
- Define strict responsibilities for each layer
- Prevent architectural drift as new features are added
- Serve as a reference for developers and student collaborators

This document is **authoritative**: when architectural questions arise, this document defines the intended design.

---

## 2. Architectural style

Thotsakan Statistics follows a **Layered Statistical Application Architecture**.

This is a custom, non-framework architecture. It is inspired by classical separation-of-concerns principles, but it is **not** an MVC framework, nor is it tied to Gradio, web patterns, or any specific frontend paradigm.

The application is divided into four main layers:

1. **Core (Statistical Logic)**
2. **Controllers (Orchestration & Validation)**
3. **UI (Presentation & Interaction)**
4. **State (Shared Application State)**

Each layer has strict responsibilities and explicit dependency rules.

---

## 3. Global dependency rule (most important)

**Dependencies always flow downward. Never upward.**

```
UI
↓
Controllers
↓
Core
```


- The Core layer must never depend on Controllers, UI, or State
- Controllers may depend on Core, but not on UI
- UI may depend on Controllers and State, but not on Core directly
- State contains data only and must not contain logic

Violating this rule is considered a **design error**, even if the code appears to work.

---

## 4. Layer responsibilities

---

### 4.1 Core layer (`core/`)

**Purpose**

Implement all statistical, mathematical, and analytical logic.

This layer defines *what is statistically correct*, independently of how results are presented or triggered.

**Key properties**

- Pure Python
- No UI code
- No Gradio imports
- No global state
- Deterministic and testable

**Typical responsibilities**

- Statistical estimators
- Confidence intervals and prediction intervals
- Hypothesis tests
- Regression models
- Construction of Matplotlib / Seaborn figures

**Directory structure**

```
core/
├── data_stats.py
├── hypothesis_tests.py
├── linear_regression.py
└── estimation/
    ├── descriptive.py
    ├── graphical_analysis.py
    └── inference/
```

**Rules**

- Do NOT import `gradio`
- Do NOT reference UI components
- Do NOT perform rounding for presentation
- Do NOT access application state
- Return raw numerical results, DataFrames, or figures
- Raise meaningful Python exceptions on invalid input

> The Core layer answers:  
> **“What is the correct statistical result?”**

---

### 4.2 Controller layer (`controllers/`)

**Purpose**

Act as the orchestration layer between the UI and the Core.

Controllers translate user intent into concrete statistical computations and adapt raw results into UI-ready outputs.

**Directory structure**

```
controllers/
├── data_controller.py
├── hypothesis_controller.py
├── linear_regression_controller.py
└── estimation/
    ├── descriptive_controller.py
    ├── graphical_controller.py
    └── inference_controller.py
```

**Typical responsibilities**

- Validate user input
- Convert UI values (strings, dropdowns) into typed parameters
- Select appropriate statistical methods or estimators
- Dispatch computation to Core functions
- Apply rounding and formatting for presentation
- Decide which outputs are returned to the UI

**Rules**

- Do NOT implement statistical formulas
- Do NOT build UI layouts
- Do NOT import from `ui.tabs`
- Import freely from `core`
- Return objects suitable for UI rendering (tables, figures, messages)

> The Controller layer answers:  
> **“Given these user choices, what should be computed and shown?”**

---

### 4.3 UI layer (`ui/`)

**Purpose**

Define the user interface and interaction flow.

This layer is responsible for *how the application looks and behaves*, not for what it computes.

**Directory structure**

```
ui/
├── layout.py
├── styles.py
├── assets.py
└── tabs/
    ├── data_tab.py
    ├── hypothesis_testing_tab.py
    ├── linear_regression_tab.py
    └── estimation/
```

**Typical responsibilities**

- Gradio layout and components
- Visibility toggles and UI logic
- Wiring UI events to controller functions
- Displaying tables and figures

**Rules**

- Do NOT implement statistical logic
- Do NOT import from `core`
- Do NOT apply statistical rounding rules
- Call controllers exclusively
- Manage user interaction and presentation flow

> The UI layer answers:  
> **“How does the user interact with the application?”**

---

### 4.4 State layer (`state/`)

**Purpose**

Store shared application state.

This layer exists to avoid recomputation and to centralize shared data.

**Directory structure**

```
state/
└── app_state.py
```

**Responsibilities**

- Store loaded datasets
- Store filtered datasets
- Store derived column metadata

**Rules**

- No statistical logic
- No UI logic
- No controller logic
- Simple data containers only

---

## 5. Extension guidelines

---

### Adding a new statistical method

1. Implement the method in `core/`
2. Expose it through a controller
3. Add UI controls in the appropriate tab

---

### Adding a new estimator

- Implement estimator logic in `core/estimation/inference/estimators.py`
- Update controller selection logic
- Update UI dropdowns

---

### Adding a new tab

- Core logic → `core/`
- Controller logic → `controllers/`
- UI layout → `ui/tabs/`

---

## 6. Explicit anti-patterns (do NOT do these)

- Importing Gradio inside `core/`
- Performing rounding inside `core/`
- Accessing `AppState` from `core/`
- UI tabs calling `core` directly
- Controllers returning UI layout components

These patterns break separation of concerns and will make the project fragile.

---

## 7. Design philosophy

This architecture is designed to:

- Prioritize statistical correctness
- Support teaching and student collaboration
- Scale without becoming fragile
- Make reasoning about correctness easier than reasoning about convenience

Correctness and clarity are valued **over brevity or cleverness**.

---

## 8. Guidance for contributors

If you are unsure where new code belongs:

> **Place it in the highest layer that does not violate dependency rules.**

If still unsure, ask before implementing.

---
