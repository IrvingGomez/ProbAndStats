# Himmapan Lab  
## Thotsakan Statistics

Thotsakan Statistics is an interactive statistical software developed under **Himmapan Lab**, an initiative focused on building advanced engineering mathematics tools created **by students, for students**.

This project is designed for teaching, learning, and expanding statistical methods through a modular software architecture that allows undergraduate students to contribute new features as part of academic projects.

---

# Vision

## Short-Term
- Support statistics teaching in engineering courses.
- Provide a structured codebase that students can expand.
- Serve as a laboratory platform for final-year undergraduate projects.

## Near-Term
- Establish an ecosystem of educational mathematical software:
  - Statistics
  - Differential Equations
  - Future advanced engineering mathematics tools

---

# Architecture Overview

This project follows a strict layered architecture:

```
UI → Controllers → Core
↓
State
```

## Layers

### `core/`
Mathematical and statistical computation layer.
- No UI logic
- No formatting
- Pure statistical algorithms

### `controllers/`
Boundary and orchestration layer.
- Validates user input
- Selects appropriate statistical procedures
- Formats outputs for the UI
- Never implements formulas

### `ui/`
Graphical interface (Gradio).
- Tabs and layout
- User controls
- Calls controller functions

### `state/`
Shared application state.
- Loaded dataset
- Filtered dataset
- Metadata (numeric/categorical columns)
- User overrides

---

# Repository Structure

```
app.py → Application entry point
core/ → Statistical engine
controllers/ → Orchestration logic
ui/ → Interface layer
state/ → Shared application state
datasets/ → Practice datasets
docs/
developers/ → Architecture and extension guides
theory/ → Statistical lecture notes
```

---

# Datasets

The `datasets/` folder contains:

- `practice/` → datasets available for students to explore in the UI
- `internal/` → testing or internal-use datasets

These datasets are used for:
- Descriptive statistics
- Graphical analysis
- Hypothesis testing
- Regression
- Inference methods

---

# Theory Notes

Statistical theory supporting this software is located in:

> docs/theory/ProbAndStatistics.pdf

These notes connect theoretical foundations with implemented methods.

---

# Running the Application

1. Install dependencies:

> pip install -r requirements.txt

2. Run:

> python app.py

---

# For Students: How to Contribute

Students are encouraged to extend the software by:

- Adding new statistical procedures
- Improving estimators
- Implementing new graphical tools
- Expanding inference methods
- Improving documentation

Before contributing, read:

> docs/developers/architecture.md
> docs/developers/adding_new_feature.md

These documents describe:

- Layer responsibilities
- Estimator selection contracts
- Feature addition workflow
- Common architectural mistakes to avoid

---

# Himmapan Lab

Himmapan Lab is an academic initiative focused on building advanced mathematical software ecosystems for engineering education.

Future projects will follow the same architectural principles used here, enabling:

- Consistent student onboarding
- Reusable design patterns
- Cross-project collaboration
- Scalable expansion into new domains

---

# License

(To be added)
