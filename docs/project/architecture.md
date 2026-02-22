# Thotsakan Statistics
## System Architecture Overview
### Himmapan Lab

This document describes the structural organization of the Thotsakan Statistics project.

Its purpose is to provide a clear and practical map of how the system is built, how its parts interact, and how new features should be integrated.

This is a structural overview.

For non-negotiable structural and mathematical rules, see:

- [Constitution](constitution.md)
- [Governance & Lifecycle Policy](governance.md)

---

# 1. Architectural Style

Thotsakan Statistics follows a strict layered architecture:

    UI  →  Controllers  →  Core

Dependencies flow **downward only**:

- UI depends on Controllers
- Controllers depend on Core
- Core depends on nothing in higher layers
- State stores shared data but contains no logic

This separation enforces clarity, testability, and long-term maintainability.

---

# 2. Directory-to-Layer Mapping

The repository structure reflects the architectural layers.

## Core (`core/`)

Role:
- Implements all statistical and computational logic.

Typical contents:
- Estimators
- Inference routines
- Regression logic
- Probability-related computations
- Plot construction

Core contains pure computational logic and does not depend on UI or controllers.

---

## Controllers (`controllers/`)

Role:
- Act as the orchestration layer between UI and Core.

Typical responsibilities:
- Validate user inputs
- Translate UI selections into typed parameters
- Select computation paths
- Call Core functions
- Prepare results for UI rendering

Controllers define the internal API that the UI consumes.

---

## UI (`ui/`)

Role:
- Define layout and interaction logic.

Typical responsibilities:
- Build Gradio components
- Wire UI events to controller functions
- Manage visibility logic
- Render outputs

UI must not implement computational logic.

---

## State (`state/`)

Role:
- Store shared application data.

Typical contents:
- Loaded dataset
- Filtered dataset
- Column metadata

State contains data only, not business logic.

---

## Tests (`tests/`)

Role:
- Validate import integrity
- Enforce architectural boundaries
- Verify computational correctness (via Core)

Tests are part of structural stability.

---

## Datasets (`datasets/`)

Role:
- Provide teaching and experimentation datasets.

They are not part of the computational architecture.

---

## Docs (`docs/`)

Role:
- Separate documentation by audience:

  - `users/` → usage guides
  - `developers/` → extension workflow
  - `project/` → structural and governance authority
  - `theory/` → mathematical background material

---

# 3. Dependency Rules

The fundamental structural constraint is:

    UI  →  Controllers  →  Core

Allowed:

- UI imports controllers
- Controllers import core
- UI reads state
- Controllers read state

Not allowed:

- Core importing controllers or UI
- UI importing core directly
- State containing logic

This ensures clear responsibility boundaries.

---

# 4. Feature Flow Through the System

When a new statistical feature is added, it must follow this structural path:

1. **Core Implementation**
   - Implement computational logic inside `core/`.

2. **Controller Exposure**
   - Add a controller function that:
     - Validates input
     - Calls the core function
     - Formats results appropriately

3. **UI Integration**
   - Add UI controls
   - Connect them to the controller
   - Display returned results

This preserves structural clarity and prevents architectural drift.

---

# 5. Extension Example (Conceptual)

Adding a new confidence interval method:

- Core:
  - Implement interval computation.
- Controller:
  - Add parameter parsing and dispatch logic.
- UI:
  - Add dropdown option and connect to controller.

No layer should absorb responsibilities from another.

---

# 6. Relationship to Constitutional Rules

This document describes *how the system is organized*.

The following documents define binding rules:

- [Constitution](docs/architecture/constitution.md) → structural and mathematical authority
- [Governance & Lifecycle Policy](docs/architecture/governance.md)  → lifecycle, versioning, and release discipline

If a structural conflict arises, [Constitution](docs/architecture/constitution.md) prevails.

---

# 7. Design Goals of This Architecture

The layered design exists to:

- Make statistical logic independently testable
- Allow UI changes without breaking mathematics
- Allow new features without structural corruption
- Support undergraduate contributors safely
- Maintain long-term stability

Clarity of structure is prioritized over convenience.

---

End of Architecture Overview.