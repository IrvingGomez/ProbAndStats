# Himmapan Lab  
# Thotsakan Statistics

**Current stable version:** v0.1.2  
See full release history in [CHANGELOG.md](CHANGELOG.md).

---

## Overview

**Thotsakan Statistics** is an interactive statistical application developed under **Himmapan Lab**, an academic initiative focused on building modular mathematical software ecosystems for engineering education.

The project serves two purposes:

1. **Pedagogical Platform** – Supporting statistics teaching in engineering programs.
2. **Structured Development Laboratory** – Providing a disciplined, extensible codebase where undergraduate students can contribute meaningful statistical functionality.

The system emphasizes:

- Architectural clarity  
- Strict separation of concerns  
- Extensibility  
- Long-term maintainability  

---

## Vision

### Short-Term
- Support undergraduate statistics courses.
- Provide a stable, well-structured contribution platform for students.
- Enable final-year project extensions.

### Medium-Term
- Expand statistical functionality (e.g., completion of the Probability module).
- Improve statistical validation testing and coverage.
- Strengthen input validation and controller robustness.

### Long-Term
- Evolve into a serious, extensible statistical platform within a broader ecosystem of engineering mathematics tools.

---

## Core Architectural Principles

The project follows a strict layered architecture:

```
app.py → Application entry point
core/ → Pure statistical computation layer
controllers/ → Validation, orchestration, estimator selection
ui/ → Gradio-based presentation layer
state/ → Shared application state
datasets/ → Teaching and practice datasets
docs/ → User, developer, and theory documentation
```

**Dependency rule:**  
UI → Controllers → Core  
(No reverse dependencies allowed.)

For full details, see:

- [Architecture Overview](docs/project/architecture.md)
- [Constitution (Structural & Mathematical Rules)](docs/project/constitution.md)
- [Governance & Lifecycle Policy](docs/project/governance.md)
- [Developer Documentation](docs/developers/README.md)

---

## Installation & Environment Setup

It is strongly recommended to use a virtual environment.

### 1. Create a Virtual Environment (once)

#### Windows (PowerShell)

```
python -m venv .venv
..venv\Scripts\activate
```

#### Windows (Git Bash / WSL)

```
python -m venv .venv
source .venv/Scripts/activate
```

#### macOS / Linux

```
python -m venv .venv
source .venv/bin/activate
```

After activation, your terminal should display:

```
(.venv)
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Run the Application

```
python app.py
```

You only need to create and install once.  
You must activate the environment each time you open a new terminal.

---

## Datasets

Practice datasets are located in:

[Practice Datasets](datasets/practice/)

They are intended for:

- Descriptive statistics  
- Estimation and inference  
- Hypothesis testing  
- Linear regression exercises  

---

## Theory Notes

Supporting statistical theory material:

[Probability and Statistics](docs/theory/prob_and_stats.pdf)

---

## Contributing

Students and contributors are encouraged to extend the software responsibly.

Possible contribution areas:

- New statistical estimators  
- Alternative inference methods  
- Visualization improvements  
- Input validation enhancements  
- Additional statistical modules  
- Test coverage expansion  

Before contributing, read:

- [Architecture Guide](docs/project/architecture.md)  
- [Feature Workflow](docs/developers/feature_workflow.md)  
- [Coding Rules](docs/developers/coding_rules.md) 

Architectural discipline is mandatory to preserve long-term stability.

---

## Running Tests

For contributors:

```
pip install -r requirements-dev.txt
pytest -q
```

All new features should include at least minimal test coverage.

---

## Releases

This project follows **Semantic Versioning (MAJOR.MINOR.PATCH)**.

- **v0.1.0** – First structured public release  
- **v0.1.1** – Metadata patch (LICENSE and CITATION added)

For full details, see:

- [CHANGELOG.md](CHANGELOG.md)

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## About Himmapan Lab

Himmapan Lab is an academic initiative focused on building modular mathematical software ecosystems for engineering education.

Each project in the lab follows consistent architectural principles to enable:

- Clear student onboarding  
- Sustainable expansion  
- Cross-project collaboration  
- Long-term scalability  
