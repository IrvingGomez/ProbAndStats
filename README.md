# Himmapan Lab  
## Thotsakan Statistics

Thotsakan Statistics is an interactive statistical application developed under **Himmapan Lab**, an initiative focused on building advanced engineering mathematics software created **by students, for students**.

This project is designed to support statistics teaching while serving as a structured platform for undergraduate students to extend and improve the software as part of academic projects.

---

# Vision

## Short-Term
- Support statistics courses in engineering programs.
- Provide a clean, extensible codebase for student contributors.
- Serve as a laboratory platform for final-year projects.

## Near-Term
- Build a broader ecosystem of educational mathematical tools:
  - Statistics
  - Differential Equations
  - Additional advanced engineering subjects

## Long-Term
- Evolve into a serious, extensible statistical platform.

---

# Environment Setup (Recommended)

It is strongly recommended to use a **virtual environment** to avoid package conflicts.

## 1. Create a virtual environment (once)

From the project root:

### Windows (Git Bash / WSL)

```
python -m venv .venv
source .venv/Scripts/activate
```

### Windows (PowerShell)

```
python -m venv .venv
..venv\Scripts\activate
```

### macOS / Linux

```
python -m venv .venv
source .venv/bin/activate
```

After activation, your terminal should display:

```
(.venv)
```

## 2. Install dependencies (once)

```
pip install -r requirements.txt
```

## 3. Run the application

```
python app.py
```

You only need to **create and install once**.  
You must **activate the environment each time you open a new terminal**.

---

# Project Structure (High-Level)

```
app.py → Application entry point
core/ → Statistical engine (pure computation)
controllers/ → Orchestration and validation layer
ui/ → User interface (Gradio)
state/ → Shared application state
datasets/ → Practice datasets
docs/ → Developer and theory documentation
```

This project follows a strict layered architecture.  
For full architectural details, see:

```
docs/developers/architecture.md
```

---

# Datasets

Practice datasets are located in:

```
datasets/practice/
```

These datasets are used for exploring descriptive statistics, inference, hypothesis testing, and regression within the application.

---

# Theory Notes

Statistical theory supporting this software can be found in:

```
docs/theory/ProbAndStatistics.pdf
```

---

# Contributing (Students & Developers)

Students are encouraged to extend the software by:

- Adding new statistical methods
- Implementing alternative estimators
- Expanding visualizations
- Improving documentation

Before adding new features, read:

```
docs/developers/architecture.md
docs/developers/adding_new_feature.md
```

Architectural rules are strictly enforced to maintain long-term stability.

---

# About Himmapan Lab

Himmapan Lab is an academic initiative focused on building modular mathematical software ecosystems for engineering education.

Each project in the lab follows consistent architectural principles to enable:

- Clear student onboarding
- Sustainable expansion
- Cross-project collaboration
- Long-term scalability

---

# License

(To be added)

