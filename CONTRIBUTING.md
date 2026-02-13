# Contributing to Thotsakan Statistics  
## Himmapan Lab

Thank you for contributing to Thotsakan Statistics.

This project is part of the **Himmapan Lab** ecosystem and is designed to support:

- Statistical correctness
- Clean architecture
- Student collaboration
- Long-term sustainability

Contributions are welcome, but architectural discipline is mandatory.

---

# Who Can Contribute?

- Undergraduate students
- Teaching assistants
- Himmapan Lab collaborators
- Maintainers

If you are new to the project, read:

```
docs/developers/index.md
```

before making changes.

---

# Before You Start

1. Read:
   - `docs/developers/architecture.md`
   - `docs/developers/adding_new_feature.md`
   - `docs/developers/coding_rules.md`

2. Ensure you can:
   - Run the application locally
   - Run tests successfully

3. Create and activate a virtual environment.

---

# Development Setup

## Install runtime dependencies

```
pip install -r requirements.txt
```

## Install development dependencies

```
pip install -r requirements-dev.txt
```

## Run tests

```
pytest -q
```

All tests must pass before submitting changes.

---

# Contribution Workflow

## 1. Create a Branch

Do not work directly on `main`.

Create a feature branch:

```
git checkout -b feature/short-description
```

Examples:

- `feature/bootstrap-ci`
- `feature/new-visualization`
- `fix/controller-validation-bug`

---

## 2. Follow the Development Order

Always implement in this order:

```
Core → Controller → UI
```

Never start with the UI.

---

## 3. Add Tests

At minimum:

- Ensure imports still work.
- Add a basic test for new statistical logic.

Even small contributions should include at least one test if possible.

---

## 4. Run Tests

Before committing:

```
pytest -q
```

If tests fail, fix them before pushing.

---

## 5. Submit a Pull Request

Include:

- Clear description of the feature or fix
- Explanation of where changes were made
- Any architectural considerations

Small, focused pull requests are preferred over large ones.

---

# Architectural Rules (Non-Negotiable)

- Core must not import UI or Controllers.
- Controllers must not implement statistical formulas.
- UI must not call Core directly.
- No rounding inside Core.
- Estimator choices must not be hard-coded if user-configurable.

Violations will require revision.

---

# Code Style

- Use descriptive names.
- Prefer clarity over cleverness.
- Keep functions focused and small.
- Avoid unnecessary abstraction.

This is an educational project. Code should be understandable.

---

# Datasets

If adding datasets:

- Place them in the correct folder (`practice/` or `internal/`).
- Ensure no private or sensitive data.
- Use clear filenames.
- Prefer CSV format.

---

# Reporting Issues

When reporting a bug:

Include:

- Description of the problem
- Steps to reproduce
- Expected behavior
- Observed behavior

Clear reports help maintain project quality.

---

# Himmapan Lab Vision

This project is part of a broader initiative to build modular mathematical software for engineering education.

Contributors are not only writing code — they are building:

- Reproducible computational tools
- Clean architectural systems
- Long-term academic infrastructure

Discipline today ensures scalability tomorrow.

---

# Final Reminder

If you are unsure about:

- Where code belongs
- Architectural implications
- Estimator logic
- Testing strategy

Ask before implementing.

Maintaining architectural integrity is more important than implementing quickly.
