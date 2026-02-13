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

```
architecture.md
```

This document defines:

- The layered architecture
- Dependency rules
- Responsibilities of each layer
- Explicit anti-patterns

It is the authoritative reference for architectural decisions.

---

## 2. Adding a New Feature

```
adding_new_feature.md
```

This document explains:

- How to implement a feature correctly
- The correct development order (Core → Controller → UI)
- Where new code belongs
- Common mistakes to avoid

Read this before writing any new code.

---

# Core Architectural Rule

Dependencies always flow downward:

```
UI
↓
Controllers
↓
Core
```

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
