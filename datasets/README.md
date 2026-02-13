# Datasets  
## Thotsakan Statistics – Himmapan Lab

This folder contains datasets used for teaching, experimentation, and development within the Thotsakan Statistics application.

Datasets are divided into categories to clarify their intended use.

---

# Folder Structure

```
datasets/
├── practice/
└── internal/
```

---

# practice/

This folder contains datasets intended for:

- Classroom demonstrations
- Student experimentation
- Homework and project exploration
- Testing statistical methods in the UI

These datasets are stable and safe for users.

Examples of typical usage:

- Descriptive statistics
- Graphical analysis
- Confidence and prediction intervals
- Hypothesis testing
- Linear regression

Students are encouraged to explore these datasets freely.

---

# internal/

This folder is reserved for:

- Development testing
- Edge-case validation
- Experimental datasets
- Instructor-only material

Files here may:

- Change frequently
- Contain incomplete data
- Be used for debugging new features

Unless instructed otherwise, students should focus on `practice/`.

---

# Dataset Usage Guidelines

When adding a new dataset:

1. Place it in the appropriate folder (`practice/` or `internal/`).
2. Use clear, descriptive filenames.
3. Prefer CSV format for compatibility.
4. Ensure column names are readable and meaningful.
5. Avoid extremely large files unless necessary.

---

# Naming Conventions

- Use descriptive names.
- Avoid spaces in filenames.
- Use underscores instead of spaces.
- Include context in the name when helpful.

Good examples:

- `ElectricityDemand.csv`
- `Regression.csv`
- `DescriptiveOutlier.csv`

Avoid:

- `data1.csv`
- `test.csv`
- `newfile.csv`

---

# Adding New Datasets

If you are contributing a dataset:

- Confirm it is appropriate for educational use.
- Ensure no sensitive or private data is included.
- Document its purpose briefly in this file (optional but encouraged).

---

# Educational Purpose

Datasets are provided to:

- Bridge statistical theory and practice
- Allow students to compare estimators
- Analyze assumptions and robustness
- Interpret real-world data scenarios

The goal is not only computation, but understanding.

---

This folder is part of the Himmapan Lab ecosystem and should remain clean, organized, and pedagogically meaningful.
