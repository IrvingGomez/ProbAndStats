# Adding a New Feature
## Step-by-step Guide for Contributors and Students

---

## 1. Purpose of this document

This document explains **how to correctly add a new feature** to the Thotsakan Statistics project.

It is written primarily for:
- Student contributors
- New developers joining the project
- Anyone unfamiliar with the internal architecture

The goal is to help you add features **without breaking architectural guarantees** or introducing technical debt.

This guide assumes you have already read `architecture.md`.

---

## 2. What counts as a “new feature”?

A *new feature* is any of the following:

- A new statistical method (e.g. a new test, estimator, interval)
- A new visualization
- A new tab or sub-tab
- An extension of existing functionality (new options, estimators, plots)

Cosmetic UI changes (labels, text, layout tweaks) are **not** considered new features and usually affect only the UI layer.

---

## 3. The golden rule (read this first)

**Never start by modifying the UI.**

Always begin with the **statistical logic**, then work upward.

Correct order:

```
Core → Controller → UI
```


If you start from the UI, you will almost certainly put logic in the wrong place.

---

## 4. Step-by-step workflow

---

### Step 1 — Clearly define the feature

Before writing code, answer these questions:

1. What is the **statistical task**?
2. What are the **inputs**?
3. What are the **outputs**?
4. Is this:
   - a new method?
   - an alternative estimator?
   - a new visualization?
   - a new tab?

Write this down **before coding**.

Example:
> “Add a bootstrap-based confidence interval for the trimmed mean.”

---

### Step 2 — Implement the statistical logic (Core layer)

Location:
- `core/`
- or `core/estimation/`
- or `core/hypothesis_tests.py`
- or `core/linear_regression.py`

What to do:
- Implement the statistical function
- Return raw numerical results, DataFrames, or Matplotlib figures
- Raise Python exceptions on invalid input

What **not** to do:
- Do NOT import Gradio
- Do NOT round values for display
- Do NOT access application state
- Do NOT format strings for presentation

Example signature:

```python
def ci_trimmed_mean_bootstrap(
    data: np.ndarray,
    alpha: float,
    trim_param: float,
    B: int
) -> tuple[float, float]:
    ...
```

---

### Step 3 — Expose the feature through a controller

Location

- `controllers/`
- or `controllers/estimation/`

What to do

- Validate user inputs  
- Convert UI values (strings, dropdowns) to proper types  
- Select which Core function to call  
- Apply rounding and formatting  
- Decide what to return to the UI  

What **not** to do

- Do **NOT** implement statistical formulas  
- Do **NOT** build UI layouts  
- Do **NOT** import UI tabs  

Example responsibilities:

- Parsing confidence levels  
- Choosing between analytic vs bootstrap  
- Selecting estimators  
- Handling edge cases gracefully  

Example pattern:

```python
def run_new_feature(...):
    alpha = parse_confidence_level(conf_level)
    result = core_function(...)
    return formatted_result
```

---

### Step 4 — Update the UI (tabs)

Location

- `ui/tabs/`
- or `ui/tabs/estimation/`

What to do

- Add UI controls (checkboxes, dropdowns, sliders)  
- Wire controls to the controller function  
- Manage visibility and interaction logic  

What **not** to do

- Do **NOT** implement statistics  
- Do **NOT** import Core modules  
- Do **NOT** manipulate raw numerical results  

The UI should only

- Collect user input  
- Call controllers  
- Display outputs  

---

### Step 5 — Test the feature incrementally

Recommended testing order

1. **Core**: call the function directly in Python  
2. **Controller**: call it with mock inputs  
3. **UI**: interactively test in the app  

If something breaks, debug **from the bottom up**.

---

## 5. Where should my code go?

Use this table as a decision guide:

| If your code...            | It belongs in |
|----------------------------|---------------|
| Computes statistics        | `core/`       |
| Chooses estimators         | `controllers/`|
| Parses user input          | `controllers/`|
| Rounds numbers             | `controllers/`|
| Builds plots               | `core/`       |
| Creates dropdowns          | `ui/`         |
| Toggles visibility         | `ui/`         |
| Stores data                | `state/`      |

---

## 6. Common mistakes to avoid

❌ Putting statistics in controllers  
❌ Importing Gradio in `core/`  
❌ Rounding inside statistical functions  
❌ UI tabs calling Core directly  
❌ Adding “quick fixes” in the UI  

These mistakes **will be rejected during review**.

---

## 7. How to ask for help (recommended)

If you are unsure:

- Write a short description of the feature  
- Specify where you think the code should go  
- Ask **before** implementing  

This saves time and prevents rewrites.

---

## 8. Final checklist before submitting code

- Core logic is independent of UI  
- Controller contains orchestration, not math  
- UI contains no statistics  
- No dependency rule violations  
- Feature matches existing behavior and style  

If all boxes are checked, your feature is ready.

---

## 9. Philosophy reminder

This project is designed to be:

- Statistically correct  
- Pedagogically clear  
- Easy to extend  
- Safe for student collaboration  

**Clarity beats cleverness.**
