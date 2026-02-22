# Getting Started

This guide explains how to set up and run **Thotsakan Statistics** on your local machine.

If you are new to Python environments, follow the steps carefully.  
You only need to perform the setup once per machine.

---

# 1. Prerequisites

Before starting, make sure you have:

- Python 3.10 or newer installed  
- A terminal (Git Bash, PowerShell, macOS Terminal, or Linux shell)

To check your Python version:

```
python --version
```

---

# 2. Clone the Repository

If you have not already cloned the project:

```
git clone <repository-url>
cd <repository-folder>
```

---

# 3. Create a Virtual Environment (Recommended)

A virtual environment keeps this project isolated from other Python projects on your computer.

From the project root:

## Windows (Git Bash / WSL)

```
python -m venv .venv
source .venv/Scripts/activate
```

## Windows (PowerShell)

```
python -m venv .venv
..venv\Scripts\activate
```

## macOS / Linux

```
python -m venv .venv
source .venv/bin/activate
```

After activation, your terminal should display:

```
(.venv)
```

You only need to create the environment once.

However, you must activate it each time you open a new terminal.

---

# 4. Install Dependencies

Once the virtual environment is activated:

```
pip install -r requirements.txt
```

This installs all required libraries for the application.

You only need to do this once unless the requirements change.

---

# 5. Run the Application

With the environment activated:

```
python app.py
```

The application will start, and a local interface will open in your browser.

---

# 6. If Something Goes Wrong

## Error: `ModuleNotFoundError`

Most common cause: the virtual environment is not activated.

Solution:
1. Activate the environment again.
2. Run the application.

## Error: Command not found

Ensure:
- Python is installed correctly.
- You are in the project root directory.

---

# 7. When You Return Later

Each time you reopen the project:

1. Open a terminal.
2. Navigate to the project folder.
3. Activate the virtual environment:

```
source .venv/Scripts/activate # Windows Git Bash
```

or

```
..venv\Scripts\activate # PowerShell
```

Then run:

```
python app.py
```

You do NOT need to recreate the environment.

---

# 8. Next Steps

Once the application runs successfully, you can:

- Load datasets from `datasets/practice/`
- Explore descriptive statistics
- Perform inference and hypothesis testing
- Run regression analyses

If you are interested in extending the software, read:

- [Architecture Guide](../project/architecture.md)  
- [Feature Workflow](../developers/feature_workflow.md)

---

Thotsakan Statistics is part of the **Himmapan Lab** ecosystem.  
Consistency and reproducibility are important. Using virtual environments ensures that everyone runs the same software under the same conditions.
