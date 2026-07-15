# Code Layout, Readability, and Reusability

## Directory Structure
- **`app.py`**: Central application logic. Routes are clearly defined and separated.
- **`database.py`**: Encapsulates all MongoDB connection and query logic for reusability.
- **`model/`**: Contains the trained `pipeline.pkl` and `metadata.json`, separating ML artifacts from web logic.
- **`templates/` & `static/`**: Clean separation of views (HTML) and presentation (CSS/JS).

## Readability
- Code is documented using Python docstrings.
- Variables are descriptively named (e.g., `applicant_income`, `loan_amount`).
- Modular functions are used to avoid repetition.