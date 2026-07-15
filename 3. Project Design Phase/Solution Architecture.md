# Solution Architecture

The architecture of SmartLender follows a classic 3-tier model:

1. **Presentation Layer (Frontend):** 
   - Built with HTML, CSS, and JS. 
   - Features templates for Home, Login, Signup, Dashboard, Predict, and Results.

2. **Application Layer (Backend & ML):**
   - **Flask API:** Handles routing, session management, and logic.
   - **ML Pipeline:** `pipeline.pkl` processes incoming data and returns predictions using XGBoost.

3. **Data Layer (Database):**
   - **MongoDB:** Stores user credentials and prediction history in a flexible JSON-like format.