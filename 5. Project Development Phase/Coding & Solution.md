# Coding & Solution

## Key Implementations

1. **Machine Learning Pipeline:**
   - Built using scikit-learn's `Pipeline` and `ColumnTransformer` to handle imputation, scaling, and encoding seamlessly before feeding data to the XGBoost classifier.

2. **Flask Backend:**
   - Uses `@app.route` decorators for clean API endpoints.
   - Implements session-based authentication for user security.

3. **MongoDB Integration:**
   - Uses `pymongo` to insert documents into the `predictions` collection asynchronously.

4. **Frontend Form Wizard:**
   - Implemented via JavaScript to show/hide sections, creating a smooth, multi-step user experience.