# Data Flow Diagram

1. **User Interface (HTML/CSS/JS):** Captures user inputs via forms.
2. **Web Server (Flask App):** Receives HTTP POST request with application data.
3. **Data Processing (Pandas/Numpy):** Formats the data to match model expectations.
4. **Machine Learning Model (scikit-learn/XGBoost):** Evaluates the processed data and generates a prediction and confidence score.
5. **Database (MongoDB):** The Flask app saves the application details and the prediction result into the `predictions` collection.
6. **Response:** Flask renders the result template and sends it back to the User Interface.