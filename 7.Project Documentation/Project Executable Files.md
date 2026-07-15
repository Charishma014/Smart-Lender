# Project Executable Files

## Running the Project

1. **Environment Setup:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   - Copy `.env.example` to `.env` and configure `MONGODB_URI`.

3. **Train the Model (Optional, if pipeline.pkl is missing):**
   ```bash
   python data/generate_dataset.py
   python train_model.py
   ```

4. **Run the Server:**
   ```bash
   python app.py
   ```
   - Access the app at `http://127.0.0.1:5000`.