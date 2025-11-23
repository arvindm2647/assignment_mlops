# CPU Usage Prediction Dashboard

🚀 **Live Demo:** [https://tbwkmvxkhkvye98cx5wzfh.streamlit.app](https://tbwkmvxkhkvye98cx5wzfh.streamlit.app)

## Assignment 2: ML Model Deployment

A machine learning application for predicting CPU usage using Random Forest regression, deployed with Streamlit.

### Features
- 🎯 Real-time CPU usage predictions
- 📊 Interactive model performance metrics
- 📈 Data analysis and visualizations
- 🤖 Auto-training on first deployment

### Tech Stack
- Python 3.14+ (compatible with Python 3.13, 3.12)
- Streamlit 1.51.0
- Scikit-learn 1.7.2
- Pandas 2.3.3, NumPy 2.3.5
- SciPy 1.16.3
- Plotly 6.4.0

### Model Details
- **Algorithm:** Random Forest Regressor
- **Features:** 5 numeric + encoded categorical
- **Framework:** Scikit-learn

## Recent Changes

### Version Updates (2025-01-23)
- **NumPy**: Updated from `2.2.6` to `2.3.5` for Python 3.14 compatibility
- **SciPy**: Updated from `1.15.3` to `1.16.3` for Python 3.14 compatibility
- **PyArrow**: Updated from `21.0.0` to `22.0.0` for Python 3.14 compatibility
- **Streamlit**: Fixed deprecation warnings by replacing `use_container_width=True` with `width='stretch'` in all chart and dataframe components

### Installation Notes
- Python 3.14 requires updated package versions with pre-built wheels
- All packages now use pre-built wheels (no compilation needed)
- If using Python 3.13 or 3.12, you may be able to use older package versions

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd Assigment_MLops
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/Mac**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

⚠️ **Important**: Always use `streamlit run` to launch the app. Do not run `python app.py` directly.

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

### Troubleshooting

If you encounter errors:
- **"missing ScriptRunContext" warnings**: You're running the app with `python app.py` instead of `streamlit run app.py`. Use the correct command above.
- **Package installation errors**: Ensure you're using Python 3.14+ or update package versions in `requirements.txt` for your Python version.
- **GCC/compilation errors**: All packages now use pre-built wheels, so compilation should not be needed. If you see compilation errors, ensure you have the latest pip version: `pip install --upgrade pip`