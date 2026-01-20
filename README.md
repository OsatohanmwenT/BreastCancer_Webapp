# Breast Cancer Prediction System

A machine learning project to predict whether a breast tumor is benign or malignant based on digitized image features.

## Project Overview
This system uses a Logistic Regression model trained on the Breast Cancer Wisconsin (Diagnostic) dataset. The web interface allows users to input tumor characteristics and get an instant prediction.

## Features Used
- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Concavity Mean

## Project Structure
```
/BreastCancer_Project_yourName_matricNo/
|- app.py               # Flask application
|- requirements.txt     # Python dependencies
|- Procfile            # Deployment configuration (Render)
|- BreastCancer_hosted_webGUI_link.txt # Submission details
|- /model/
     |- model_building.ipynb  # Jupyter notebook for training
     |- model_development.py  # Python script for training
     |- breast_cancer_model.pkl # Trained model (joblib)
|- /static/
     |- style.css       # CSS styling
|- /templates/
     |- index.html      # HTML template
```

## Setup Instructions

1. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Train Model (Optional)**
   The model is already included in `model/breast_cancer_model.pkl`. To retrain:
   ```
   python model/model_development.py
   ```

3. **Run Web App**
   ```
   python app.py
   ```
   Open http://localhost:5000 in your browser.

## Deployment (Render.com)

1. Upload this repository to GitHub.
2. Sign up/Login to [Render.com](https://render.com).
3. Click "New +" -> "Web Service".
4. Connect your GitHub repository.
5. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Click "Create Web Service".
7. Once deployed, copy the URL and paste it into `BreastCancer_hosted_webGUI_link.txt`.

## Disclaimer
This project is for educational purposes only and should not be used for medical diagnosis.
