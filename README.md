Forest Cover Type Classification

This repository contains a machine learning project that compares Random Forest and XGBoost classifiers for predicting forest cover types using the UCI Covertype dataset. The project includes data preprocessing, model training, hyperparameter tuning, performance evaluation, and result visualization.

📁 Dataset

Source: UCI Covertype Dataset

Description: Predicts forest cover type from cartographic variables.

Target Classes: 7 forest cover types labeled from 1 to 7.

📊 Models Compared

✅ Random Forest

Baseline model using default scikit-learn parameters.

✅ XGBoost

Trained using default settings.

Tuned with RandomizedSearchCV for improved performance.

🔧 Features

Data preprocessing: Scaling using StandardScaler

Train/Test Split with stratification

Multi-class classification

Hyperparameter tuning for XGBoost

Model evaluation using:

Accuracy, Precision, Recall, F1-score

Confusion matrix (heatmap)

Feature importance visualization

Precision‑Recall curve (Class 2 example)

📦 Requirements

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib

▶️ Usage

Clone the repository:

git clone https://github.com/githubmostafa/forest-cover-classification.git
cd forest-cover-classification

Open the Jupyter Notebook forest_cover_classification.ipynb

Run cells in order to reproduce results

💾 Model Saving

Trained models are saved using joblib:

xgb_model_tuned.pkl

random_forest_model.pkl

📈 Sample Results

Below are the accuracy scores for each model:

Random Forest Accuracy: 87%

XGBoost Accuracy: 86%

XGBoost Tuned Accuracy: 88%

📊 Result Summary

The Random Forest model performed well with 87% accuracy using default parameters.

The XGBoost model gave slightly lower performance (86%) before tuning.

After applying hyperparameter tuning, the XGBoost Tuned model achieved the best accuracy of 88%, demonstrating the impact of optimization techniques.

📌 License

This project is for educational and research purposes.

