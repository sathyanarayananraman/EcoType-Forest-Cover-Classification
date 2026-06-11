# 🌲 EcoType: Forest Cover Type Prediction Using Machine Learning

## 📌 Overview

EcoType is a machine learning project designed to predict forest cover types based on cartographic and environmental features such as elevation, slope, soil type, and wilderness area. The solution helps automate forest classification to support environmental monitoring, forestry management, ecological research, and land-use planning.

---

## 🎯 Objectives

* Predict forest cover types using supervised machine learning.
* Analyze environmental and geospatial factors influencing forest distribution.
* Compare multiple classification algorithms and select the best-performing model.
* Deploy the final model through an interactive Streamlit application.

---

## 📚 Skills Demonstrated

* Exploratory Data Analysis (EDA)
* Data Cleaning & Preprocessing
* Feature Engineering
* Feature Selection
* Class Imbalance Handling (SMOTE / RandomOverSampler)
* Machine Learning Classification
* Hyperparameter Tuning
* Model Evaluation
* Streamlit Application Development

---

## 🌿 Domain

Environmental Data Analytics & Geospatial Predictive Modeling

---

## 📊 Dataset Information

* **Dataset Size:** 145,891 Rows × 13 Columns
* **Target Variable:** Cover_Type (7 Classes)

### Features

* Elevation
* Aspect
* Slope
* Horizontal Distance to Hydrology
* Vertical Distance to Hydrology
* Horizontal Distance to Roadways
* Hillshade (9am, Noon, 3pm)
* Horizontal Distance to Fire Points
* Wilderness Area
* Soil Type

---

## 🔄 Project Workflow

### 1. Data Collection

* Loaded dataset using Pandas.
* Reviewed dataset structure and target classes.

### 2. Data Understanding

* Statistical analysis using:

  * `.shape()`
  * `.info()`
  * `.describe()`
  * `.value_counts()`
* Checked missing values, duplicates, and class distribution.

### 3. Data Preprocessing

* Missing value treatment.
* Outlier detection using IQR/Z-score methods.
* Skewness correction using transformations.
* Categorical encoding.

### 4. Feature Engineering

* Created derived environmental features.
* Generated interaction variables where relevant.
* Saved encoders for inference consistency.

### 5. Exploratory Data Analysis

* Univariate Analysis
* Bivariate Analysis
* Correlation Heatmaps
* Class Distribution Analysis
* Feature Importance Visualization

### 6. Class Imbalance Handling

* RandomOverSampler
* SMOTE

### 7. Feature Selection

* Random Forest Feature Importance
* Correlation-Based Selection

### 8. Model Development

Implemented and evaluated:

* Random Forest Classifier
* Decision Tree Classifier
* Logistic Regression
* K-Nearest Neighbors (KNN)
* XGBoost Classifier

### 9. Hyperparameter Tuning

* GridSearchCV
* RandomizedSearchCV

### 10. Model Deployment

* Saved best model using Pickle/Joblib.
* Built Streamlit application for real-time predictions.

---

## 📈 Model Evaluation

Models were compared using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report
* Cross-Validation

---

## 🖥️ Streamlit Application

### Features

* User-friendly interface
* Numeric input fields
* Dropdown selections
* Real-time forest cover prediction
* Model inference using saved `.pkl` file

---

## 💼 Real-World Applications

### 🌲 Forest Resource Management

Supports forest classification for conservation and planning.

### 🔥 Wildfire Risk Assessment

Combines vegetation information with fire-risk analysis.

### 🗺️ Land Cover Mapping

Assists geospatial analysts in monitoring land-use patterns.

### 🌱 Ecological Research

Supports biodiversity and habitat studies.

---

## 🛠️ Tech Stack

### Programming

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Version Control

* Git
* GitHub

---

## 📦 Deliverables

* Data Analysis Notebook
* Feature Engineering Pipeline
* Trained ML Model (.pkl)
* Performance Evaluation Reports
* Visualizations & Insights
* Streamlit Web Application

---

## 🚀 Future Enhancements

* Integrate geospatial mapping visualizations.
* Deploy application to cloud platforms.
* Add satellite imagery-based classification.
* Implement deep learning models for improved accuracy.

---


