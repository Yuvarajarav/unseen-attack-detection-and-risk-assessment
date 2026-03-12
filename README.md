# Unseen Cyber Attack Detection and Risk Assessment System

A machine learning-based system designed to detect previously unseen cyber attacks and assess their risk levels. The project uses the UNSW-NB15 dataset to train classification models that identify malicious network traffic and generate cyber risk scores. An interactive dashboard built with Streamlit allows users to visualize predictions and analyze potential threats.

## Features

- Detection of unseen cyber attacks using machine learning models
- Risk score generation for detected network threats
- Classification of attacks into risk categories (High, Medium, Low)
- Interactive visualization dashboard for monitoring results
- Data-driven analysis of network traffic patterns

## Technologies Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib / Seaborn (for visualization)

## Dataset

The project uses the **UNSW-NB15 dataset**, which contains modern network traffic data with multiple attack categories. The dataset includes features related to network flow statistics, protocol information, and traffic behavior.

## System Workflow

1. Load and preprocess the UNSW-NB15 dataset.
2. Perform feature selection and data cleaning.
3. Train machine learning models to detect malicious traffic.
4. Use the XGBoost algorithm to improve classification accuracy.
5. Generate risk scores based on prediction probabilities.
6. Display results through an interactive Streamlit dashboard.

