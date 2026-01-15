# 🏠 Housing Price Prediction using Ensemble Learning

## 📌 Project Overview
This project focuses on the implementation of supervised Machine Learning algorithms to solve a dual problem in the Real Estate sector:
1.  **Regression:** Predicting the exact sale price of a property.
2.  **Classification:** Categorizing properties into price ranges (Low, Medium, High).

The solution compares the performance of single **Decision Trees** versus **Random Forest** (Ensemble Learning), demonstrating how bagging techniques significantly reduce variance and overfitting.

*Developed as part of the Artificial Intelligence Specialization at UNIR.*

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
* **Techniques:** Data Preprocessing, Grid Search CV (Hyperparameter Tuning), Cross-Validation.

## 📊 Methodology & Results

### 1. Data Engineering (ETL)
* Handling missing values (Imputation).
* Feature selection based on correlation analysis.
* Encoding categorical variables for model ingestion.

### 2. Model Performance
We evaluated models using RMSE (Root Mean Square Error) for regression and Accuracy for classification.

| Model | Task | Key Metric | Result |
|-------|------|------------|--------|
| **Decision Tree** | Regression | RMSE | High Variance (Overfitting observed) |
| **Random Forest** | Regression | RMSE | **~33,218** (Best Performance) |
| **Random Forest** | Classification | Accuracy | Optimized with GridSearch |

> **Key Insight:** The Random Forest model improved prediction stability significantly compared to the single Decision Tree, confirming the effectiveness of ensemble methods for high-dimensional datasets.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/camilo-ferro/Housing-Price-Prediction-Ensemble-Learning.git](https://github.com/camilo-ferro/Housing-Price-Prediction-Ensemble-Learning.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the models:**
    ```bash
    # To run the Decision Tree analysis
    python src/train_decision_tree.py

    # To run the Random Forest analysis (Best Model)
    python src/train_random_forest.py
    ```

## 📂 Project Structure
* `src/`: Contains the Python scripts for training and evaluation.
* `data/`: Contains the `housing_train.csv` dataset.
* `docs/`: Detailed laboratory report and findings.

---
**Author:** Camilo Ferro
*Specialization in Artificial Intelligence - UNIR*
