# 🏡 House Price Prediction Using XGBoost and Ensemble Regression Models

## 📌 Project Overview

This project focuses on building a predictive machine learning model to estimate the sale prices of residential properties. Using structured real estate data, the goal was to apply regression models and ensemble techniques to deliver accurate and interpretable predictions. The project follows a full machine learning pipeline from data loading and cleaning to model tuning and final evaluation.

---

## 🎯 Problem Statement

Accurate prediction of housing prices is essential for buyers, sellers, developers, and lenders to make informed decisions. In this project, we aim to answer:

 **"Which features most accurately predict house sale prices, and how do ensemble models like XGBoost compare to traditional regressors in terms of performance?"**

---

## 📦 Dataset Description

- **Source**: Kaggle – *House Prices: Advanced Regression Techniques*
- **Rows**: 1,460
- **Features**: 80+ (numerical and categorical)
- **Target Variable**: `SalePrice` – the property's final sale price in USD

The dataset includes features such as:
- `GrLivArea`: Above ground living area (sq ft)
- `OverallQual`: Overall quality of the material and finish
- `Neighborhood`: Physical location within Ames city limits
- `YearBuilt`, `GarageCars`, `FullBath`, and more.

---

## 🧱 ML Pipeline Summary

### 1. **Data Preparation**
- Handled missing values using domain-informed imputation
- Dropped uninformative columns
- Encoded categorical variables (Label & One-Hot Encoding)
- Applied log transformation to skewed features like `SalePrice`

### 2. **Exploratory Data Analysis (EDA)**
- Visualized SalePrice distribution and applied log transform
- Identified strong correlations with `OverallQual`, `GrLivArea`
- Explored `Neighborhood` impact using boxplots
- Created heatmaps and scatter plots to uncover relationships

### 3. **Modeling**
Trained and evaluated the following models:
- Linear Regression (Baseline)
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Tuned XGBoost using `GridSearchCV`

### 4. **Evaluation Metrics**
Used both log scale and real-dollar scale:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Adjusted R² Score

### 5. **Feature Importance**
Extracted feature importance from the final tuned XGBoost model. Top predictors included:
- `ExterQual`
- `OverallQual`
- `GrLivArea`
- `GarageCars`

---

## 🔍 Model Comparison (Summary)

| Model              | RMSE (log) | MAE (log) | R² Score | Adjusted R² | RMSE ($) | MAE ($)   |
|--------------------|------------|-----------|-----------|--------------|-----------|------------|
| Linear Regression  | 0.16       | 0.09      | 0.862     | 0.490        | $25,768   | $15,092    |
| Decision Tree      | 0.18       | 0.11      | 0.847     | 0.432        | $30,752   | $17,928    |
| Random Forest      | 0.15       | 0.10      | 0.883     | 0.568        | $30,098   | $17,671    |
| XGBoost (Default)  | 0.14       | 0.09      | 0.888     | 0.588        | $26,902   | $16,599    |
| **Tuned XGBoost**  | **0.13**   | **0.09**  | **0.905** | **0.650**    | **$26,885** | **$15,947** |

✅ **Tuned XGBoost outperformed all other models** across all evaluation metrics.

---

## Presentation

- Project overview
- Data insights from EDA
- Model selection and evaluation
- Final takeaways and limitations


---

## 🧠 Key Takeaways

- XGBoost with hyperparameter tuning gave the best prediction performance
- EDA played a critical role in identifying valuable features
- A full end-to-end ML pipeline was implemented independently

---

## 📁 Project Structure

HOUSE-PRICE-PREDICTION/
│
├── 📁 data/                          
│   ├── data_description.txt          # Description of all 80+ features
│   ├── house_prices.csv              # Renamed version of train.csv
│   ├── test.csv                      # Test data (optional for final submission)
│   └── train.csv                     # Original training data from Kaggle
│
├── 📁 notebook/
│   └── House_Price_Prediction.ipynb  # Final Jupyter notebook (end-to-end ML pipeline)
│
├── 📁 results/
│   └── 📁 screenshoots/              # Visual outputs used in the presentation
│       ├── boxplot_neighborhood.png
│       ├── correlation_heatmap.png
│       ├── eda_saleprice_dist.png
│       ├── final_comparison.png
│       ├── grlivarea_vs_saleprice.png
│       ├── model_rmse_bar_chart.png
│       ├── residual_plot_xgb.png
│       └── xgb_feature_importance.png
│
├── README.md                         # Project documentation (this file)
└── requirements.txt                  # Required Python libraries


---

## 📌 Future Improvements

- Include external datasets (e.g., school ratings, crime index)
- Model feature interactions explicitly (e.g., OverallQual × GrLivArea)
- Deploy the final model as a web application

---

## 🚀 Tools & Libraries

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- XGBoost
- Jupyter Notebook

---


## 🧠 Final Reflection

Working on this project helped me reinforce my understanding of the complete supervised regression workflow. Unlike my previous project which was focused on classification, this project introduced me to more nuanced concepts like log-transforming skewed targets, interpreting RMSE on different scales, and analyzing residuals to detect model bias.

I especially gained confidence in using ensemble models like Random Forest and XGBoost, and tuning them with GridSearchCV. Visualizing feature importance and comparing models through metrics and plots added an interpretability layer that’s essential in real-world applications.

Overall, this project strengthened my skills in building robust, end-to-end machine learning pipelines for regression tasks.

