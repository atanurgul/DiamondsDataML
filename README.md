# 💎 DiamondsDataML — Diamond Price Prediction

A complete machine learning project for **diamond price prediction** using the classic *Diamonds* dataset from Kaggle.  
This notebook demonstrates the full data science workflow — from **EDA** and **data cleaning** to **modeling**, **hyperparameter tuning**, and **evaluation**.  
All work is contained in **`MLdiamonds.ipynb`**.

---

## 📊 Dataset Overview

- **Source:** `/kaggle/input/diamonds/diamonds.csv`  
- **Shape:** 53,940 × 11  
- **Target variable:** `price`  

**Feature types:**
- **Numeric:** `carat`, `depth`, `table`, `x`, `y`, `z`
- **Categorical:** `cut`, `color`, `clarity`

> The dataset had no missing values, but contained invalid records (x, y, z = 0) which were removed.  
> After cleaning, 53,905 valid samples remained.

---

## 🔍 Exploratory Data Analysis (EDA)

- Inspected data types, ranges, and summary statistics with `df.info()` and `df.describe()`
- Visualized pairwise feature relationships using `seaborn.pairplot`
- Examined correlations between `price` and numerical variables (`carat`, `x`, `y`, `z`, `depth`, `table`)
- Checked categorical distributions (`cut`, `color`, `clarity`) and their effects on price
- Identified outliers and unrealistic values in geometric dimensions

**Observations:**
- `carat` is strongly correlated with `price`
- Higher quality (`cut`, `color`, `clarity`) generally increases price
- Outliers (depth > 75, table > 75, or zero dimensions) distort relationships

---

## 🧹 Data Cleaning and Filtering

**Removed invalid or extreme values:**
- Dropped all rows where `x`, `y`, or `z` = 0  
- Applied reasonable range filters:  
  45 < depth < 75
  40 < table < 75
  2 < z < 30
  y < 20
✅ **Final shape:** 53,905 × 10  

> These steps ensure only physically possible and realistic diamond measurements remain for modeling.

---

## ⚙️ Feature Preparation

1. **Encoding:**  
 - Used `LabelEncoder` to convert categorical columns: `cut`, `color`, `clarity`
2. **Scaling:**  
 - Applied `StandardScaler` to all numeric features
3. **Train/Test Split:**  
 - 75% training and 25% testing  
 - `random_state=15` for reproducibility

✅ **Final shape:** 53,905 × 10  

> These steps ensure only physically possible and realistic diamond measurements remain for modeling.

---

## ⚙️ Feature Preparation

1. **Encoding:**  
 - Used `LabelEncoder` to convert categorical columns: `cut`, `color`, `clarity`
2. **Scaling:**  
 - Applied `StandardScaler` to all numeric features
3. **Train/Test Split:**  
 - 75% training and 25% testing  
 - `random_state=15` for reproducibility

🤖 Modeling Approach

Three regression approaches were tested:

Linear Regression — baseline model

Support Vector Regression (SVR) — nonlinear kernel-based method

SVR with GridSearchCV — hyperparameter tuning for best performance


Model Performance

All results evaluated on the test set:

| Model                  |        MAE |         MSE |         R² |
| ---------------------- | ---------: | ----------: | ---------: |
| LinearRegression       |     867.77 |   1,849,507 |     0.8849 |
| SVR (default)          |   1,397.85 |   8,151,981 |     0.4926 |
| **SVR (GridSearchCV)** | **480.67** | **874,992** | **0.9455** |


Developed by Atanur Gül
https://www.kaggle.com/atanurgl
