# DiamondsDataML

DiamondsDataML — Diamond Price Prediction

Machine learning for diamond price prediction using the classic diamonds dataset from Kaggle. The workflow covers EDA, data cleaning, feature preparation, model training, hyperparameter tuning, and evaluation. The entire work is in MLdiamonds.ipynb.

Dataset

Source file: /kaggle/input/diamonds/diamonds.csv

Shape (initial): 53,940 × 11

Target: price

Features:

Numeric: carat, depth, table, x, y, z

Categorical: cut (Fair–Ideal), color (J–D), clarity (I1–IF)

Before cleaning: No missing values; some geometric dimensions (x, y, z) include impossible zeros.

Project Flow

1.EDA

  head(), info(), describe()

  Pairwise relationships via pairplot

  Scatter plots of price vs. x, y, z, depth, table

2.Data Cleaning & Filtering

3.Feature Preparation

  LabelEncoder for cut, color, clarity

  StandardScaler for numeric features

  Train/Test split: test_size=0.25, random_state=15

4.Modeling

  Linear Regression (baseline)

  SVR (default hyperparameters)

  SVR + GridSearchCV (hyperparameter optimization)

5.Evaluation

  Metrics: MAE, MSE, R²

  Scatter plots (true vs. predicted)

Data Cleaning & Filters

  Removed physically impossible measurements where any of x, y, or z equals 0
  
  x==0: 8 rows; y==0: 7 rows; z==0: 20 rows
  
  Applied reasonable ranges:
  
  45 < depth < 75
  
  40 < table < 75
  
  2 < z < 30
  
  y < 20

  Final shape after cleaning: 53,905 × 10

Rationale: Zero dimensions are not physically valid; filtering reduces the impact of extreme outliers on model performance.

Modeling

Features/Target:

X = df.drop("price", axis=1)
y = df["price"]

Encoding: LabelEncoder for cut, color, clarity

Scaling: StandardScaler (fit on train, transform on test)

Split: train_test_split(test_size=0.25, random_state=15)


Tried Models
  
  LinearRegression (baseline)
  
  SVR (default)
  
  SVR + GridSearchCV
  
  Search space:
  
  kernel ∈ {linear, rbf}
  
  C ∈ {0.1, 1, 10, 100, 1000}
  
  gamma ∈ {1, 0.1, 0.001}

Results

All scores are on the test set.

| Model                  |        MAE |         MSE |         R² |
| ---------------------- | ---------: | ----------: | ---------: |
| LinearRegression       |     867.77 |   1,849,507 |     0.8849 |
| SVR (default)          |   1,397.85 |   8,151,981 |     0.4926 |
| **SVR (GridSearchCV)** | **480.67** | **874,992** | **0.9455** |


Best model: SVR (rbf)
Best hyperparameters: {'C': 1000, 'gamma': 0.1, 'kernel': 'rbf'}
