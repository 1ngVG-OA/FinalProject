# Demographic Time Series Forecasting Results Report

## Overview
This report presents the results of a comprehensive time series forecasting analysis on a demographic time series taken from the M3C dataset, comparing four different modeling approaches: SARIMA, Holt-Winter's Exponential Smoothing, Multi-Layer Perceptron (MLP), and XGBoost.

## Dataset Information
- **Source File:** demographic.csv
- **Training samples:** 111
- **Validation samples:** 12
- **Test samples:** 12
- **Total observations:** 135

## Data Preprocessing Pipeline
- **Transformation Method:** Box-Cox transformation for variance stabilization only for SARIMA
- **Scaling:** Applied only for MLP model (Standard scaler)

## Model Configurations

### 1. SARIMA (Seasonal AutoRegressive Integrated Moving Average)
**Model Type:** Statistical time series model

> *The hyperparameters of the model were selected automatically using the auto_arima function from the pmdarima library with the differencing parameter manually set to two (d=2). The Augmented Dickey-Fuller (ADF) test confirmed that the series is non-stationary (p-value > 0.05), requiring second-order differencing to achieve stationarity. Autocorrelation and partial autocorrelation plots indicated significant seasonality with a period of 12, leading to the implementation of a SARIMA(p,2,q)(P,D,Q)[12] specification.*

**Final selected model:**

- **ARIMA Order:** (2, 2, 1)
- **Seasonal Order:** (2, 1, 1, 12)
- **Seasonal Period:** 12 months
- **Parameter Selection:** Auto-optimized using AIC criterion

**Performance Metrics:**
| Metric   | Validation | Test    |
|----------|------------|---------|
| RMSE     | 27.4790    | 61.8194 |
| MAE      | 23.7475    | 41.7500 |
| MAPE (%) | 0.37       | 0.64    |

### 2. Holt-Winter's Exponential Smoothing
**Model Type:** Statistical time series model

> *The hyperparameters of the model were selected through a grid search, exploring different configurations of damped_trend, trend, seasonal, and seasonal periods.*

#### Grid Search Parameter Space

- **damped_trend:** {True, False}
- **boxcox:** {True, False}
- **trend:** {None, 'add', 'mul'}
- **Seasonal:** {'add', 'mul'}
- **Seasonal Periods:** 12 months
- **initialization_method:** estimated

> *The final configuration was selected based on the lowest RMSE obtained on the validation set.*

**Final selected model:**

- **damped_trend:** True
- **boxcox:** True
- **trend:** add
- **Seasonal:** mul
- **Seasonal Periods:** 12 months
- **initialization_method:** estimated

**Performance Metrics:**
| Metric   | Validation | Test    |
|----------|------------|---------|
| RMSE     | 12.4705    | 54.1925 |
| MAE      | 10.1550    | 35.1208 |
| MAPE (%) | 0.16       | 0.54    |

### 3. MLP (Multi-Layer Perceptron)
**Model Type:** Neural network regression model

> *The architecture and hyperparameters of the model were selected through a grid search, exploring different configurations of look-back window, hidden layer size, dropout rate, batch size, learning rate, and activation function.*

#### Grid Search Parameter Space

- **Look-back window:** {12, 18, 24, 30, 36}  
- **Hidden layer size:** {2, 4, 6, 8}  
- **Learning rate:** {0.01, 0.001, 0.0001}  
- **Dropout:** {0, 0.1}  
- **Batch size:** {8, 16, 32}  
- **Activation function:** {ReLU, tanh}

> *The final configuration was selected based on the lowest RMSE obtained on the validation set.*

**Final selected model:**

- **Architecture:** Input Layer → Dense (2 units, ReLU) → Dense (1 unit, Linear)  
- **Look-back window:** 30 months  
- **Training epochs:** 500  
- **Batch size:** 16  
- **Optimizer:** Adam with learning rate = 0.01 
- **Activation function:** ReLU 
- **Loss function:** Mean Squared Error (MSE)

**Performance Metrics:**
| Metric   | Validation | Test    |
|----------|------------|---------|
| RMSE     | 24.3552    | 84.2840 |
| MAE      | 18.5499    | 68.7237 |
| MAPE (%) | 0.29       | 1.05    |


### 4. XGBoost (Extreme Gradient Boosting)
**Model Type:** Ensemble tree-based regression model

> *The model's hyperparameters were selected through a grid search exploring various configurations of look-back window, number of trees, tree depth, learning rate (eta), and subsampling ratios.*

#### Grid Search Parameter Space

- **Look-back window:** {12, 18, 24, 30, 36}  
- **Number of estimators (trees):** {5, 10, 15, 20, 30, 50, 80, 100}  
- **Max depth:** {1, 3, 5, 7}  
- **Learning rate (eta):** {0.1, 0.2, 0.4, 0.6, 0.8, 1}  
- **Subsample:** {0.1, 0.3, 0.7, 1}  
- **Colsample bytree:** {0.1, 0.3, 0.7, 1}  
- **Random seed:** 1234  

> *The final configuration was selected based on the lowest RMSE obtained on the validation set.*

**Final selected model:**

- **look-back window:** 24 months  
- **Number of estimators:** 15  
- **Max depth:** 3  
- **Learning rate (eta):** 0.8  
- **Subsample:** 0.1  
- **Colsample bytree:** 0.3  


**Performance Metrics:**
| Metric   | Validation | Test    |
|----------|------------|---------|
| RMSE     | 22.4340    | 59.6968 |
| MAE      | 17.7634    | 45.4805 |
| MAPE (%) | 0.28       | 0.70    |

## Comparative Analysis

### Validation Set Performance
| Model         | RMSE        | MAE        | MAPE (%) | Rank by RMSE |
| ------------- | ----------- | ---------- | -------- | ------------ |
| Holt-Winter's | 12.4705     | 10.1550    | 0.16     | 1            |
| XGBoost       | 22.4340     | 17.7634    | 0.28     | 2            |
| MLP           | 24.3552     | 18.5499    | 0.29     | 3            |
| SARIMA        | 27.4790     | 23.7475    | 0.37     | 4            |

### Test Set Performance (Final Evaluation)
| Model         | RMSE        | MAE        | MAPE (%) | Rank by RMSE |
| ------------- | ----------- | ---------- | -------- | ------------ |
| Holt-Winter's | 54.1925     | 35.1208    | 0.54     | 1            |
| XGBoost       | 59.6968     | 45.4805    | 0.70     | 2            |
| SARIMA        | 61.8194     | 41.7500    | 0.64     | 3            |
| MLP           | 84.2840     | 68.7237    | 1.05     | 4            |

## Key Results

### Best Performing Model
**Holt-Winter's** emerged as the top-performing model, achieving the lowest error rates across all evaluation metrics.