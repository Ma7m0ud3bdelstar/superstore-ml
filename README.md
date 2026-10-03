Superstore ML — Order Profitability Estimator

An end-to-end Machine Learning project built on the Sample Superstore dataset to analyze order profitability and estimate the financial risk of new orders.

The project combines classification and regression to answer two practical business questions:

1. How likely is an order to result in a loss?
2. What profit can we expect from the order?

The trained models are integrated into an interactive Streamlit application where users can enter order details and receive predictions instantly.

--------------------------------------------------------------------------------------------------------

 Project Objective
The main objective of this project is to transform historical Superstore order data into a practical ML-based decision-support tool.

The application allows users to enter:
- Product sub-category
- Unit price
- Quantity
- Discount
- State

The system then estimates:
- Order Sales
- Probability of Loss
- Expected Profit
- Loss/Profit Alert

This can help identify potentially risky orders before they are placed.

--------------------------------------------------------------------------------------------------------

Dataset
The project uses the **Sample Superstore** dataset.

The dataset contains historical sales transactions with information related to:
- Orders
- Products
- Customers
- Categories
- Locations
- Sales
- Profit
- Discounts

The dataset is used for both exploratory analysis and Machine Learning development.

--------------------------------------------------------------------------------------------------------

Machine Learning Approach
The project uses two separate Machine Learning models.

1. Loss Classification
A classification model is used to estimate the probability that an order will result in a loss.
Model:
HistGradientBoosting Classifier
The model outputs a probability that is used to determine whether an order should trigger a loss alert.
Two alert thresholds are available in the Streamlit application:
- 0.50 — Balanced
- 0.15 — Catch most losses
  
A lower threshold makes the system more sensitive to potential losses and therefore generates more alerts.

Profit Regression
A regression model is used to estimate the expected profit of an order.
Model:
Random Forest Regressor

The regression model predicts the expected profit based on the order features supplied by the user.

--------------------------------------------------------------------------------------------------------

Model Evaluation
The project evaluates the models using both random and time-based splits.
Classification Metrics

The loss classifier is evaluated using:
- ROC-AUC
- Recall
- Precision

Regression Metrics
The profit estimator is evaluated using:
- R²
- Median Absolute Error

The Streamlit application includes a **Model Card** that displays the evaluation results directly inside the application.

--------------------------------------------------------------------------------------------------------

Key Model Observations

- The loss classification model provides strong separation between loss-making and non-loss-making orders.
- The profit regression model performs well for typical orders, while predictions become less reliable for rare and very large orders.
- This is important when interpreting predictions because historical Superstore data contains orders with significantly different scales.
- The application therefore includes warnings when an input is outside the ranges observed during training.

--------------------------------------------------------------------------------------------------------

Streamlit Application
The trained models are integrated into an interactive Streamlit application called:

Order Profitability Estimator
The application provides:
- Product Information

Users can select:
- Sub-Category
- Unit Price
- Quantity
- Discount

Location
Users can select:
- State
The application automatically determines the corresponding region.

Predictions
The application displays:

- Order Value (Sales)
- Probability of Loss
- Estimated Profit
- Profitability Alert

--------------------------------------------------------------------------------------------------------

Input Sanity Checks

The application also checks whether the entered order is significantly different from the training data.

Warnings may appear when:

* The unit price is outside the historical range for the selected sub-category.
* The quantity is larger than values observed during training.
* The order value is unusually large.
* The discount is relatively high.

These warnings help communicate when a prediction should be treated as a rough estimate rather than a highly reliable prediction.

--------------------------------------------------------------------------------------------------------

File Description

| File                           | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `app.py`                       | Streamlit application                             |
| `features.py`                  | Feature construction used by the application      |
| `train_final.py`               | Final model training script                       |
| `requirements.txt`             | Python dependencies                               |
| `models/clf.joblib`            | Trained loss classification model                 |
| `models/reg.joblib`            | Trained profit regression model                   |
| `models/meta.json`             | Model metadata, metrics, mappings, and thresholds |
| `superstore_ml_final.ipynb`    | Main project notebook                             |
| `data/Sample - Superstore.csv` | Superstore dataset                                |

--------------------------------------------------------------------------------------------------------

Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

 Machine Learning
- HistGradientBoosting
- Random Forest
- Classification
- Regression
- Model evaluation
- Feature engineering

--------------------------------------------------------------------------------------------------------

Important Notes

- This project is a **Machine Learning portfolio project** based on the Sample Superstore dataset.

- The predictions should not be interpreted as guaranteed financial outcomes.

- The profit estimator is less reliable for unusual or very large orders because these cases are less represented in the training data.

- The application therefore provides additional warnings to help users understand when predictions may involve extrapolation.

---

Possible Future Improvements

Potential improvements include:
- More advanced feature engineering
- Hyperparameter optimization
- Additional model comparison
- Better handling of rare and high-value orders
- Model explainability using SHAP
- More detailed business dashboards
- Continuous model monitoring
- Deployment with automated CI/CD

--------------------------------------------------------------------------------------------------------

Project Goal

The goal of this project is to address a key business problem: understanding and reducing unprofitable orders in the Superstore business.
The company wants to understand which orders are likely to result in losses and how much profit can be expected from a new order.

To address this problem, the project uses historical Superstore data to build two Machine Learning models:
- A classification model to estimate the probability that an order will result in a loss.
- A regression model to estimate the expected profit of an order.

These models are integrated into an interactive Streamlit application that allows users to enter order details and receive an estimate of the order's sales, loss probability, and expected profit.

The overall goal is to help the business identify potentially unprofitable orders and better understand expected profitability before making decisions.


