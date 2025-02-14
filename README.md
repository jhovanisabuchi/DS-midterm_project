# Data Science Midterm Project

## Project/Goals
The goal of this project is to develop a machine learning model to predict housing prices based on various features of the house. By leveraging deep learning, random forest, and decision tree models, the project aims to:

* Accurately estimate house prices based on key attributes such as number of baths, stories, lot size, garage size, and location.
* Compare different machine learning models to determine the most effective approach for price prediction.
* Handle missing data, perform feature selection, and optimize model performance.
* Provide insights into the most influential factors affecting house prices.
* Develop a robust predictive system that can assist buyers, sellers, and real estate professionals in making informed decision
## Process
### 1. Data Gathering and Preprocessing
* Data Extraction:

Converted the raw JSON data into a structured DataFrame by iterating through nested fields.
Extracted key attributes related to house features, location, and tags.
Merged all separate data sources into one unified dataset.
* Data Cleaning:

Dropped unnecessary features that were irrelevant or highly correlated.
Removed rows with excessive missing values (above 80%).
Handled missing values using:
Mean/Median for numerical data.
Mode or forward/backward fill for categorical/time-based data.
Checked for duplicates and inconsistencies in the dataset.
* Feature Engineering & Preprocessing:

Encoded categorical features using One-Hot Encoding (OHE).
Scaled numerical features using StandardScaler or MinMaxScaler.
Checked feature correlations and removed low-impact features.
Split the data into 80% training and 20% testing sets for model evaluation.

### 2. Model Selection and Training
* Tried different models to find the best fit for housing price prediction:

Linear Regression (Baseline model).
Decision Tree (Captures non-linear relationships).
Random Forest (Handles overfitting and improves accuracy).
Deep Learning Model (For capturing complex feature interactions).
* Trained each model on the training data and evaluated them on the test data.

* Initial evaluation metrics:

  - MAE (Mean Absolute Error) – Measures absolute error.
  - MSE (Mean Squared Error) – Penalizes large errors.
  - R² Score – Measures how well the model explains variance in the target variable.
* Compared models and selected the best-performing one for fine-tuning.
### 3. Model Tuning and Optimization
* Hyperparameter Tuning:

Used GridSearchCV and RandomizedSearchCV to find optimal hyperparameters.
Adjusted parameters like tree depth, number of estimators (for Random Forest), learning rate (for deep learning models), and regularization terms.
* Feature Selection:

Identified and removed irrelevant features to improve accuracy.
Used feature importance scores from models like Random Forest to select key features.
* Final Model Evaluation and Visualization:

Plotted predicted vs. actual house prices to analyze model accuracy.
Boxplots and scatter plots to detect outliers affecting predictions.
Residual plots to assess whether errors were evenly distributed.
Correlation heatmaps to understand relationships between variables.
* Final Decision:

Selected the best-performing model based on real-world validation and interpretability.

## Results
(fill in how your model performed)

## Challenges 
(discuss challenges you faced in the project)

## Future Goals
(what would you do if you had more time?)
