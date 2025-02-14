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
# 1. Random Forest Regressor
After training and testing the Random Forest Regressor, we obtained the following performance metrics:

Mean Absolute Error (MAE): 106,864.39
Mean Squared Error (MSE): 70,341,344,252.25
R-Squared (R²): 0.7953
These results indicate that the model captures some patterns in the data.
![alt text](image-1.png)

# 2. Decision Tree Regressor
We initially trained the Decision Tree Regressor, and the results were:

Mean Absolute Error (MAE): 4,222.46
Mean Squared Error (MSE): 1,479,414,586.36
R-Squared (R²): 0.9957
This model showed strong signs of overfitting, meaning it performed well on the training data but might not generalize well.

After Hyperparameter Tuning
To reduce overfitting, we adjusted parameters such as max_depth and other hyperparameters. The revised model results are:

Mean Absolute Error (MAE): 145,417.03
Mean Squared Error (MSE): 51,779,538,032.65
R-Squared (R²): 0.8493
The performance is now more balanced, showing improved generalization at the cost of a slight increase in error.

## Challenges 
# 1. Inconsistent JSON Data Structure
The raw data was stored in JSON format, but its structure was inconsistent. Retrieving and converting it into a structured DataFrame required extensive iteration through nested files, making data extraction more complex than expected.

# 2. Handling Missing and Irrelevant Data
The dataset contained many irrelevant features, which had to be identified and removed.
A significant portion of the dataset had missing values. While some were filled using statistical methods (mean, median, mode, forward-fill, and backward-fill), others—especially location-related missing values—could not be imputed, making it difficult to utilize location as a key predictive feature.
Cleaning and preprocessing the data consumed most of the project time.
# 3. Small Dataset & Overfitting Issues
The dataset was relatively small, making it difficult for models to generalize well.
Initial Decision Tree models overfitted, meaning they performed exceptionally well on training data but failed to generalize on test data. Hyperparameter tuning was necessary to address this issue.
Deep Learning models were considered but performed poorly due to the limited dataset size, as deep learning typically requires large amounts of data to train effectively.
# 4. Difficulty in Predicting House Prices
House pricing is influenced by various external factors (e.g., market trends, economic conditions, neighborhood desirability) that were not captured in the dataset.
The limited amount of training data and missing location data restricted the model's ability to make highly accurate predictions.

## Future Goals
# 1. Collecting a Larger and More Diverse Dataset
The accuracy of house price prediction models heavily depends on the amount and diversity of data.
Future work should focus on acquiring a more extensive dataset that includes more properties across different locations and price ranges.
Integrating real estate APIs (e.g., Zillow, Redfin) can help obtain real-time data for better predictions.
# 2. Enhancing Feature Engineering
Incorporate external factors such as:
Economic indicators (interest rates, inflation)
Nearby amenities (schools, hospitals, shopping centers)
Crime rates and neighborhood safety
Perform geospatial analysis using latitude and longitude to enhance location-based predictions.
# 3. Handling Missing Data More Effectively
Instead of dropping missing values, advanced imputation techniques (e.g., KNN Imputer, regression-based imputation) can be used for better estimation.
For missing location data, reverse geocoding techniques can be applied to approximate missing geographical details.
# 4. Addressing Overfitting Issues
Implement cross-validation techniques (such as k-fold cross-validation) to ensure the model generalizes well.
Use ensemble methods like Gradient Boosting (XGBoost, LightGBM, CatBoost) to balance performance and prevent overfitting.
# 5. Exploring More Advanced Models
Deep learning was initially considered but didn’t perform as needed due to the small dataset.
Future improvements can include Neural Networks with Transfer Learning or Hybrid Models that combine deep learning with traditional methods.
Bayesian optimization or AutoML frameworks can be explored to automate hyperparameter tuning for better performance.

