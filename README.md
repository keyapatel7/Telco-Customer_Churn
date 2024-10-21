# Telco-Customer_Churn
## Project Overview
  This project aims to analyze customer churn in a telecom company using a publicly available dataset. The analysis explores various factors that influence customer churn, and the insights are used to build predictive models. Customer churn refers to when customers stop using a company's service. The dataset contains information about customer demographics, services they subscribe to, contract details, payment methods, tenure, and whether they churned.

## Dataset Information
- The dataset is a sample set containing Telco customer data, focusing on customers who left last month.
- The dataset includes 7,043 customer records and 21 features related to their services and demographics.
## Libraries Used
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN

## Steps in the Project

1. **Data Loading and Exploration**  
   Loaded the dataset using pandas. Displayed the first few rows to understand the structure of the data. Analyzed the dataset's shape, column names, and data types.

2. **Data Cleaning**  
   The column `TotalCharges` was converted to a numerical type (it was originally a string). Handled 11 missing values in the `TotalCharges` column by removing them, as they made up only 0.15% of the dataset.

3. **Feature Engineering**  
   Divided customers into tenure groups for easier analysis. Removed irrelevant columns such as `customerID` and `tenure` to streamline analysis.

4. **Target Variable Analysis**  
   Analyzed the distribution of the target variable (`Churn`), which was highly imbalanced with a 73:27 ratio of customers who did not churn to those who did. Converted the target variable `Churn` into binary values for model building, where 'Yes' = 1 and 'No' = 0.

5. **Data Visualization**  
   Used Seaborn and Matplotlib to create count plots to visualize the distribution of features by `Churn`. Created a missing value plot to confirm that there were no significant missing values in the dataset.

6. **Categorical Variable Encoding**  
   Converted all categorical variables into dummy variables for analysis and modeling.

7. **Statistical Summary**  
   Used `describe()` to summarize numerical features. Key insights:
   - 75% of customers have tenure less than 55 months.
   - The average monthly charge is USD 64.76, with 25% of customers paying more than USD 89.85 per month.
   - `SeniorCitizen` is a categorical variable but was encoded as an integer, so special handling was applied during analysis.

8. **Churn Distribution**  
   Visualized churn distribution to assess the imbalance and used the percentage of customers who churned to inform subsequent analyses.

9. **Univariate Analysis**  
   Visualized the distribution of individual predictors by `Churn` using Seaborn's `countplot()`.

10. **Data Preprocessing**  
   Applied the following steps for data preprocessing:
   - Splitting the dataset into features (`X`) and target (`Y`).
   - Dividing the data into training and testing sets using `train_test_split` with an 80-20 ratio.
   - To address the class imbalance, SMOTEENN was used to oversample the minority class and reduce noise in the data.

   ```python
   sm = SMOTEENN()
   X_resampled, y_resampled = sm.fit_sample(x, y)
   xr_train, xr_test, yr_train, yr_test = train_test_split(X_resampled, y_resampled, test_size=0.2)


## Model Building
### Built two models:

  - Decision Tree Classifier: Achieved an accuracy of 78.18% before balancing, and 93.44% after applying SMOTEENN.

Parameters:

DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=8)
Random Forest Classifier: Achieved an accuracy of 79.53% before balancing, and 94.27% after applying SMOTEENN.

Parameters:
RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)

## Performance Evaluation
### Before SMOTEENN (Random Forest Classifier)
Accuracy: 79.53%
Precision, Recall, F1-Score for class 1 (Churned): 0.69, 0.45, 0.55
### After SMOTEENN (Random Forest Classifier)
Accuracy: 94.27%
### Precision, Recall, F1-Score for class 1 (Churned): 0.94, 0.96, 0.95

print(metrics.classification_report(yr_test1, yr_predict1))
### Confusion Matrix after applying SMOTEENN:

[[478  40]
 [ 27 625]]
 
## Dimensionality Reduction with PCA
PCA was applied to reduce the number of features and retain 90% of the variance. However, applying PCA did not improve the performance. Accuracy after PCA was around 72.39%, which was lower than without PCA.

from sklearn.decomposition import PCA
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train1)
xr_test_pca = pca.transform(xr_test1)

## Saving the Model
The final model (Random Forest with SMOTEENN) was saved using the pickle library so it can be accessed later via APIs.
import pickle
filename = 'model.sav'
pickle.dump(model_rf_smote, open(filename, 'wb'))

## To load the model:
load_model = pickle.load(open(filename, 'rb'))

## Insights
Customers with month-to-month contracts, those without dependents, and those using paperless billing are more likely to churn.
Churn rate is higher among customers with fiber optic service as compared to DSL.
Senior citizens are more likely to churn than non-senior citizens.
