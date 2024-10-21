# Telco-Customer_Churn
Project Overview
This project aims to analyze customer churn in a telecom company using a publicly available dataset. The analysis explores various factors that influence customer churn, and the insights are used to build predictive models. Customer churn refers to when customers stop using a company's service. The dataset contains information about customer demographics, services they subscribe to, contract details, payment methods, tenure, and whether they churned.

Dataset Information
The dataset is a sample set containing Telco customer data, focusing on customers who left last month.
The dataset includes 7,043 customer records and 21 features related to their services and demographics.
Libraries Used
python
Copy code
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
%matplotlib inline
Steps in the Project
1. Data Loading and Exploration
Loaded the dataset using pandas.
Displayed the first few rows to understand the structure of the data.
Analyzed the dataset's shape, column names, and data types.
2. Data Cleaning
The column TotalCharges was converted to a numerical type (it was originally a string).
Handled 11 missing values in the TotalCharges column by removing them, as they made up only 0.15% of the dataset.
3. Feature Engineering
Divided customers into tenure groups for easier analysis.
Removed irrelevant columns such as customerID and tenure to streamline analysis.
4. Target Variable Analysis
Analyzed the distribution of the target variable (Churn), which was highly imbalanced with a 73:27 ratio of customers who did not churn to those who did.
Converted the target variable Churn into binary values for model building, where 'Yes' = 1 and 'No' = 0.
5. Data Visualization
Used seaborn and matplotlib to create count plots to visualize the distribution of features by Churn.
Created a missing value plot to confirm that there were no significant missing values in the dataset.
6. Categorical Variable Encoding
Converted all categorical variables into dummy variables for analysis and modeling.
7. Statistical Summary
Used describe() to summarize numerical features.
Key insights:
75% of customers have tenure less than 55 months.
The average monthly charge is USD 64.76, with 25% of customers paying more than USD 89.85 per month.
SeniorCitizen is a categorical variable but was encoded as an integer, so special handling was applied during analysis.
8. Churn Distribution
Visualized churn distribution to assess the imbalance and used the percentage of customers who churned to inform subsequent analyses.
9. Univariate Analysis
Visualized the distribution of individual predictors by churn using seaborn's countplot().
File Structure
Telco-Churn-EDA.ipynb: Jupyter Notebook containing the exploratory data analysis (EDA).
WA_Fn-UseC_-Telco-Customer-Churn.csv: Dataset file.
README.md: This file, providing an overview of the project.
How to Use
Install the required Python libraries:
bash
Copy code
pip install numpy pandas seaborn matplotlib
Load the dataset and follow along with the analysis in the Jupyter Notebook.
Insights
Customers with month-to-month contracts, those without dependents, and those using paperless billing are more likely to churn.
Churn rate is higher among customers with fiber optic service as compared to DSL.
Senior citizens are more likely to churn than non-senior citizens.
