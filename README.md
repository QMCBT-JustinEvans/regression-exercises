# How do we stop the Churn Burn?





# Goal: 

* Discover driving features affecting churn
* Use drivers to develop a machine learning model to predict churn
* Use these predictions to inform preemptive decisions aimed at alleviating future churn



# Acquire

* telco_churn data from Codeup SQL database was used for this project.
* The data was initially pulled on 26-OCT.
* The initial DataFrame contained 7043 records with 44 features  
    (44 columns and 7043 rows) before cleaning & preparation.
* Each row represents a customer record both current & historical.
* Each column represents a feature provided by telco or an informational element about the customer.



# Prepare

**Prepare Actions**
* **DROP**: Removed 4 index_id, 18 duplicate, and 1 corrupted data column
* **RENAME**: Initially did not need to Rename any original columns
* **REFORMAT**: 2 columns contained inappropriate data types that needed to be reformatted 
* **REPLACE**: 7 columns had a third value that could be determined by another feature, replaced third value in each column with appropriate yes/no value. 1 column had empty non-null values that were replaced with 0
* **ENCODED**: 14 categorical columns from variables to boolean numeric values
* **MELT**: No melts needed
* **PIVOT**: 3 columns with more than two variables were pivotted
* **FEATURE ENGINEER:**: No new features were added
* **DROP2**: 16 Columns duplicated by Encoded and Pivot Columns were dropped
* **RENAME2**: 13 encoded columns were renamed after original columns were dropped

**NaN/Null**: Only one column contained NaN/nulls in the data (it was in the corrupted field that was removed).
**OUTLIERS**: No outliers have been removed or altered</div>
**IMPUTE**: No data was imputed</div>



# Split

* **SPLIT**: train, validate and test (approx. 60/20/20), stratifying on target of 'churn'
* **SCALED**: no scaling was conducted



## A Summary of the data

Most features with 0min and 1max, the mean will represent the percentage of True values

Print nunique of all Columns shows a count of True and False for each feature, giving a quick glance at variance between feature values and allowing a quick infference into approximate percentages.



# Explore

* Each of the three features were tested for relationship or difference against Churn.
    1. Tenure
    2. Monthly Charges
    3. Tech Support 

* All three comparison features showed a significant relationship with the target feature Churn.

* There were four feature specific questions asked across three features all compared against our Target Feature of Churn.
    * 1.1 Is the average Tenure of Active customers greater than the average Tenure of Churned customers?
    * 2.1 Are the average monthly charges of customers that Churn higher than the average monthly charges of Active customers?
    * 3.1 Is the average of customer Churn without Tech Support greater than the average of customer Churn with Tech Support?
    * 3.2 Is the average of customer Churn without Tech Support greater than the average of Active customers without tech support?

* Three statistical tests were used to test these questions.
    1. T-Test
    2. Pearson's R
    3. $Chi^2$

* The first two questions 1.1 and 2.1 did not test positively against our stated question.

* The remaining two questions 3.1 and 3.2 involving Tech Support both tested positively against our stated question.



# Exploration Summary

30% of all customers without tech support churn
82% of all churn is attributed to customer that do NOT have tech support
Only 17% of customers with tech support churn
Only 18% of all churn can be attributed to customers with tech support



# Features I am moving to modeling With

* Churn is incredibly important as our target feature



# Features I'm not moving to modeling with

* Tenure 
* Monthly Charges
* Tech Support



# Modeling

* Accuracy is our evaluation metric  
* Our Target feature Churn, splits the data 27% Churn, 73% Active 

* Simply guessing Active for every customer, we could achieve an accuracy of 73%
* Therefore 73% will be the baseline accuracy used for this project

* Models will be developed and evaluated using three different model types and various hyperparameter configurations 
    * Decision Tree
    * Random Forest
    * KNN

* Models will be evaluated on train and validate data
* The model that performs the best will ultimately be the one and only model evaluated on our test data 



## Comparing Models

* Decision Tree, Random Forest, and KNN models all performed above the Baseline of 73%

* The KNN model performed slightly better on train data than it did on the validate data which may be a sign of overfit.

* Because the results of the Decision Tree, Random Forest, and KNN models were all very similar and above Baseline, we could proceed to test with any of these models.

* ```Random Forest``` however, is the best model that retained high performance across both train and validate data and will likely perform well above Baseline on the Test data.



# Conclusions



## Reccomendations

* Consider implementing incentives for increased Tech Support



## Next Steps

* Decision Tree focused on other driving features above Tech Support
    * Investigate further into these features
    * Try running models with less features to isolate cause of predictions