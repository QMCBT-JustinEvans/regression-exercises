# Regression Exercises





# Goal: 

* Learn how to apply Regression Models while practicing the repetitve skills of:
    * Acquire
    * Prepare
    * Split 
    * Explore
    * Feature Engineer
    * Hypothesis Creation
    * Hypothesis Testing
    * Feature Selection
    * Model Creation
    * Model Evaluation
    * Model Selection
    * Model Test
    * Summary & Conclusion of Analaysis


# Acquisition and Preperation Lesson
* Create ```regression-exercises``` repository
    * Create this ```README.md``` file
    * Create ```wrangle.ipynb``` Jupyter Notebook to show work
    * Create ```wrangle.py``` file to run custom Functions
    * Create function ```wrangle_zillow``` to perform all Acquire and Prepare tasks

# Scaling Numeric Data Lesson
* Create ```scaling.ipynb``` Jupyter Notebook to show work
* Create a Function in your ```prepare.py``` to scale the zillow DataFrame
    * I wrote the function ```scale_data``` into my ```QMCBT_wrangle.py``` file
        * ```scale_data``` takes in arguments (train, test, validate, columns_to_scale, scaler, return_scaler=False)
        * train = Assign the train DataFrame
        * validate = Assign the validate DataFrame 
        * test = Assign the test DataFrame
        * columns_to_scale = Assign the Columns that you want to scale
        * scaler = Assign the scaler to use MinMaxScaler(),
                                            StandardScaler(), 
                                            RobustScaler(), or 
                                            QuantileTransformer()
        * return_scaler = False by default and will not return scaler data
                          True will return the scaler data before displaying the _scaled data

# Exploration Lesson
* Create ```explore.ipynb``` Jupyter Notebook to show work
* Create ```explore.py``` file to hold custom functions
    * I wrote my functions into a combined ```QMCBT_explore_evaluate.py``` file
    * Create a function named ```plot_variable_pairs``` that accepts a dataframe as input and plots all of the pairwise relationships along with the regression line for each pair.
    * Create a function named ```plot_categorical_and_continuous_vars``` that accepts your dataframe and the name of the columns that hold the continuous and categorical features and outputs 3 different plots for visualizing a categorical variable and a continuous variable
    
# Evaluating Regression Models Lesson
* Create ```evaluate.ipynb``` Jupyter Notebook to show work
* Create ```evaluate.py``` file to hold custom functions
    * I wrote my functions into a combined ```QMCBT_explore_evaluate.py``` file
    * Create the following Fuctions:
        * ```plot_residuals(y, yhat)```: creates a residual plot
        * ```regression_errors(y, yhat)```: returns the following values:
            sum of squared errors (SSE)
            explained sum of squares (ESS)
            total sum of squares (TSS)
            mean squared error (MSE)
            root mean squared error (RMSE)
        * ```baseline_mean_errors(y)```: computes the SSE, MSE, and RMSE for the baseline model
        * ```better_than_baseline(y, yhat)```: returns true if your model performs better than the baseline, otherwise false
        
# Feature Engineering Lesson
* Create ```feature_engineering.ipynb``` Jupyter Notebook to show work

# Modeling Lesson
* Create ```modeling.ipynb``` Jupyter Notebook to show work

