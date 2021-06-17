#!/usr/bin/env python
# coding: utf-8

# ### Data Description

# The details and description of the Boston Housing Data can be found here:
# 
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
# 
# and
# 
# https://towardsdatascience.com/things-you-didnt-know-about-the-boston-housing-dataset-2e87a6f960e8

# In[50]:


# import relevant libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# OPTIONAL:
# import pearson correlation library
from scipy.stats import pearsonr

# to show grid lines in plots
sns.set_style('whitegrid')

# to make all plots well positioned in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


# Import the Boston DataSet

from sklearn.datasets import load_boston


# In[52]:


# store the dataset

boston = load_boston()


# In[53]:


# check the different keys

boston.keys()


# In[54]:


# You can choose to print each of these keys to see their content

# kindly delete the '#' and run the code to see


## for data
# print(boston['data'])

## for target or predictors
# print(boston['target'])

## for feature_names OR predictor/column names
# print(boston['feature_names'])

## for DESCR: description of each feature/predictor
print(boston['DESCR'])

## for filename 
# (not relevant)


# ## Exploratory Data Analysis
# #### Let's explore the data a little !

# ##### First, let us put the data in a Dataframe

# In[55]:


# chect the current data type

type(boston['data'])


# In[56]:


# Convert the numpy array " boston['data'] " into a dataframe

bostonData_array = boston['data']

#boston_df = pd.DataFrame(bostonData_array, columns = ['CRIM','ZN','INDUS','CHAS', 'NOX', 'RM',
 #                                                     'AGE', 'DIS', 'RAD', 'TAX', ' PTRATIO', 'B', 'LSTAT'])

boston_df = pd.DataFrame(bostonData_array, columns = boston['feature_names'])

# add the RESPONSE Variable
boston_df['Med. Worth of Home'] = boston['target']

boston_df.head()


# ###### (1) Compare 
# the 'Average number of rooms per dwelling [RM]' (predictor) with the 'Median value of owner-occupied homes in $1000's [boston['target']] (the response vairable). 
# 
# ######  Does the correlation make sense?

# In[57]:


sns.jointplot(x = 'RM',
              y = 'Med. Worth of Home',
              data = boston_df ,
              color = 'k',
              stat_func = pearsonr)   # optional for the correlation value and p-value


# ###### YES, correlation makes sense.
# There exist is a positive correlation/relationship between  RM (Average number of rooms per dwelling) and boston['target'] (Median value of owner-occupied homes in $1000's). This could possibly infer that, higher average number of rooms per dwelling would result to a higher median value of the homes.

# In[ ]:





# ##### (2) Let us do a pair plot to see the relationship between "selected" predictors/columns and their correlation

# In[58]:


sns.pairplot(data = boston_df[
                                ['CRIM', 'RM', 'AGE', 'DIS', 'TAX', 'Med. Worth of Home']
                             ]
            )


# In[ ]:





# ### We can as well create a linear model plot

# In[59]:


sns.lmplot(x = 'RM', y = 'Med. Worth of Home', data = boston_df) 


# In[60]:


## It will be nice to represent this 'RM' and 'Med. Worth of Home' with a hex plot


# In[61]:


sns.jointplot(x = 'RM',
              y = 'Med. Worth of Home',
              data = boston_df ,
              kind = 'hex',          # also try 'scatter', 'reg', 'resid', 'kde' *it is optional
              color = 'k',
              stat_func = pearsonr)


# #### Interpretation:
#     ** There is a dense between 5.5 to 7.0 in 'RM' which corresponds to between 15.0 to 25.0 in the Median Worth of Home,
#     ** This implies that most owner-occupied homes in Boston have an average of 6 rooms per dwelling and the median worth of these homes is around between 150,000 to 250,000
#     ** We see that there is a strong correlation between the average number of rooms per dwelling and the Median Value or Worth of the Homes. Correlation is strong being 0.7.

# In[ ]:





# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# 
# ** Our variable X will equal the numerical features/columns which is boston['data'] ** 
# 
# ** Our variable y will equal the response variable which is boston['target'], i.e. Median value of owner-occupied homes in \$1000's **

# In[62]:


# predictors

X = boston['data']


# In[63]:


# response (predicands)

y = boston['target']


# In[64]:


# Split data into training and testing set

from sklearn.model_selection import train_test_split


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.3, # 30% of data should be used to test so # 70% to train
                                                    random_state = 101)


# #### Train the Model with train-data

# In[66]:


# import LinearRegression

from sklearn.linear_model import LinearRegression


# In[67]:


# create an object for the linear regression

lm = LinearRegression()


# In[68]:


# train and fit 'lm' on the training data

lm.fit(X = X_train,
       y = y_train)


# In[69]:


# Get the coefficients of the predictors

lm.coef_

print('Coefficients: ', lm.coef_)


# In[70]:


# Get the intercept of the predictors


lm.intercept_

# print('Intercept = %0.3f' % lm.intercept_)   # to 3 decimal place


# In[ ]:





# In[71]:


## The model is thus given by:


print("Our linear model is: "
      " 'Medain Value of Home (Y)' = {:.4} + {:.4}*CRIM + {:.4}*ZN + {:.4}*INDUS + {:.4}*CHAS + {:.4}*NOX + {:.4}*RM + "
                                           " {:.4}*AGE + {:.4}*DIS + {:.4}*RAD + {:.4}*TAX + {:.4}*PTRATIO + {:.4}*B + "
                                           " {:.4}*LSTAT ".format(
                                   lm.intercept_, 
                                lm.coef_[0], lm.coef_[1], lm.coef_[2], lm.coef_[3], lm.coef_[4], lm.coef_[5], 
                           lm.coef_[6], lm.coef_[7], lm.coef_[8], lm.coef_[9], lm.coef_[10], lm.coef_[11], lm.coef_[12]))


# In[ ]:





# ### Prediction of Model
# ** We evaluate its performance of our model by predicting the test values! **

# In[72]:


predictions = lm.predict(X = X_test)


# In[73]:


predictions


# ### Compare:
# **Now let's see how strong the relationship is, between our predictions and the real (original y-values)**

# In[74]:


# Using a scatter plot to check for correlation



# Using matplotlib

plt.scatter(x = y_test,
            y = predictions)


# In[75]:


# Using seaborn scatterplot

#sns.scatterplot(x = y_test,        # original or real values from data
#                y = predictions)   # predicted values




# Using Seaborn Jointplot (so we can call the correlation value)

sns.jointplot(x = y_test,
              y = predictions,
              kind = 'scatter',
              stat_func = pearsonr)


# ### Interpreation:
#     ** There is obviously a strong correlation between our prediction and the original values. Correlation value is 0.85,   hence our model is good enough to be used for predictions in real life.

# In[ ]:





# ## Evaluating the Model
# ** Let's evaluate our model performance by calculating the residual sum of squares and the variance score (R^2) **

# In[76]:


# import the library

from sklearn import metrics


# ##### Quick important note:
#    ** The RMSE is the square root of the variance of the residuals. It indicates the absolute fit of the model to the data–how close the observed data points are to the model’s predicted values. Whereas R-squared is a relative measure of fit, RMSE is an absolute measure of fit. As the square root of a variance, RMSE can be interpreted as the standard deviation of the unexplained variance, and has the useful property of being in the same units as the response variable. **Lower values of RMSE indicate better fit**. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction. (Source: https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/)**

# In[77]:


# for MAE:

mae = metrics.mean_absolute_error(y_true = y_test,
                                  y_pred = predictions)


# for MSE:

mse = metrics.mean_squared_error(y_true = y_test,
                                 y_pred = predictions)


# for RMSE

rmse = np.sqrt( metrics.mean_squared_error(y_true = y_test,
                                           y_pred = predictions) 
               )

print("MAE: ", mae )
print("MSE: ", mse )
print("RMSE:", rmse)


# In[78]:


# Since the RMSE is low, we can say that our model accurately predicts the reponse.


# In[ ]:





# ## Residuals:
#     ** Let's quickly explore the residuals to make sure everything was okay with our data. We do this by ploting the               histogram of the residuals to ensure it is normmally distributed **

# In[79]:


# Using Seaborn

sns.distplot(a = (y_test - predictions),   # this is how residual is calculated
             bins = 50)


# In[80]:


# YES: there is a good level of normality in our residuals. Hence, we finally accepts the model


# In[ ]:





# ## Further Thoughts:
#     ** We can tell how powerful each predictor/variable is in the model, usign the coeeficients of the predictors.
#     ** We can tell the significance of each predictor/variable in predicting the response variable, using their p-values 

# ##### Effect of predictors on model

# In[81]:


# Create a dataframe that will take the coefficeint and also the column names




# recall the column names
boston_df.columns


# In[82]:


# remove the last column (i.e. the predictor)

boston_df.drop(labels = 'Med. Worth of Home',   # name of the column to drop
               axis = 1,      # means column, axis = 0 means row 
               inplace = True)    # make the drop permanent

# now check the columns again
boston_df.columns


# In[83]:


# DataFrame

cdf = pd.DataFrame(data = lm.coef_,
                   index = boston_df.columns,
                   columns = ['Coefficient'])

cdf


# In[84]:


# Let's sort the values of by the Coefficient column


cdf.sort_values(by = ['Coefficient'],
                ascending = False)


# ### Interpretation:
#     ** We see the effect of each predictor (based on their coefficients) in the above table in descendeing order
#     ** Clearly, 'CHAS'':Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)' and 'RM: ''average number of rooms per dwelling', happens to give the most positive (increasing) effect on the model
#     ** And 'NOX: '' nitric oxides concentration (parts per 10 million)' gives the most negative (decreasing) effect on the model.

# In[ ]:




