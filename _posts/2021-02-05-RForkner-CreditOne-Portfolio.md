---
title: "Project 1 Post"
date: 2021-02-05
tags: [data science, messy data, data analytics]
header:
image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Messy Data, Data Analytics"
mathjax: "true"
---

## Overview
### Over the past year or so Credit One has seen an increase in the number of customers who have defaulted on loans they have secured from various partners, and Credit One, as their credit scoring service, could risk losing business if the problem is not solved right away. The bottom line is they need a much better way to understand how much credit to allow someone to use or, at the very least, if someone should be approved or not.

# Project Goals
### Using the demographic data supplied by the client, determine:
	1. How much credit should customers be allowed
	2. Should a potential customer be approved for credit?

### Step 1. Install Modules and Libraries


```python
#Import SQL querying modules and Pandas into your notebook
from sqlalchemy import create_engine
import pymysql
import pandas as pd
```


```python
#DS Basics
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import math
from math import sqrt
import seaborn as sns

#Pandas Profiling
import pandas_profiling

#SKLearn Stuff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import linear_model

#model metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import metrics


```

### Step 2. Download dataset with SQL connection


```python
#create a connection to MySQL database
db_connection_str = 'mysql+pymysql://deepanalytics:Sqltask1234!@34.73.222.197/deepanalytics'
```


```python
#connect to the MySQL
db_connection = create_engine(db_connection_str)
```


```python
#use the following SELECT statement and query the Credit One data to extract it into a Pandas dataframe
df = pd.read_sql('SELECT * FROM credit', con=db_connection)
```


```python
#save data to CSV
df.to_csv('credit_one.csv',index=False)
```

### Known issues with data and strategy for addressing concerns:
* Issues
1. Not all data are numerical, and even numerical columns read as objects
2. Repeated rows
3. Repeated headers
4. Classification analyses will require discretization

* Fixes
1. Remove all non-numeric rows.  This was done manually after identifying them in excel as an extra set of headers.  However, even after their removal, in order for Jupyter to read the column numerical, the dataset needs to be saved as a csv file and re-imported.  Otherwise conversions must be coded manually per column/variable
2. Delete repeated rows
3. Delete extra headers
4. Discretize variables as required.


```python
#rename csv file as df and skip row 1, which are labels in this case
df=pd.read_csv('credit_one.csv',skiprows=[0])
```


```python
#drop duplicate rows
df = df.drop_duplicates()
```


```python
#check for missing values
print(df.isnull().sum())
```

    ID                            1
    LIMIT_BAL                     0
    SEX                           0
    EDUCATION                     0
    MARRIAGE                      0
    AGE                           0
    PAY_0                         0
    PAY_2                         0
    PAY_3                         0
    PAY_4                         0
    PAY_5                         0
    PAY_6                         0
    BILL_AMT1                     0
    BILL_AMT2                     0
    BILL_AMT3                     0
    BILL_AMT4                     0
    BILL_AMT5                     0
    BILL_AMT6                     0
    PAY_AMT1                      0
    PAY_AMT2                      0
    PAY_AMT3                      0
    PAY_AMT4                      0
    PAY_AMT5                      0
    PAY_AMT6                      0
    default payment next month    0
    dtype: int64



```python
#rename dataframe to credit
credit = df
```


```python
credit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>default</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000</td>
      <td>female</td>
      <td>university</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000</td>
      <td>male</td>
      <td>university</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>not default</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



#### The data include the following information:
* LIMIT_BAL: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
* SEX: male / female
* EDUCATION: graduate school; university; high school; others).
* MARRIAGE: Marital status (1 = married; 2 = single; 3 = divorced; 0=others).
* AGE: Age in years
* PAY_: History of past payment. We tracked the past monthly payment records (from April to September) as follows: PAY_0 = the repayment status in September; PAY_1 = the repayment status in August; . . .; PAY_6 = the repayment status in April. *The measurement scale for the repayment status is: -2: No consumption; -1: Paid in full; 0: The use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.*
* BILL_AMT: Amount of bill statement (NT dollar). BILL_AMT1 = amount of bill statement in September; BILL_AMT2 = amount of bill statement in August; . . .; BILL_AMT6 = amount of bill statement in April.
* PAY_AMT: Amount of previous payment (NT dollar). PAY_AMT1 = amount paid in September; PAY_AMT2 = amount paid in August; . . .; PAY_AMT6 = amount paid in April.
* Default Payment Next Month: If the customer defaulted in October


```python
credit.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30001</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>...</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
      <td>30002</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>30001</td>
      <td>83</td>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>58</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>...</td>
      <td>21550</td>
      <td>21012</td>
      <td>20606</td>
      <td>7945</td>
      <td>7901</td>
      <td>7520</td>
      <td>6939</td>
      <td>6899</td>
      <td>6941</td>
      <td>4</td>
    </tr>
    <tr>
      <th>top</th>
      <td>1195</td>
      <td>50000</td>
      <td>female</td>
      <td>university</td>
      <td>2</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>not default</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>3365</td>
      <td>18112</td>
      <td>14030</td>
      <td>15964</td>
      <td>1605</td>
      <td>14737</td>
      <td>15730</td>
      <td>15764</td>
      <td>16455</td>
      <td>...</td>
      <td>3195</td>
      <td>3506</td>
      <td>4020</td>
      <td>5249</td>
      <td>5396</td>
      <td>5968</td>
      <td>6408</td>
      <td>6703</td>
      <td>7173</td>
      <td>23364</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 25 columns</p>
</div>




```python
credit.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 30002 entries, 0 to 30202
    Data columns (total 25 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
     0   ID                          30001 non-null  object
     1   LIMIT_BAL                   30002 non-null  object
     2   SEX                         30002 non-null  object
     3   EDUCATION                   30002 non-null  object
     4   MARRIAGE                    30002 non-null  object
     5   AGE                         30002 non-null  object
     6   PAY_0                       30002 non-null  object
     7   PAY_2                       30002 non-null  object
     8   PAY_3                       30002 non-null  object
     9   PAY_4                       30002 non-null  object
     10  PAY_5                       30002 non-null  object
     11  PAY_6                       30002 non-null  object
     12  BILL_AMT1                   30002 non-null  object
     13  BILL_AMT2                   30002 non-null  object
     14  BILL_AMT3                   30002 non-null  object
     15  BILL_AMT4                   30002 non-null  object
     16  BILL_AMT5                   30002 non-null  object
     17  BILL_AMT6                   30002 non-null  object
     18  PAY_AMT1                    30002 non-null  object
     19  PAY_AMT2                    30002 non-null  object
     20  PAY_AMT3                    30002 non-null  object
     21  PAY_AMT4                    30002 non-null  object
     22  PAY_AMT5                    30002 non-null  object
     23  PAY_AMT6                    30002 non-null  object
     24  default payment next month  30002 non-null  object
    dtypes: object(25)
    memory usage: 6.0+ MB



```python
#display row of choice, in this case to confirm the rows that containt he duplicate headers so that we can be sure we'll be deleting the right rows.
display(credit.loc[201])
```


    ID                            NaN
    LIMIT_BAL                      X1
    SEX                            X2
    EDUCATION                      X3
    MARRIAGE                       X4
    AGE                            X5
    PAY_0                          X6
    PAY_2                          X7
    PAY_3                          X8
    PAY_4                          X9
    PAY_5                         X10
    PAY_6                         X11
    BILL_AMT1                     X12
    BILL_AMT2                     X13
    BILL_AMT3                     X14
    BILL_AMT4                     X15
    BILL_AMT5                     X16
    BILL_AMT6                     X17
    PAY_AMT1                      X18
    PAY_AMT2                      X19
    PAY_AMT3                      X20
    PAY_AMT4                      X21
    PAY_AMT5                      X22
    PAY_AMT6                      X23
    default payment next month      Y
    Name: 201, dtype: object



```python
#Having looked at the data in Excel I know that the first 201 rows are duplicated and the headers are then repeated.  I removed the duplicate rows above, but now need to find the duplicate headers and delete them manually.
credit.drop([201],inplace=True)
credit.drop([202],inplace=True)
```

### Now that the data are clean they are exported as a new .csv file and re-imported so that they cleaned columns are recognized as numeric.  Non-numeric data will be label encoded to enable thei inclusion in the analyses.


```python
#export cleaned df called 'credit' to CSV called 'Credit_Clean'
credit.to_csv (r'C:\Users\rob\Data Analytics Course\Project 2\Credit_Clean.csv', index = False, header=True)
```


```python
#import clean csv
data = pd.read_csv('Credit_Clean.csv')
Credit_Clean=data
```


```python
#check for missing values
print(Credit_Clean.isnull().sum())
```

    ID                            0
    LIMIT_BAL                     0
    SEX                           0
    EDUCATION                     0
    MARRIAGE                      0
    AGE                           0
    PAY_0                         0
    PAY_2                         0
    PAY_3                         0
    PAY_4                         0
    PAY_5                         0
    PAY_6                         0
    BILL_AMT1                     0
    BILL_AMT2                     0
    BILL_AMT3                     0
    BILL_AMT4                     0
    BILL_AMT5                     0
    BILL_AMT6                     0
    PAY_AMT1                      0
    PAY_AMT2                      0
    PAY_AMT3                      0
    PAY_AMT4                      0
    PAY_AMT5                      0
    PAY_AMT6                      0
    default payment next month    0
    dtype: int64



```python
Credit_Clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 25 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
     0   ID                          30000 non-null  int64
     1   LIMIT_BAL                   30000 non-null  int64
     2   SEX                         30000 non-null  object
     3   EDUCATION                   30000 non-null  object
     4   MARRIAGE                    30000 non-null  int64
     5   AGE                         30000 non-null  int64
     6   PAY_0                       30000 non-null  int64
     7   PAY_2                       30000 non-null  int64
     8   PAY_3                       30000 non-null  int64
     9   PAY_4                       30000 non-null  int64
     10  PAY_5                       30000 non-null  int64
     11  PAY_6                       30000 non-null  int64
     12  BILL_AMT1                   30000 non-null  int64
     13  BILL_AMT2                   30000 non-null  int64
     14  BILL_AMT3                   30000 non-null  int64
     15  BILL_AMT4                   30000 non-null  int64
     16  BILL_AMT5                   30000 non-null  int64
     17  BILL_AMT6                   30000 non-null  int64
     18  PAY_AMT1                    30000 non-null  int64
     19  PAY_AMT2                    30000 non-null  int64
     20  PAY_AMT3                    30000 non-null  int64
     21  PAY_AMT4                    30000 non-null  int64
     22  PAY_AMT5                    30000 non-null  int64
     23  PAY_AMT6                    30000 non-null  int64
     24  default payment next month  30000 non-null  object
    dtypes: int64(22), object(3)
    memory usage: 5.7+ MB


### As you can see, not all of the data are numeric.  We'll address this by encoding the data listed as objects into numeric values:


```python
#Label Encode Sex
le=LabelEncoder()
le.fit(Credit_Clean['SEX'])
Credit_Clean['SEX']=le.transform(Credit_Clean['SEX'])
```


```python
#Label Encode Default
le=LabelEncoder()
le.fit(Credit_Clean['default payment next month'])
Credit_Clean['default payment next month']=le.transform(Credit_Clean['default payment next month'])
```


```python
#Label Encode Default
le=LabelEncoder()
le.fit(Credit_Clean['EDUCATION'])
Credit_Clean['EDUCATION']=le.transform(Credit_Clean['EDUCATION'])
```


```python
#convert non-numeric columns to a series of binary numeric 'Dummy' columns if encoding will produce more than 2 values
#Credit_Clean = pd.get_dummies(Credit_Clean)
```


```python
#check data types are all numeric after encoding
Credit_Clean.dtypes
```




    ID                            int64
    LIMIT_BAL                     int64
    SEX                           int32
    EDUCATION                     int32
    MARRIAGE                      int64
    AGE                           int64
    PAY_0                         int64
    PAY_2                         int64
    PAY_3                         int64
    PAY_4                         int64
    PAY_5                         int64
    PAY_6                         int64
    BILL_AMT1                     int64
    BILL_AMT2                     int64
    BILL_AMT3                     int64
    BILL_AMT4                     int64
    BILL_AMT5                     int64
    BILL_AMT6                     int64
    PAY_AMT1                      int64
    PAY_AMT2                      int64
    PAY_AMT3                      int64
    PAY_AMT4                      int64
    PAY_AMT5                      int64
    PAY_AMT6                      int64
    default payment next month    int32
    dtype: object



### Now that the data are workable, we'll do a little EDA.  Since we're wondering how much credit to give customers, we'll look at LIMIT_BAL as the dependent variable


```python
#Data Visualization, Plot Histogram with bins
plt.hist(Credit_Clean['LIMIT_BAL'], bins=4)
plt.show()
```


![png](RForkner_Credit_One_Portfolio_files/RForkner_Credit_One_Portfolio_33_0.png)



```python
#Data Visualization, Plot Histogram with bins
plt.hist(Credit_Clean['LIMIT_BAL'], bins=10)
plt.show()
```


![png](RForkner_Credit_One_Portfolio_files/RForkner_Credit_One_Portfolio_34_0.png)



```python
#Basic Correlation Matrix
#corrMat = Credit_Clean.corr()
#print(corrMat)
```

#### Looking at a quick heatmap of correlation between variables it's apparent that there isn't a direct relationship between the credit limit granted customers (LIMIT_BAL) and any other variable.  However, it is apparent that whether or not a customer will default does correlate with their payment history (green box).
![Heatmap%20Corr.png](attachment:Heatmap%20Corr.png)


```python
#Pandas Profiling
#pandas_profiling.ProfileReport(Credit_Clean)
```

### In order to set up the analysis, we'll re-arrange the variables in the dataset so that the variable we're modeling for, the credit limit (LIMIT_BAL) is last in the dataframe.  We'll then designate which data are used to the model the resulting credit limit.


```python
#re-order columns to put desired dependent variable last
column_names = ["ID", "SEX", "MARRIAGE", "AGE","EDUCATION","default payment next month","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6", "LIMIT_BAL"]
Credit_Clean = Credit_Clean.reindex(columns=column_names)
#print(Credit_Clean)
```


```python
#features, setting independent variables
X = Credit_Clean.iloc[:,1:23]
print('Summary of feature sample')
X.head()
```

    Summary of feature sample





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEX</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>EDUCATION</th>
      <th>default payment next month</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>...</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>...</td>
      <td>3102</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>26</td>
      <td>3</td>
      <td>0</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1725</td>
      <td>2682</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>34</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14027</td>
      <td>13559</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>37</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>48233</td>
      <td>49291</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>57</td>
      <td>3</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5670</td>
      <td>35835</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
#Setting Dependent Variable
y = Credit_Clean['LIMIT_BAL']
y.head()
```




    0     20000
    1    120000
    2     90000
    3     50000
    4     50000
    Name: LIMIT_BAL, dtype: int64



### To build and assess both models recursively, we create an empty list to store the results and another to hold the name of each algorithm so we can easily print out the results and keep them separated.  In this case, we'll use Random Forest Regression, Linear Regression, and Support Vector Regression to examine the relationship between the data and predicting customer credit limit.


```python
#Prepare regression algorithms
algosClass = []
```


```python
#To build and assess both models recursively, we create an empty list to store the results and another to hold the name of each algorithm so we can easily print out the results and keep them separated
algosClass.append(('Random Forest Regressor',RandomForestRegressor()))
```


```python
#To build and assess both models recursively, we create an empty list to store the results and another to hold the name of each algorithm so we can easily print out the results and keep them separated
algosClass.append(('Linear Regression',LinearRegression()))
```


```python
#To build and assess both models recursively, we create an empty list to store the results and another to hold the name of each algorithm so we can easily print out the results and keep them separated
algosClass.append(('Support Vector Regression',SVR()))
```


```python
#regression
results = []
names = []
for name, model in algosClass:
    result = cross_val_score(model, X,y, cv=3, scoring='r2')
    names.append(name)
    results.append(result)
```


```python
#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
```

### Random Forest Regression
#### A method composed of multiple interations of random sampling through a dataset to arrive at the outcome.  The model then aggregates the results at the end of the anaysis to compare to ground truth.


```python
#Regression Modeling
algo = RandomForestRegressor()
model = algo.fit(X_train,y_train)
```


```python
#Predictions
preds = model.predict(X_test)
```


```python
#R squared value
print(r2_score(y_test, preds))
```

    0.46465528438750847



```python
#RMSE root mean square error can't be calculated in one go, so we calculate mse then take the square root
mse = mean_squared_error(y_test, preds)
```


```python
rmse = math.sqrt(mse)
```


```python
print((rmse))
```

    94050.66857693906



```python
#Make Predictions
predictions = model.predict(X_test)
predRsquared = r2_score(y_test,predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))
print('R Squared: %.3f' % predRsquared)
print('RMSE: %.3f' % rmse)
```

    R Squared: 0.465
    RMSE: 94050.669



```python
#plotting results
plt.scatter(y_test, predictions, color=['blue'], alpha = 0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();
```


![png](RForkner_Credit_One_Portfolio_files/RForkner_Credit_One_Portfolio_57_0.png)


### Linear Regression
#### Attempting to model the relationship between modeled and ground truth results by fitting a linear equation to the data.


```python
#Regression Modeling
algo = LinearRegression()
model = algo.fit(X_train,y_train)
```


```python
#Predictions
preds = model.predict(X_test)
```


```python
#R squared value
print(r2_score(y_test, preds))
```

    0.3361562778442996



```python
#RMSE root mean square error can't be calculated in one go, so we calculate mse then take the square root
mse = mean_squared_error(y_test, preds)
```


```python
rmse = math.sqrt(mse)
```


```python
print((rmse))
```

    104731.67596963384



```python
#Make Predictions
predictions = model.predict(X_test)
predRsquared = r2_score(y_test,predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))
print('R Squared: %.3f' % predRsquared)
print('RMSE: %.3f' % rmse)
```

    R Squared: 0.336
    RMSE: 104731.676



```python
#plotting results
plt.scatter(y_test, predictions, color=['green'], alpha = 0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();
```


![png](RForkner_Credit_One_Portfolio_files/RForkner_Credit_One_Portfolio_66_0.png)


### Support Vector Regression
#### Similar to a linear regression except allowing for a wider boundary of error around the regression line.


```python
#Regression Modeling
algo = SVR()
model = algo.fit(X_train,y_train)
```


```python
#Predictions
preds = model.predict(X_test)
```


```python
#R squared value
print(r2_score(y_test, preds))
```

    -0.03583003584384947



```python
#RMSE root mean square error can't be calculated in one go, so we calculate mse then take the square root
mse = mean_squared_error(y_test, preds)
```


```python
rmse = math.sqrt(mse)
```


```python
print((rmse))
```

    130824.5884282775



```python
#Make Predictions
predictions = model.predict(X_test)
predRsquared = r2_score(y_test,predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))
print('R Squared: %.3f' % predRsquared)
print('RMSE: %.3f' % rmse)
```

    R Squared: -0.036
    RMSE: 130824.588



```python
#plotting results
plt.scatter(y_test, predictions, color=['red'], alpha = 0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();
```


![png](RForkner_Credit_One_Portfolio_files/RForkner_Credit_One_Portfolio_75_0.png)



```python
for i in range(len(names)):
    print(names[i],results[i].mean())
```

    Random Forest Regressor 0.45994203673169737
    Linear Regression 0.3262248324035236
    Support Vector Regression -0.050522201381185905


### All three regression analyses returned very poor results with regards to accuracy!  Perhaps we should try classification analyses instead!

### Since we're gonig to try classification analyses the dependent variable (credit limit) needs to be discretized or binned.  We'll look at the distribution of the credit limit data to help guide the binning:



```python
#Data Visualization, Plot Histograms
sns.distplot(Credit_Clean['LIMIT_BAL'])
```

    C:\Users\rob\anaconda3\envs\Data_Analytics\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)





    <AxesSubplot:xlabel='LIMIT_BAL', ylabel='Density'>




![png](RForkner_Credit_One_Portfolio_files/RForkner_Credit_One_Portfolio_78_2.png)



```python
#distribution of data
Credit_Clean['LIMIT_BAL'].describe(percentiles=[0, 1/3, 2/3, 1])
```




    count      30000.000000
    mean      167484.322667
    std       129747.661567
    min        10000.000000
    0%         10000.000000
    33.3%      80000.000000
    50%       140000.000000
    66.7%     200000.000000
    100%     1000000.000000
    max      1000000.000000
    Name: LIMIT_BAL, dtype: float64



### Since 2/3 of the customers have credit limits under 200K, but the maximum credit limit allowed is 1million, we will split the data into no more than 5 bins.


```python
#discretization using pandas cut.
Credit_Clean['LIMIT_BAL']  = pd.cut(Credit_Clean['LIMIT_BAL'], bins=4, labels=False)
Credit_Clean['LIMIT_BAL'].value_counts()
```




    0    23283
    1     6511
    2      200
    3        6
    Name: LIMIT_BAL, dtype: int64



#### Now re-run the analysis with random forest classifier.


```python
#re-order columns to put desired dependent variable last
column_names = ["ID", "SEX", "MARRIAGE", "AGE","EDUCATION","default payment next month","LIMIT_BAL","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
Credit_Clean = Credit_Clean.reindex(columns=column_names)
#print(Credit_Clean)
```


```python
#features, setting independent variables; removed ID as it was skewing the dataset
X = Credit_Clean.iloc[:,1:6]
print('Summary of feature sample')
X.head()
```

    Summary of feature sample





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEX</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>EDUCATION</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>26</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>34</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>37</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>57</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Setting Dependent Variable
y = Credit_Clean['LIMIT_BAL']
y.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: LIMIT_BAL, dtype: int64




```python
#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
```


```python
#Modeling (Classification)
algo = RandomForestClassifier(n_estimators=100)
model = algo.fit(X_train,y_train)
```


```python
#Predictions
preds = model.predict(X_test)
```


```python
print(classification_report(y_test, preds))
```

                  precision    recall  f1-score   support

               0       0.79      0.97      0.87      5874
               1       0.42      0.08      0.13      1568
               2       0.00      0.00      0.00        57
               3       0.00      0.00      0.00         1

        accuracy                           0.78      7500
       macro avg       0.30      0.26      0.25      7500
    weighted avg       0.71      0.78      0.71      7500



    C:\Users\rob\anaconda3\envs\Data_Analytics\lib\site-packages\sklearn\metrics\_classification.py:1268: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



```python
cf=confusion_matrix(y_test, preds)
print(cf)
```

    [[5704  170    0    0]
     [1442  126    0    0]
     [  51    6    0    0]
     [   1    0    0    0]]



```python
print("Accuracy:",metrics.accuracy_score(y_test, preds))
```

    Accuracy: 0.7773333333333333


### In this case using binning and classification modeling we've been able to arrive at a better model for predicting credit limit among customers.  The model is about 78% accurate and weighs Age and Education as the most important variables in determining credit limit.


```python
feature_names=X.columns
print(feature_names)
```

    Index(['SEX', 'MARRIAGE', 'AGE', 'EDUCATION', 'default payment next month'], dtype='object')



```python
#importance of variables
feature_imp = pd.Series(algo.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp
```




    AGE                           0.625595
    EDUCATION                     0.235429
    default payment next month    0.057261
    MARRIAGE                      0.056731
    SEX                           0.024984
    dtype: float64



### What about whether someone should be granted credit at all?  In this case we need to change the dependent variable we're modeling for from credit limit (LIMIT_BAL) to default.  Default is already a classified answer, either someone defaults or not, so the model will be based on classification methods rather than a regression.

####  First some additional tidying.  We'll remove past payment data that do not affect credit score, just to slim down the model.  The values -2 = No consumption; -1 = Paid in full, so we will remove them:


```python
#Remove Past payment data that do not affect credit score.
Credit_Clean['PAY_0'] = Credit_Clean['PAY_0'].replace(-2,0)
Credit_Clean['PAY_2'] = Credit_Clean['PAY_2'].replace(-2,0)
Credit_Clean['PAY_3'] = Credit_Clean['PAY_3'].replace(-2,0)
Credit_Clean['PAY_4'] = Credit_Clean['PAY_4'].replace(-2,0)
Credit_Clean['PAY_5'] = Credit_Clean['PAY_5'].replace(-2,0)
Credit_Clean['PAY_6'] = Credit_Clean['PAY_6'].replace(-2,0)

# do the same for the -1 variable
Credit_Clean['PAY_0'] = Credit_Clean['PAY_0'].replace(-1,0)
Credit_Clean['PAY_2'] = Credit_Clean['PAY_2'].replace(-1,0)
Credit_Clean['PAY_3'] = Credit_Clean['PAY_3'].replace(-1,0)
Credit_Clean['PAY_4'] = Credit_Clean['PAY_4'].replace(-1,0)
Credit_Clean['PAY_5'] = Credit_Clean['PAY_5'].replace(-1,0)
Credit_Clean['PAY_6'] = Credit_Clean['PAY_6'].replace(-1,0)
```

### Now we'll re-order the dataframe and set the variables.


```python
#re-order columns to put desired dependent variable last
column_names = ["ID", "SEX", "MARRIAGE", "AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","EDUCATION","LIMIT_BAL","default payment next month"]
Credit_Clean = Credit_Clean.reindex(columns=column_names)
#print(Credit_Clean)
```


```python
#features, setting independent variables; removed ID as it was skewing the dataset
X = Credit_Clean.iloc[:,1:24]
print('Summary of feature sample')
X.head()
```

    Summary of feature sample





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEX</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>...</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>EDUCATION</th>
      <th>LIMIT_BAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3913</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>26</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2682</td>
      <td>...</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239</td>
      <td>...</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990</td>
      <td>...</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8617</td>
      <td>...</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
#Setting Dependent Variable
y = Credit_Clean['default payment next month']
y.head()
```




    0    0
    1    0
    2    1
    3    1
    4    1
    Name: default payment next month, dtype: int32



### Now we'll prepare and run the model


```python
#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 123)
```


```python
#Modeling (Classification)
algo = RandomForestClassifier(n_estimators=100)
model = algo.fit(X_train,y_train)
```


```python
#Predictions
preds = model.predict(X_test)
```


```python
print(classification_report(y_test, preds))
```

                  precision    recall  f1-score   support

               0       0.60      0.39      0.47      1627
               1       0.85      0.93      0.88      5873

        accuracy                           0.81      7500
       macro avg       0.72      0.66      0.68      7500
    weighted avg       0.79      0.81      0.79      7500




```python
print("Accuracy:",metrics.accuracy_score(y_test, preds))
```

    Accuracy: 0.8104



```python
feature_names=X.columns
print(feature_names)
```

    Index(['SEX', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
           'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
           'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
           'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'EDUCATION', 'LIMIT_BAL'],
          dtype='object')



```python
#importance of variables
feature_imp = pd.Series(algo.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp
```




    PAY_0        0.092924
    AGE          0.077121
    BILL_AMT1    0.067179
    BILL_AMT2    0.060077
    BILL_AMT3    0.056737
    PAY_AMT1     0.056130
    BILL_AMT6    0.055266
    BILL_AMT4    0.054641
    BILL_AMT5    0.054381
    PAY_AMT2     0.053888
    PAY_AMT6     0.050081
    PAY_AMT3     0.049997
    PAY_AMT5     0.048750
    PAY_AMT4     0.048078
    PAY_2        0.041329
    PAY_3        0.024142
    EDUCATION    0.020740
    PAY_4        0.018375
    PAY_5        0.017828
    PAY_6        0.015764
    MARRIAGE     0.015345
    SEX          0.012817
    LIMIT_BAL    0.008410
    dtype: float64



### So we end with a result of about 81% accuracy in answering the question whether or not we can come up with a model that can predict whether a customer should be given credit at all, and using our one example model we see that arriving at that answer used quite a few of the variables, including payment and billing history.


```python

```