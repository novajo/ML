
# Identifying safe loans with decision trees

The [LendingClub](https://www.lendingclub.com/) is a peer-to-peer leading company that directly connects borrowers and potential lenders/investors. In this notebook, you will build a classification model to predict whether or not a loan provided by LendingClub is likely to [default](https://en.wikipedia.org/wiki/Default_%28finance%29).

In this notebook you will use data from the LendingClub to predict whether a loan will be paid off in full or the loan will be [charged off](https://en.wikipedia.org/wiki/Charge-off) and possibly go into default. In this assignment you will:

* Use SFrames to do some feature engineering.
* Train a decision-tree on the LendingClub dataset.
* Visualize the tree.
* Predict whether a loan will default along with prediction probabilities (on a validation set).
* Train a complex tree model and compare it to simple tree model.

Let's get started!

## Fire up GraphLab Create

Make sure you have the latest version of GraphLab Create. If you don't find the decision tree module, then you would need to upgrade GraphLab Create using

```
   pip install graphlab-create --upgrade
```


```python
# import graphlab
# graphlab.canvas.set_target('ipynb')
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn import tree
```

# Load LendingClub dataset

We will be using a dataset from the [LendingClub](https://www.lendingclub.com/). A parsed and cleaned form of the dataset is availiable [here](https://github.com/learnml/machine-learning-specialization-private). Make sure you **download the dataset** before running the following command.


```python
loans = pandas.read_csv('D:/ml_data/lending-club-data.csv')
# loans = pandas.read_csv('/home/jo/我的坚果云/lending-club-data.csv')
```

    C:\anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (19,47) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
# loans.iloc[47]
```

## Exploring some features

Let's quickly explore what the dataset looks like. First, let's print out the column names to see what features we have in this dataset.


```python
loans.dtypes.index
```




    Index(['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
           'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title',
           'emp_length', 'home_ownership', 'annual_inc', 'is_inc_v', 'issue_d',
           'loan_status', 'pymnt_plan', 'url', 'desc', 'purpose', 'title',
           'zip_code', 'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
           'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
           'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
           'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
           'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
           'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
           'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
           'collections_12_mths_ex_med', 'mths_since_last_major_derog',
           'policy_code', 'not_compliant', 'status', 'inactive_loans', 'bad_loans',
           'emp_length_num', 'grade_num', 'sub_grade_num', 'delinq_2yrs_zero',
           'pub_rec_zero', 'collections_12_mths_zero', 'short_emp',
           'payment_inc_ratio', 'final_d', 'last_delinq_none', 'last_record_none',
           'last_major_derog_none'],
          dtype='object')



Here, we see that we have some feature columns that have to do with grade of the loan, annual income, home ownership status, etc. Let's take a look at the distribution of loan grades in the dataset.


```python
loans['grade'].head()
```




    0    B
    1    C
    2    C
    3    C
    4    A
    Name: grade, dtype: object




```python
loans['bad_loans'].head()
```




    0    0
    1    1
    2    0
    3    0
    4    0
    Name: bad_loans, dtype: int64



We can see that over half of the loan grades are assigned values `B` or `C`. Each loan is assigned one of these grades, along with a more finely discretized feature called `sub_grade` (feel free to explore that feature column as well!). These values depend on the loan application and credit report, and determine the interest rate of the loan. More information can be found [here](https://www.lendingclub.com/public/rates-and-fees.action).

Now, let's look at a different feature.


```python
loans['home_ownership'].head()
```




    0    RENT
    1    RENT
    2    RENT
    3    RENT
    4    RENT
    Name: home_ownership, dtype: object



This feature describes whether the loanee is mortaging, renting, or owns a home. We can see that a small percentage of the loanees own a home.

## Exploring the target column

The target column (label column) of the dataset that we are interested in is called `bad_loans`. In this column **1** means a risky (bad) loan **0** means a safe  loan.

In order to make this more intuitive and consistent with the lectures, we reassign the target to be:
* **+1** as a safe  loan, 
* **-1** as a risky (bad) loan. 

We put this in a new column called `safe_loans`.


```python
# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
del loans['bad_loans']
```


```python
loans.dtypes.index
```




    Index(['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
           'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title',
           'emp_length', 'home_ownership', 'annual_inc', 'is_inc_v', 'issue_d',
           'loan_status', 'pymnt_plan', 'url', 'desc', 'purpose', 'title',
           'zip_code', 'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
           'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
           'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
           'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
           'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
           'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
           'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
           'collections_12_mths_ex_med', 'mths_since_last_major_derog',
           'policy_code', 'not_compliant', 'status', 'inactive_loans',
           'emp_length_num', 'grade_num', 'sub_grade_num', 'delinq_2yrs_zero',
           'pub_rec_zero', 'collections_12_mths_zero', 'short_emp',
           'payment_inc_ratio', 'final_d', 'last_delinq_none', 'last_record_none',
           'last_major_derog_none', 'safe_loans'],
          dtype='object')



Now, let us explore the distribution of the column `safe_loans`. This gives us a sense of how many safe and risky loans are present in the dataset.


```python
loans['safe_loans'].value_counts()
```




     1    99457
    -1    23150
    Name: safe_loans, dtype: int64




```python
loans['safe_loans'].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fc66bd0>




![png](output_22_1.png)


You should have:
* Around 81% safe loans
* Around 19% risky loans

It looks like most of these loans are safe loans (thankfully). But this does make our problem of identifying risky loans challenging.

## Features for the classification algorithm

In this assignment, we will be using a subset of features (categorical and numeric). The features we will be using are **described in the code comments** below. If you are a finance geek, the [LendingClub](https://www.lendingclub.com/) website has a lot more details about these features.


```python
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]
```

What remains now is a **subset of features** and the **target** that we will use for the rest of this notebook. 


```python
loans.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>home_ownership</th>
      <th>dti</th>
      <th>purpose</th>
      <th>term</th>
      <th>last_delinq_none</th>
      <th>last_major_derog_none</th>
      <th>revol_util</th>
      <th>total_rec_late_fee</th>
      <th>safe_loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>27.65</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>83.7</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>C4</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>1.00</td>
      <td>car</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>9.4</td>
      <td>0.00</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>8.72</td>
      <td>small_business</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>98.5</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>20.00</td>
      <td>other</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>21.0</td>
      <td>16.97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>A4</td>
      <td>0</td>
      <td>4</td>
      <td>RENT</td>
      <td>11.20</td>
      <td>wedding</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>28.3</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(loans)
```




    122607



## Sample data to balance classes

As we explored above, our data is disproportionally full of safe loans.  Let's create two datasets: one with just the safe loans (`safe_loans_raw`) and one with just the risky loans (`risky_loans_raw`).


```python
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print ("Number of safe loans  : {}" .format(len(safe_loans_raw)))
print ("Number of risky loans : {}" .format(len(risky_loans_raw)))
```

    Number of safe loans  : 99457
    Number of risky loans : 23150
    

Now, write some code to compute below the percentage of safe and risky loans in the dataset and validate these numbers against what was given using `.show` earlier in the assignment:


```python
numofsafe = int(len(safe_loans_raw))
numofrisky = int(len(risky_loans_raw))
total = numofsafe + numofrisky
print ("Percentage of safe loans  : {:.2%}".format(numofsafe/total))
print ("Percentage of risky loans : {:.2%}".format(numofrisky/total))
```

    Percentage of safe loans  : 81.12%
    Percentage of risky loans : 18.88%
    

One way to combat class imbalance is to undersample the larger class until the class distribution is approximately half and half. Here, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points. We used `seed=1` so everyone gets the same results.


```python
risky_loans_raw
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>home_ownership</th>
      <th>dti</th>
      <th>purpose</th>
      <th>term</th>
      <th>last_delinq_none</th>
      <th>last_major_derog_none</th>
      <th>revol_util</th>
      <th>total_rec_late_fee</th>
      <th>safe_loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>C4</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>1.00</td>
      <td>car</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>9.4</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>F</td>
      <td>F2</td>
      <td>0</td>
      <td>5</td>
      <td>OWN</td>
      <td>5.55</td>
      <td>small_business</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>32.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B</td>
      <td>B5</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>18.08</td>
      <td>other</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>36.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>C</td>
      <td>C1</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>10.08</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>91.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>4</td>
      <td>RENT</td>
      <td>7.06</td>
      <td>other</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>55.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>B</td>
      <td>B4</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>13.22</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>90.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>B</td>
      <td>B3</td>
      <td>0</td>
      <td>2</td>
      <td>RENT</td>
      <td>2.40</td>
      <td>major_purchase</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>29.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>C</td>
      <td>C2</td>
      <td>0</td>
      <td>10</td>
      <td>RENT</td>
      <td>15.22</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>57.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>D</td>
      <td>D2</td>
      <td>0</td>
      <td>3</td>
      <td>RENT</td>
      <td>13.97</td>
      <td>other</td>
      <td>60 months</td>
      <td>0</td>
      <td>1</td>
      <td>59.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>A</td>
      <td>A5</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>16.33</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>62.1</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>9</td>
      <td>MORTGAGE</td>
      <td>9.12</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>63.7</td>
      <td>24.170</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>5</td>
      <td>RENT</td>
      <td>20.88</td>
      <td>car</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>90.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>E</td>
      <td>E4</td>
      <td>0</td>
      <td>8</td>
      <td>RENT</td>
      <td>21.58</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>97.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>58</th>
      <td>D</td>
      <td>D3</td>
      <td>0</td>
      <td>6</td>
      <td>RENT</td>
      <td>13.16</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>70.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>F</td>
      <td>F2</td>
      <td>0</td>
      <td>5</td>
      <td>RENT</td>
      <td>12.48</td>
      <td>small_business</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>73.9</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>D</td>
      <td>D2</td>
      <td>0</td>
      <td>6</td>
      <td>RENT</td>
      <td>20.22</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>67.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>D</td>
      <td>D3</td>
      <td>0</td>
      <td>8</td>
      <td>MORTGAGE</td>
      <td>21.31</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>86.1</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>89</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>3</td>
      <td>RENT</td>
      <td>20.64</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>47.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>93</th>
      <td>D</td>
      <td>D2</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>23.18</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>79.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>102</th>
      <td>D</td>
      <td>D5</td>
      <td>0</td>
      <td>2</td>
      <td>RENT</td>
      <td>24.14</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>96.3</td>
      <td>36.247</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>108</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>5</td>
      <td>RENT</td>
      <td>22.80</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>54.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>E</td>
      <td>E4</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>20.70</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>87.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>118</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>9</td>
      <td>MORTGAGE</td>
      <td>9.17</td>
      <td>home_improvement</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>71.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>124</th>
      <td>A</td>
      <td>A2</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>29.85</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>62.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>132</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>3</td>
      <td>RENT</td>
      <td>7.83</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>65.4</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>136</th>
      <td>B</td>
      <td>B3</td>
      <td>0</td>
      <td>9</td>
      <td>RENT</td>
      <td>22.08</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>29.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>138</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>21.89</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>65.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>2</td>
      <td>MORTGAGE</td>
      <td>12.24</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>90.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>151</th>
      <td>A</td>
      <td>A3</td>
      <td>1</td>
      <td>0</td>
      <td>OWN</td>
      <td>16.30</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>42.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>158</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>6</td>
      <td>RENT</td>
      <td>6.92</td>
      <td>small_business</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>69.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>122470</th>
      <td>D</td>
      <td>D1</td>
      <td>0</td>
      <td>10</td>
      <td>RENT</td>
      <td>10.84</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>54.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122471</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>21.86</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>86.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122475</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>6.16</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>0</td>
      <td>82.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122483</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>13.22</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>0</td>
      <td>1</td>
      <td>40.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122488</th>
      <td>D</td>
      <td>D4</td>
      <td>0</td>
      <td>8</td>
      <td>MORTGAGE</td>
      <td>13.73</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>44.0</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122492</th>
      <td>F</td>
      <td>F3</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>11.97</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>30.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122495</th>
      <td>D</td>
      <td>D1</td>
      <td>0</td>
      <td>10</td>
      <td>RENT</td>
      <td>14.90</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>81.4</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122509</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>26.02</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>40.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122511</th>
      <td>C</td>
      <td>C1</td>
      <td>1</td>
      <td>0</td>
      <td>MORTGAGE</td>
      <td>28.06</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>0</td>
      <td>0</td>
      <td>69.9</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122527</th>
      <td>B</td>
      <td>B4</td>
      <td>0</td>
      <td>11</td>
      <td>OWN</td>
      <td>5.88</td>
      <td>home_improvement</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122534</th>
      <td>E</td>
      <td>E5</td>
      <td>0</td>
      <td>7</td>
      <td>OWN</td>
      <td>10.90</td>
      <td>small_business</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>21.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122542</th>
      <td>D</td>
      <td>D1</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>6.83</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>64.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122543</th>
      <td>C</td>
      <td>C2</td>
      <td>0</td>
      <td>2</td>
      <td>MORTGAGE</td>
      <td>6.60</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>50.0</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122556</th>
      <td>D</td>
      <td>D4</td>
      <td>0</td>
      <td>6</td>
      <td>MORTGAGE</td>
      <td>22.73</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>64.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122557</th>
      <td>D</td>
      <td>D1</td>
      <td>0</td>
      <td>3</td>
      <td>MORTGAGE</td>
      <td>16.16</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>63.1</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122563</th>
      <td>C</td>
      <td>C2</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>7.46</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>0</td>
      <td>12.4</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122565</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>2</td>
      <td>RENT</td>
      <td>23.76</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>39.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122566</th>
      <td>B</td>
      <td>B3</td>
      <td>1</td>
      <td>0</td>
      <td>RENT</td>
      <td>11.37</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>40.0</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122570</th>
      <td>A</td>
      <td>A4</td>
      <td>0</td>
      <td>7</td>
      <td>MORTGAGE</td>
      <td>27.68</td>
      <td>major_purchase</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>70.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122573</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>18.37</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>55.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122580</th>
      <td>B</td>
      <td>B5</td>
      <td>0</td>
      <td>2</td>
      <td>RENT</td>
      <td>27.75</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>83.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122583</th>
      <td>C</td>
      <td>C4</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>18.09</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122587</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>9</td>
      <td>MORTGAGE</td>
      <td>11.48</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>64.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122590</th>
      <td>B</td>
      <td>B5</td>
      <td>1</td>
      <td>1</td>
      <td>OWN</td>
      <td>19.62</td>
      <td>small_business</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>68.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122594</th>
      <td>C</td>
      <td>C3</td>
      <td>1</td>
      <td>1</td>
      <td>OWN</td>
      <td>26.21</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>28.4</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122596</th>
      <td>D</td>
      <td>D1</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>19.56</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>52.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122601</th>
      <td>B</td>
      <td>B5</td>
      <td>0</td>
      <td>5</td>
      <td>MORTGAGE</td>
      <td>18.69</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>29.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122602</th>
      <td>E</td>
      <td>E5</td>
      <td>1</td>
      <td>0</td>
      <td>MORTGAGE</td>
      <td>1.50</td>
      <td>medical</td>
      <td>60 months</td>
      <td>0</td>
      <td>0</td>
      <td>14.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122604</th>
      <td>D</td>
      <td>D3</td>
      <td>0</td>
      <td>6</td>
      <td>MORTGAGE</td>
      <td>12.28</td>
      <td>medical</td>
      <td>60 months</td>
      <td>0</td>
      <td>0</td>
      <td>10.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>122605</th>
      <td>D</td>
      <td>D5</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>18.45</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>46.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>23150 rows × 13 columns</p>
</div>




```python
# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
```


```python
percentage
```




    0.2327639080205516




```python
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(frac=percentage, random_state=1)
print(safe_loans)
# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)
```

           grade sub_grade  short_emp  emp_length_num home_ownership    dti  \
    14937      B        B5          0               8           RENT  21.97   
    104761     A        A1          0              11           RENT   0.92   
    77248      B        B4          1               1           RENT   7.73   
    120821     A        A5          1               1           RENT   4.12   
    104521     D        D4          0              11           RENT   9.34   
    89340      B        B4          0              11       MORTGAGE  16.08   
    97993      A        A4          0              11       MORTGAGE  18.83   
    51510      A        A5          0               5       MORTGAGE   8.19   
    90723      B        B3          0              11       MORTGAGE  18.10   
    19435      C        C3          0               6           RENT  20.13   
    32947      C        C3          0              11           RENT  14.00   
    119000     C        C3          1               1           RENT  20.28   
    97425      A        A3          0              11       MORTGAGE  23.78   
    10733      A        A2          0              11       MORTGAGE  22.55   
    68245      D        D4          1               0       MORTGAGE  34.20   
    83085      C        C1          0              10       MORTGAGE   9.28   
    48666      B        B3          0               8           RENT  13.97   
    21646      D        D2          0               2           RENT  11.08   
    104040     A        A2          0               2           RENT  16.88   
    19715      A        A3          0               4       MORTGAGE   8.14   
    106034     C        C4          0              11           RENT   9.29   
    54040      C        C2          1               1           RENT  25.62   
    32830      B        B4          0               5           RENT  14.68   
    120528     F        F3          0              11            OWN  26.82   
    45981      F        F5          0               9           RENT  26.69   
    112102     C        C4          0              11       MORTGAGE  14.83   
    31689      A        A4          0               6           RENT   5.07   
    79139      A        A2          0               8           RENT   1.92   
    20506      A        A2          0               4           RENT  23.94   
    69886      D        D1          1               0       MORTGAGE  22.94   
    ...      ...       ...        ...             ...            ...    ...   
    55027      B        B5          0               7       MORTGAGE   7.28   
    100438     C        C1          0               6       MORTGAGE  15.37   
    122519     D        D4          1               0            OWN  29.98   
    4161       B        B1          0               5       MORTGAGE  17.33   
    13520      C        C5          0               4           RENT  15.77   
    51558      C        C4          0              11       MORTGAGE  20.88   
    110837     A        A2          0               3           RENT   3.00   
    83788      C        C2          1               0       MORTGAGE  22.27   
    6556       D        D1          0               5       MORTGAGE   5.84   
    106731     A        A5          0               3       MORTGAGE  17.59   
    111533     B        B2          0               8           RENT  16.09   
    115626     C        C5          1               1       MORTGAGE  19.01   
    121848     C        C2          0              11       MORTGAGE  18.37   
    108846     C        C3          0              10       MORTGAGE  13.12   
    111652     C        C5          0              11       MORTGAGE  20.90   
    74640      A        A1          0              11       MORTGAGE  10.15   
    90254      B        B2          0               4           RENT  14.21   
    73781      C        C3          0               7       MORTGAGE  22.36   
    46499      B        B4          0              11       MORTGAGE  12.02   
    6011       B        B2          0              11       MORTGAGE  10.22   
    92462      C        C1          0               4           RENT  26.35   
    122138     C        C5          0              11       MORTGAGE   4.41   
    98381      B        B5          1               1           RENT   8.37   
    104878     C        C1          0               8            OWN   4.22   
    38265      C        C3          0               7       MORTGAGE  24.32   
    10016      A        A4          0               4       MORTGAGE  21.36   
    58367      B        B1          0               8       MORTGAGE  16.29   
    90431      B        B1          0               9           RENT  14.40   
    115727     F        F5          0               4           RENT  29.45   
    105752     D        D5          0               2       MORTGAGE  11.65   
    
                       purpose        term  last_delinq_none  \
    14937   debt_consolidation   60 months                 1   
    104761                 car   36 months                 1   
    77248   debt_consolidation   36 months                 1   
    120821      major_purchase   36 months                 1   
    104521         credit_card   36 months                 1   
    89340          credit_card   36 months                 0   
    97993   debt_consolidation   36 months                 1   
    51510   debt_consolidation   60 months                 1   
    90723   debt_consolidation   36 months                 1   
    19435   debt_consolidation   36 months                 1   
    32947   debt_consolidation   36 months                 1   
    119000         credit_card   36 months                 0   
    97425   debt_consolidation   36 months                 1   
    10733                  car   36 months                 1   
    68245   debt_consolidation   36 months                 0   
    83085          credit_card   36 months                 1   
    48666          credit_card   36 months                 1   
    21646                  car   36 months                 1   
    104040  debt_consolidation   36 months                 1   
    19715          credit_card   36 months                 1   
    106034  debt_consolidation   36 months                 1   
    54040          credit_card   36 months                 1   
    32830          credit_card   36 months                 1   
    120528  debt_consolidation   36 months                 1   
    45981   debt_consolidation   60 months                 0   
    112102    home_improvement   36 months                 0   
    31689          credit_card   36 months                 0   
    79139       small_business   36 months                 1   
    20506             vacation   36 months                 1   
    69886     home_improvement   36 months                 1   
    ...                    ...         ...               ...   
    55027          credit_card   36 months                 0   
    100438  debt_consolidation   36 months                 1   
    122519  debt_consolidation   36 months                 1   
    4161           credit_card   36 months                 1   
    13520   debt_consolidation   60 months                 1   
    51558          credit_card   36 months                 1   
    110837         credit_card   36 months                 1   
    83788   debt_consolidation   36 months                 1   
    6556    debt_consolidation   36 months                 1   
    106731  debt_consolidation   60 months                 1   
    111533  debt_consolidation   36 months                 0   
    115626         credit_card   60 months                 1   
    121848  debt_consolidation   60 months                 0   
    108846    home_improvement   60 months                 1   
    111652  debt_consolidation   36 months                 1   
    74640   debt_consolidation   36 months                 1   
    90254   debt_consolidation   36 months                 0   
    73781          credit_card   36 months                 0   
    46499   debt_consolidation   36 months                 0   
    6011           credit_card   36 months                 0   
    92462          credit_card   36 months                 1   
    122138  debt_consolidation   60 months                 1   
    98381   debt_consolidation   36 months                 1   
    104878         credit_card   36 months                 1   
    38265          credit_card   60 months                 1   
    10016   debt_consolidation   36 months                 1   
    58367   debt_consolidation   36 months                 1   
    90431   debt_consolidation   36 months                 1   
    115727  debt_consolidation   60 months                 0   
    105752  debt_consolidation   60 months                 0   
    
            last_major_derog_none  revol_util  total_rec_late_fee  safe_loans  
    14937                       1        77.7                 0.0           1  
    104761                      1         0.6                 0.0           1  
    77248                       1        40.9                 0.0           1  
    120821                      1        15.4                 0.0           1  
    104521                      1        66.3                 0.0           1  
    89340                       1        46.7                 0.0           1  
    97993                       1        27.2                 0.0           1  
    51510                       1        10.6                 0.0           1  
    90723                       1        61.7                 0.0           1  
    19435                       1        75.0                 0.0           1  
    32947                       1        33.1                 0.0           1  
    119000                      0        64.3                 0.0           1  
    97425                       1        66.9                 0.0           1  
    10733                       1         3.1                 0.0           1  
    68245                       0        59.2                 0.0           1  
    83085                       1        82.1                 0.0           1  
    48666                       1        72.1                 0.0           1  
    21646                       1        85.2                 0.0           1  
    104040                      1        63.5                 0.0           1  
    19715                       1        52.5                 0.0           1  
    106034                      1        75.5                 0.0           1  
    54040                       1        87.2                 0.0           1  
    32830                       1        34.4                 0.0           1  
    120528                      1        86.6                 0.0           1  
    45981                       0        97.5                 0.0           1  
    112102                      0        87.0                 0.0           1  
    31689                       1        17.8                 0.0           1  
    79139                       1        22.3                 0.0           1  
    20506                       1         0.0                 0.0           1  
    69886                       1        77.6                 0.0           1  
    ...                       ...         ...                 ...         ...  
    55027                       1        53.5                 0.0           1  
    100438                      1        63.8                 0.0           1  
    122519                      1        92.7                 0.0           1  
    4161                        1        77.1                 0.0           1  
    13520                       1        27.5                 0.0           1  
    51558                       1        78.8                 0.0           1  
    110837                      1         1.2                 0.0           1  
    83788                       1        71.4                 0.0           1  
    6556                        1        71.2                 0.0           1  
    106731                      1         9.0                 0.0           1  
    111533                      0        60.0                 0.0           1  
    115626                      1        70.7                 0.0           1  
    121848                      1        28.2                 0.0           1  
    108846                      1        40.0                 0.0           1  
    111652                      1        94.0                 0.0           1  
    74640                       1        67.2                 0.0           1  
    90254                       1         6.3                 0.0           1  
    73781                       1        72.4                 0.0           1  
    46499                       0        36.0                 0.0           1  
    6011                        1        49.1                 0.0           1  
    92462                       1        76.1                 0.0           1  
    122138                      0        36.0                 0.0           1  
    98381                       1        71.3                 0.0           1  
    104878                      1        81.6                 0.0           1  
    38265                       1        50.1                 0.0           1  
    10016                       1        16.5                 0.0           1  
    58367                       1        41.0                 0.0           1  
    90431                       1        36.3                 0.0           1  
    115727                      0        36.2                 0.0           1  
    105752                      1        83.0                 0.0           1  
    
    [23150 rows x 13 columns]
    


```python
loans_data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>home_ownership</th>
      <th>dti</th>
      <th>purpose</th>
      <th>term</th>
      <th>last_delinq_none</th>
      <th>last_major_derog_none</th>
      <th>revol_util</th>
      <th>total_rec_late_fee</th>
      <th>safe_loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>C4</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>1.00</td>
      <td>car</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>9.4</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>F</td>
      <td>F2</td>
      <td>0</td>
      <td>5</td>
      <td>OWN</td>
      <td>5.55</td>
      <td>small_business</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>32.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B</td>
      <td>B5</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>18.08</td>
      <td>other</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>36.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>C</td>
      <td>C1</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>10.08</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>91.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>4</td>
      <td>RENT</td>
      <td>7.06</td>
      <td>other</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>55.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>B</td>
      <td>B4</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>13.22</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>90.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>B</td>
      <td>B3</td>
      <td>0</td>
      <td>2</td>
      <td>RENT</td>
      <td>2.40</td>
      <td>major_purchase</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>29.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>C</td>
      <td>C2</td>
      <td>0</td>
      <td>10</td>
      <td>RENT</td>
      <td>15.22</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>57.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>D</td>
      <td>D2</td>
      <td>0</td>
      <td>3</td>
      <td>RENT</td>
      <td>13.97</td>
      <td>other</td>
      <td>60 months</td>
      <td>0</td>
      <td>1</td>
      <td>59.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>A</td>
      <td>A5</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>16.33</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>62.1</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>9</td>
      <td>MORTGAGE</td>
      <td>9.12</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>63.7</td>
      <td>24.170</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>5</td>
      <td>RENT</td>
      <td>20.88</td>
      <td>car</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>90.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>E</td>
      <td>E4</td>
      <td>0</td>
      <td>8</td>
      <td>RENT</td>
      <td>21.58</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>97.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>58</th>
      <td>D</td>
      <td>D3</td>
      <td>0</td>
      <td>6</td>
      <td>RENT</td>
      <td>13.16</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>70.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>F</td>
      <td>F2</td>
      <td>0</td>
      <td>5</td>
      <td>RENT</td>
      <td>12.48</td>
      <td>small_business</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>73.9</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>D</td>
      <td>D2</td>
      <td>0</td>
      <td>6</td>
      <td>RENT</td>
      <td>20.22</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>67.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>D</td>
      <td>D3</td>
      <td>0</td>
      <td>8</td>
      <td>MORTGAGE</td>
      <td>21.31</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>86.1</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>89</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>3</td>
      <td>RENT</td>
      <td>20.64</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>47.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>93</th>
      <td>D</td>
      <td>D2</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>23.18</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>79.7</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>102</th>
      <td>D</td>
      <td>D5</td>
      <td>0</td>
      <td>2</td>
      <td>RENT</td>
      <td>24.14</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>96.3</td>
      <td>36.247</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>108</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>5</td>
      <td>RENT</td>
      <td>22.80</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>54.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>E</td>
      <td>E4</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>20.70</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>87.6</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>118</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>9</td>
      <td>MORTGAGE</td>
      <td>9.17</td>
      <td>home_improvement</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>71.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>124</th>
      <td>A</td>
      <td>A2</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>29.85</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>62.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>132</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>3</td>
      <td>RENT</td>
      <td>7.83</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>65.4</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>136</th>
      <td>B</td>
      <td>B3</td>
      <td>0</td>
      <td>9</td>
      <td>RENT</td>
      <td>22.08</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>29.3</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>138</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>11</td>
      <td>RENT</td>
      <td>21.89</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>65.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>2</td>
      <td>MORTGAGE</td>
      <td>12.24</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>90.8</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>151</th>
      <td>A</td>
      <td>A3</td>
      <td>1</td>
      <td>0</td>
      <td>OWN</td>
      <td>16.30</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>42.2</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>158</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>6</td>
      <td>RENT</td>
      <td>6.92</td>
      <td>small_business</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>69.5</td>
      <td>0.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>55027</th>
      <td>B</td>
      <td>B5</td>
      <td>0</td>
      <td>7</td>
      <td>MORTGAGE</td>
      <td>7.28</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>53.5</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100438</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>6</td>
      <td>MORTGAGE</td>
      <td>15.37</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>63.8</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>122519</th>
      <td>D</td>
      <td>D4</td>
      <td>1</td>
      <td>0</td>
      <td>OWN</td>
      <td>29.98</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>92.7</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4161</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>5</td>
      <td>MORTGAGE</td>
      <td>17.33</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>77.1</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13520</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>4</td>
      <td>RENT</td>
      <td>15.77</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>27.5</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51558</th>
      <td>C</td>
      <td>C4</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>20.88</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>78.8</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>110837</th>
      <td>A</td>
      <td>A2</td>
      <td>0</td>
      <td>3</td>
      <td>RENT</td>
      <td>3.00</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>1.2</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>83788</th>
      <td>C</td>
      <td>C2</td>
      <td>1</td>
      <td>0</td>
      <td>MORTGAGE</td>
      <td>22.27</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>71.4</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6556</th>
      <td>D</td>
      <td>D1</td>
      <td>0</td>
      <td>5</td>
      <td>MORTGAGE</td>
      <td>5.84</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>71.2</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>106731</th>
      <td>A</td>
      <td>A5</td>
      <td>0</td>
      <td>3</td>
      <td>MORTGAGE</td>
      <td>17.59</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>9.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111533</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>8</td>
      <td>RENT</td>
      <td>16.09</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>115626</th>
      <td>C</td>
      <td>C5</td>
      <td>1</td>
      <td>1</td>
      <td>MORTGAGE</td>
      <td>19.01</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>70.7</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>121848</th>
      <td>C</td>
      <td>C2</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>18.37</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>0</td>
      <td>1</td>
      <td>28.2</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>108846</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>10</td>
      <td>MORTGAGE</td>
      <td>13.12</td>
      <td>home_improvement</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>40.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111652</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>20.90</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>94.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>74640</th>
      <td>A</td>
      <td>A1</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>10.15</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>67.2</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90254</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>4</td>
      <td>RENT</td>
      <td>14.21</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>6.3</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73781</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>7</td>
      <td>MORTGAGE</td>
      <td>22.36</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>72.4</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46499</th>
      <td>B</td>
      <td>B4</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>12.02</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>0</td>
      <td>0</td>
      <td>36.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6011</th>
      <td>B</td>
      <td>B2</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>10.22</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>0</td>
      <td>1</td>
      <td>49.1</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>92462</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>4</td>
      <td>RENT</td>
      <td>26.35</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>76.1</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>122138</th>
      <td>C</td>
      <td>C5</td>
      <td>0</td>
      <td>11</td>
      <td>MORTGAGE</td>
      <td>4.41</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>98381</th>
      <td>B</td>
      <td>B5</td>
      <td>1</td>
      <td>1</td>
      <td>RENT</td>
      <td>8.37</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>71.3</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>104878</th>
      <td>C</td>
      <td>C1</td>
      <td>0</td>
      <td>8</td>
      <td>OWN</td>
      <td>4.22</td>
      <td>credit_card</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>81.6</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38265</th>
      <td>C</td>
      <td>C3</td>
      <td>0</td>
      <td>7</td>
      <td>MORTGAGE</td>
      <td>24.32</td>
      <td>credit_card</td>
      <td>60 months</td>
      <td>1</td>
      <td>1</td>
      <td>50.1</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10016</th>
      <td>A</td>
      <td>A4</td>
      <td>0</td>
      <td>4</td>
      <td>MORTGAGE</td>
      <td>21.36</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>16.5</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>58367</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>8</td>
      <td>MORTGAGE</td>
      <td>16.29</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>41.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90431</th>
      <td>B</td>
      <td>B1</td>
      <td>0</td>
      <td>9</td>
      <td>RENT</td>
      <td>14.40</td>
      <td>debt_consolidation</td>
      <td>36 months</td>
      <td>1</td>
      <td>1</td>
      <td>36.3</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>115727</th>
      <td>F</td>
      <td>F5</td>
      <td>0</td>
      <td>4</td>
      <td>RENT</td>
      <td>29.45</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>0</td>
      <td>0</td>
      <td>36.2</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>105752</th>
      <td>D</td>
      <td>D5</td>
      <td>0</td>
      <td>2</td>
      <td>MORTGAGE</td>
      <td>11.65</td>
      <td>debt_consolidation</td>
      <td>60 months</td>
      <td>0</td>
      <td>1</td>
      <td>83.0</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>46300 rows × 13 columns</p>
</div>



Now, let's verify that the resulting percentage of safe and risky loans are each nearly 50%.


```python
print ("Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data)))
print ("Percentage of risky loans                :", len(risky_loans) / float(len(loans_data)))
print ("Total number of loans in our new dataset :", len(loans_data))
```

    Percentage of safe loans                 : 0.5
    Percentage of risky loans                : 0.5
    Total number of loans in our new dataset : 46300
    

**Note:** There are many approaches for dealing with imbalanced data, including some where we modify the learning algorithm. These approaches are beyond the scope of this course, but some of them are reviewed in this [paper](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=5128907&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel5%2F69%2F5173046%2F05128907.pdf%3Farnumber%3D5128907 ). For this assignment, we use the simplest possible approach, where we subsample the overly represented class to get a more balanced dataset. In general, and especially when the data is highly imbalanced, we recommend using more advanced methods.

## Split data into training and validation sets

We split the data into training and validation sets using an 80/20 split and specifying `seed=1` so everyone gets the same results.

**Note**: In previous assignments, we have called this a **train-test split**. However, the portion of data that we don't train on will be used to help **select model parameters** (this is known as model selection). Thus, this portion of data should be called a **validation set**. Recall that examining performance of various potential models (i.e. models with different parameters) should be on validation set, while evaluation of the final selected model should always be on test data. Typically, we would also save a portion of the data (a real test set) to test our final model on or use cross-validation on the training set to select our final model. But for the learning purposes of this assignment, we won't do that.


```python
(train_data,validation_data) = train_test_split(loans_data, train_size=0.8, random_state=0)
```


```python
train_data.shape, validation_data.shape
```




    ((37040, 13), (9260, 13))



# Use decision tree to build a classifier

Now, let's use the built-in GraphLab Create decision tree learner to create a loan prediction model on the training data. (In the next assignment, you will implement your own decision tree learning algorithm.)  Our feature columns and target column have already been decided above. Use `validation_set=None` to get the same results as everyone else.


```python
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# for i in range(13):
#     print(i)
#     print(train_data.values[:,i])
#     train_data.values[:,i] = le.fit_transform(train_data.values[:,i])
#     print(train_data.values[:,i])
```


```python
train_data.columns
```




    Index(['grade', 'sub_grade', 'short_emp', 'emp_length_num', 'home_ownership',
           'dti', 'purpose', 'term', 'last_delinq_none', 'last_major_derog_none',
           'revol_util', 'total_rec_late_fee', 'safe_loans'],
          dtype='object')




```python
type(train_data)
```




    pandas.core.frame.DataFrame




```python
train_data['grade'].head(3), train_data['safe_loans'].head(3)
```




    (9894     B
     32654    A
     65019    D
     Name: grade, dtype: object, 9894    -1
     32654    1
     65019   -1
     Name: safe_loans, dtype: int64)




```python
# for column_name in train_data.columns:
#     print(train_data[column_name].dtype)
#     if train_data[column_name].dtype == object:
#         print(1)
#     else:
#         print(-1)
```

## Use LabelEncoder to transfer all the object items into matrix


```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in train_data.columns:
    if train_data[column_name].dtype == object:
        train_data[column_name] = le.fit_transform(train_data[column_name])
        validation_data[column_name] = le.fit_transform(validation_data[column_name])
    else:
        pass
```

    C:\anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    C:\anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    


```python
train_data.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>home_ownership</th>
      <th>dti</th>
      <th>purpose</th>
      <th>term</th>
      <th>last_delinq_none</th>
      <th>last_major_derog_none</th>
      <th>revol_util</th>
      <th>total_rec_late_fee</th>
      <th>safe_loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9894</th>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>23.25</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>71.3</td>
      <td>14.9917</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>32654</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>15.00</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8.2</td>
      <td>0.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>65019</th>
      <td>3</td>
      <td>19</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>28.22</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>83.5</td>
      <td>0.0000</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# le.fit(train_data['purpose'])
```


```python
# list(le.classes_)
```


```python
# le.transform(train_data['purpose'])
```


```python
# len(le.transform(train_data.values[:,6])), len(train_data.values[:,6]), type(train_data['grade'])
```


```python
# train_data['purpose'] = le.transform(train_data['purpose'])
# train_data.loc[:,('purpose')] = le.transform(train_data.loc[:,('purpose')])
```


```python
# train_data['purpose']
```


```python
# train_data['term'].head(3)
```


```python
# X = train_data.values[:,0:12]
# Y = train_data.values[:,12].reshape(-1, 1)
# # X = train_data.iloc[:,12]
# # Y = train_data.iloc[:,0:12]
# print(X,Y)
```


```python
train_Y = train_data['safe_loans'].as_matrix()
train_X = train_data.drop('safe_loans', axis=1).as_matrix()
print(train_X)
print(train_Y)
```

    [[  1.       7.       1.     ...,   1.      71.3     14.9917]
     [  0.       1.       0.     ...,   1.       8.2      0.    ]
     [  3.      19.       0.     ...,   1.      83.5      0.    ]
     ..., 
     [  0.       0.       0.     ...,   1.      10.6      0.    ]
     [  1.       6.       0.     ...,   1.      39.3      0.    ]
     [  6.      32.       0.     ...,   1.      70.8      0.    ]]
    [-1  1 -1 ...,  1  1 -1]
    


```python
decision_tree_model = DecisionTreeClassifier(max_depth=6)
```


```python
# You can't pass str to your model fit() method.
decision_tree_model.fit(train_X, train_Y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



## Visualizing a learned model

As noted in the [documentation](https://dato.com/products/create/docs/generated/graphlab.boosted_trees_classifier.create.html#graphlab.boosted_trees_classifier.create), typically the max depth of the tree is capped at 6. However, such a tree can be hard to visualize graphically.  Here, we instead learn a smaller model with **max depth of 2** to gain some intuition by visualizing the learned tree.


```python
# small_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
#                    target = target, features = features, max_depth = 2)
small_model = DecisionTreeClassifier(criterion='gini', max_depth=2)
```


```python
small_model.fit(train_X, train_Y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



In the view that is provided by GraphLab Create, you can see each node, and each split at each node. This visualization is great for considering what happens when this model predicts the target of a new data point. 

**Note:** To better understand this visual:
* The root node is represented using pink. 
* Intermediate nodes are in green. 
* Leaf nodes in blue and orange. 


```python
dot_data = tree.export_graphviz(decision_tree_model, out_file=None, 
                         feature_names=features,  
                         class_names=("+1","-1"),  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = Source(dot_data)  
graph
```




![svg](output_73_0.svg)




```python
# graph = Source(tree.export_graphviz(decision_tree_model, out_file=None, feature_names=features))
```


```python
# graph
```


```python
# graph.format = 'png'
# graph.render('dtree_render',view=True)
```

# Making predictions

Let's consider two positive and two negative examples **from the validation set** and see what the model predicts. We will do the following:
* Predict whether or not a loan is safe.
* Predict the probability that a loan is safe.


```python
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
```


```python
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>home_ownership</th>
      <th>dti</th>
      <th>purpose</th>
      <th>term</th>
      <th>last_delinq_none</th>
      <th>last_major_derog_none</th>
      <th>revol_util</th>
      <th>total_rec_late_fee</th>
      <th>safe_loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113161</th>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>29.64</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>39.8</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>106326</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>18.60</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8.5</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16626</th>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>6.89</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>39.1</td>
      <td>0.0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>62925</th>
      <td>3</td>
      <td>17</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>21.14</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>68.8</td>
      <td>0.0</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample_exclude_target = sample_validation_data.iloc[:,0:12]
```

## Explore label predictions

Now, we will use our model  to predict whether or not a loan is likely to default. For each row in the **sample_validation_data**, use the **decision_tree_model** to predict whether or not the loan is classified as a **safe loan**. 

**Hint:** Be sure to use the `.predict()` method.


```python
decision_tree_model.predict(sample_exclude_target)
```




    array([-1,  1, -1, -1], dtype=int64)



**Quiz Question:** What percentage of the predictions on `sample_validation_data` did `decision_tree_model` get correct?

## Explore probability predictions

For each row in the **sample_validation_data**, what is the probability (according **decision_tree_model**) of a loan being classified as **safe**? 


**Hint:** Set `output_type='probability'` to make **probability** predictions using **decision_tree_model** on `sample_validation_data`:


```python
decision_tree_model.predict_proba(sample_exclude_target)
```




    array([[ 0.5       ,  0.5       ],
           [ 0.27347196,  0.72652804],
           [ 0.60796325,  0.39203675],
           [ 0.60477941,  0.39522059]])



**Quiz Question:** Which loan has the highest probability of being classified as a **safe loan**?

**Checkpoint:** Can you verify that for all the predictions with `probability >= 0.5`, the model predicted the label **+1**?

### Tricky predictions!

Now, we will explore something pretty interesting. For each row in the **sample_validation_data**, what is the probability (according to **small_model**) of a loan being classified as **safe**?

**Hint:** Set `output_type='probability'` to make **probability** predictions using **small_model** on `sample_validation_data`:


```python
small_model.predict_proba(sample_exclude_target)
```




    array([[ 0.42313756,  0.57686244],
           [ 0.24494122,  0.75505878],
           [ 0.66362791,  0.33637209],
           [ 0.66362791,  0.33637209]])



**Quiz Question:** Notice that the probability preditions are the **exact same** for the 2nd and 3rd loans. Why would this happen?

## Visualize the prediction on a tree


Note that you should be able to look at the small tree, traverse it yourself, and visualize the prediction being made. Consider the following point in the **sample_validation_data**


```python
sample_validation_data[0:1]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>home_ownership</th>
      <th>dti</th>
      <th>purpose</th>
      <th>term</th>
      <th>last_delinq_none</th>
      <th>last_major_derog_none</th>
      <th>revol_util</th>
      <th>total_rec_late_fee</th>
      <th>safe_loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113161</th>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>29.64</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>39.8</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize the small tree here to do the traversing for this data point.


```python
graph = Source(tree.export_graphviz(small_model, out_file=None, feature_names=features))
# graph.format = 'png'
# graph.render('dtree_render',view=True)
```


```python
graph
```




![svg](output_95_0.svg)



**Note:** In the tree visualization above, the values at the leaf nodes are not class predictions but scores (a slightly advanced concept that is out of the scope of this course). You can read more about this [here](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf).  If the score is $\geq$ 0, the class +1 is predicted.  Otherwise, if the score < 0, we predict class -1.


**Quiz Question:** Based on the visualized tree, what prediction would you make for this data point?

Now, let's verify your prediction by examining the prediction made using GraphLab Create.  Use the `.predict` function on `small_model`.


```python
small_model.predict(sample_exclude_target)
```




    array([ 1,  1, -1, -1], dtype=int64)



# Evaluating accuracy of the decision tree model

Recall that the accuracy is defined as follows:
$$
\mbox{accuracy} = \frac{\mbox{# correctly classified examples}}{\mbox{# total examples}}
$$

Let us start by evaluating the accuracy of the `small_model` and `decision_tree_model` on the training data


```python
# print small_model.evaluate(train_data)['accuracy']
# print decision_tree_model.evaluate(train_data)['accuracy']
```

**Checkpoint:** You should see that the **small_model** performs worse than the **decision_tree_model** on the training data.


Now, let us evaluate the accuracy of the **small_model** and **decision_tree_model** on the entire **validation_data**, not just the subsample considered above.


```python
def accuracy(model, validation_data):
    predict = model.predict(validation_data.iloc[:,0:12])
    # print(predict)
    actual = validation_data.iloc[:,12]
    # print(actual)
    result = (predict==actual).value_counts()
    # print(result.values[0])
    return (result.values[0])/len(actual)
```


```python
accuracy(small_model, sample_validation_data)
```




    1.0




```python
accuracy(decision_tree_model, sample_validation_data)
```




    0.75




```python
accuracy(small_model, validation_data)
```




    0.60421166306695462




```python
accuracy(decision_tree_model, validation_data)
```




    0.62926565874730023



**Quiz Question:** What is the accuracy of `decision_tree_model` on the validation set, rounded to the nearest .01?

## Evaluating accuracy of a complex decision tree model

Here, we will train a large decision tree with `max_depth=10`. This will allow the learned tree to become very deep, and result in a very complex model. Recall that in lecture, we prefer simpler models with similar predictive power. This will be an example of a more complicated model which has similar predictive power, i.e. something we don't want.


```python
big_model = DecisionTreeClassifier(max_depth=10)
big_model.fit(train_X, train_Y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



Now, let us evaluate **big_model** on the training set and validation set.


```python
# print big_model.evaluate(train_data)['accuracy']
# print big_model.evaluate(validation_data)['accuracy']
accuracy(big_model, sample_validation_data)
```




    1.0




```python
accuracy(big_model, validation_data)
```




    0.61295896328293742



**Checkpoint:** We should see that **big_model** has even better performance on the training set than **decision_tree_model** did on the training set.

**Quiz Question:** How does the performance of **big_model** on the validation set compare to **decision_tree_model** on the validation set? Is this a sign of overfitting?
