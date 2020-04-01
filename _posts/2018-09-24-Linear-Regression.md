---
title: "Linear Regression"
date: 2018-09-24
tags: [Machine Learning, Data Science, Linear Regression, Normal Equation, Cost Function]
excerpt: "Machine Learning, Linear Regression, Data Science, Normal Equation, Cost Function"
mathjax: "true"
---

# Medical cost personal data

In this notebook we will build Linear regression model for Medical cost dataset.

**Steps**
1. [Definition & Working principle](#Definition-&-Working-principle)
2. [Hypothesis representation](#Hypothesis-representation)
3. [Import Librarys and Dataset](#Import-Librarys-and-Dataset)
4. [Cost function](#Cost-function)
5. [Normal Equation](#Normal-Equation)
6. [Exploratory data analysis](#Exploratory-data-analysis)
7. [Data Preprocessing](#Data-Preprocessing)
8. [Box -Cox transformation](#Box--Cox-transformation)
9. [Polynomial feature](#Polynomial-feature)
10. [Train Test split](#Train-Test-split)
11. [Model building](#Model-building)
12. [Model evaluation](#Model-evaluation)
13. [Backward elimination](#Backward-elimination) 
14. [Model Validation](#Model-Validation)
15. [Summary](#Summary)

## Definition & Working principle
Let's build model using **Linear regression**.

Linear regression is a **supervised learining** algorithm used when target / dependent variable  **continues** real number. It establishes relationship between dependent variable $y$ and one or more independent variable $x$ using best fit line.   It work on the principle of ordinary least square $(OLS)$ / Mean square errror $(MSE)$. In statistics ols is method to estimated unkown parameter of linear regression function, it's goal is to minimize sum of square difference between observed dependent variable in the given data set and those predicted by linear regression fuction. 

## Hypothesis representation

We will use $x_i$ to denote the independent variable and $y_i$ to denote dependent variable. A pair of $(x_i,y_i)$ is called training example. The subscripe $i$ in the notation is simply index into the training set. We have $m$ training example then $i = 1,2,3,...m$. 

The goal of supervised learning is to learn a *hypothesis function $h$*, for a given training set that can used to estimate $y$ based on $x$. So hypothesis fuction represented as 

$$h_\theta(x_{i}) = \theta_0 + \theta_1x_i$$   
$\theta_0,\theta_1$ are parameter of hypothesis.This is equation for **Simple / Univariate Linear regression**. 

For **Multiple Linear regression** more than one independent variable exit then we will use $x_{ij}$ to denote indepedent variable and $y_{i}$ to denote dependent variable. We have $n$ independent variable then $j=1,2,3 ..... n$. The hypothesis function represented as

$$h_\theta(x_{i}) = \theta_0 + \theta_1x_{i1} + \theta_2 x_{i2} + ..... \theta_j x_{ij} ...... \theta_n  x_{mn}$$
$$\theta_0,\theta_1,....\theta_j....\theta_n$$ 

are parameter of hypothesis,
$m$ Number of training exaples,
$n$ Number of independent variable,
$x_{ij}$ is $i^{th}$ training exaple of $j^{th}$ feature.


## Import Librarys and Dataset
>Now we will import couple of python library required for our analysis and import dataset 


```python
# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] =14
#plt.rcParams.keys()
```


```python
# Import dataset
path ='dataset/'
#path = '../input/'
df = pd.read_csv(path+'insurance.csv')
print('\nNumber of rows and columns in the data set: ',df.shape)
print('')

#Lets look into top few rows and columns in the dataset
df.head()
```

    
    Number of rows and columns in the data set:  (1338, 7)
    





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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>



Now we have import dataset. When we look at the shape of dataset it has return as (1338,7).So there are $m=1338$ training exaple and $n=7$ independent variable. The target variable here is charges and remaining six variables such as age, sex, bmi, children, smoker, region are independent variable. There are multiple independent variable, so we need to fit Multiple linear regression. Then the hypothesis function looks like

$$h_\theta(x_{i}) = \theta_0+\theta_1 age + \theta_2 sex + \theta_3 bmi + \theta_4 children + \theta_5 smoker +
\theta_6 region$$  
This multiple linear regression equation for given dataset.  
If $i=1$ then $h_\theta(x_{1}) = \theta_0+\theta_1 19 + \theta_2 female + \theta_3 27.900 + \theta_4 1 + \theta_5 yes + \theta_6 southwest$ & $y_1 = 16884.92400$  
If $i=3$ then $h_\theta(x_{3}) = \theta_0+\theta_1 28 + \theta_2 male + \theta_3 33.000 + \theta_4 3 + \theta_5 no + \theta_6 northwest$ & $y_3 = 4449.46200$  
*Note*: In python index starts from 0.
$x_1 = \left(\begin{matrix} x_{11} & x_{12} & x_{13} & x_{14} & x_{15} & x_{16}\end{matrix}\right) 
    = \left(\begin{matrix} 19 & female & 27.900 & 1 & no & northwest\end{matrix}\right) $    

## Matrix Formulation

In general we can write above vector as $x_{ij} = \left( \begin{smallmatrix}  x_{i1} & x_{i2} &.&.&.& x_{in} \end{smallmatrix} \right)$

Now we combine all aviable individual vector into single input matrix of size $(m,n)$ and denoted it by $X$ input matrix, which consist of all training exaples,
$$X = \left( \begin{smallmatrix} x_{11} & x_{12} &.&.&.&.& x_{1n}\\
                                x_{21} & x_{22} &.&.&.&.& x_{2n}\\
                                x_{31} & x_{32} &.&.&.&.& x_{3n}\\
                                .&.&.&. &.&.&.& \\
                                .&.&.&. &.&.&.& \\
                                x_{m1} & x_{m2} &.&.&.&.&. x_{mn}\\
                                \end{smallmatrix} \right)_{(m,n)}$$

We represent parameter of function and dependent variable in vector form as  
$$\theta = \left (\begin{matrix} \theta_0 \\ \theta_1 \\ .\\.\\ \theta_j\\.\\.\\ \theta_n \end {matrix}\right)_{(n+1,1)} 
y = \left (\begin{matrix} y_1\\ y_2\\. \\. \\ y_i \\. \\. \\ y_m \end{matrix} \right)_{(m,1)}$$

So we represent hypothesis function in vectorize form $h_\theta{(x)} = X\theta$.


```python
""" for our visualization purpose will fit line using seaborn library only for bmi as independent variable 
and charges as dependent variable"""

sns.regplot(x='bmi',y='charges',data=df,marker='^',color='purple')
plt.xlabel('Boby Mass Index$(kg/m^2)$: as Independent variable')
plt.ylabel('Insurance Charges: as Dependent variable')
plt.title('Charge Vs BMI');
```


![png](output_7_0.png)


In above plot we fit regression line into the variables.

## Cost function

  A cost function measures how much error in the model is in terms of ability to estimate the relationship between $x$ and $y$. 
  We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference of observed dependent variable in the given the dataset and those predicted by the hypothesis function.
  
$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}_i - y_i)^2$$
$$J(\theta) =  \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x_i) - y_i)^2$$
To implement the linear regression, take training example add an extra column that is $x_0$ feature, where $x_0=1$. $x_{o} = \left( \begin{smallmatrix} x_{i0}+ x_{i1} & x_{i2} &.&.&.& x_{mi} \end{smallmatrix} \right)$,where $x_{i0} =0$ and input matrix will become as

$$X = \left( \begin{smallmatrix} x_{10} & x_{11} & x_{12} &.&.&.&.& x_{1n}\\
                                x_{20} & x_{21} & x_{22} &.&.&.&.& x_{2n}\\
                                x_{30} & x_{31} & x_{32} &.&.&.&.& x_{3n}\\
                                 .&.&.&.&. &.&.&.& \\
                                 .&.&.&.&. &.&.&.& \\
                                x_{m0} & x_{m1} & x_{m2} &.&.&.&.&. x_{mn}\\
                                \end{smallmatrix} \right)_{(m,n+1)}$$  
Each of the m input samples is similarly a column vector with n+1 rows $x_0$ being 1 for our convenience, that is $x_{10},x_{20},x_{30} .... x_{m0} =1$. Now we rewrite the ordinary least square cost function in matrix form as
$$J(\theta) = \frac{1}{m} (X\theta - y)^T(X\theta - y)$$

Let's look at the matrix multiplication concept,the multiplication of two matrix happens only if number of column of firt matrix is equal to number of row of second matrix. Here input matrix $X$ of size $(m,n+1)$, parameter of function is of size $(n+1,1)$ and dependent variable vector of size $(m,1)$. The product of matrix $X_{(m,n+1)}\theta_{(n+1,1)}$ will return a vector of size $(m,1)$, then product of $(X\theta - y)^T_{(1,m})(X\theta - y)_{(m,1)}$ will return size of unit vector. 

## Normal Equation
The normal equation is an analytical solution to the linear regression problem with a ordinary least square cost function. To minimize our cost function, take partial derivative of $J(\theta)$ with respect to $\theta$ and equate to $0$. The derivative of function is nothing but if a small change in input what would be the change in output of function.
 $$min_{\theta_0,\theta_1..\theta_n} J({\theta_0,\theta_1..\theta_n})$$
 $$\frac{\partial J(\theta_j)}{\partial\theta_j} =0$$ 
 where $j = 0,1,2,....n$
 
 Now we will apply partial derivative of our cost function,
 $$\frac{\partial J(\theta_j)}{\partial\theta_j} = \frac{\partial }{\partial \theta} \frac{1}{m}(X\theta - y)^T(X\theta - y) $$
 I will throw $\frac {1}{m}$ part away since we are going to compare a derivative to $0$. And solve $J(\theta),  
 
 $$J(\theta) = (X\theta -y)^T(X\theta - y)$$
 $$= (X\theta)^T - y^T)(X\theta -y)$$   
 $$= (\theta^T X^T - y^T)(X\theta - y)$$
 $$= \theta^T X^T X \theta - y^T X \theta - \theta^T X^T y + y^T y$$
 $$ = \theta^T X^T X \theta  - 2\theta^T X^T y + y^T y$$

Here $y^T_{(1,m)} X_{(m,n+1)} \theta_{(n+1,1)} = \theta^T_{(1,n+1)} X^T_{(n+1,m)} y_{(m,1)}$ because unit vector.

$$\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial}{\partial \theta} (\theta^T X^T X \theta  - 2\theta^T X^T y + y^T y )$$
$$ = X^T X \frac {\partial \theta^T \theta}{\partial\theta} - 2 X^T y \frac{\partial \theta^T}{\partial\theta} + \frac {\partial y^T y}{\partial\theta}$$
Partial derivative $\frac {\partial x^2}{\partial x} = 2x$, $\frac {\partial kx^2}{\partial x} = kx$,
$\frac {\partial Constact}{\partial x} = 0$

$$\frac{\partial J(\theta)}{\partial\theta} = X^T X 2\theta - 2X^T y +0$$
$$ 0 = 2X^T X \theta - 2X^T y$$
$$ X^T X \theta = X^T $$
$$ \theta = (X^TX)^{-1} X^Ty$$
this the normal equation for linear regression

## Exploratory data analysis


```python
df.describe()
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
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>13270.422265</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>12110.011237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>4740.287150</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.033000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>16639.912515</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>63770.428010</td>
    </tr>
  </tbody>
</table>
</div>



###  Check for missing value


```python
plt.figure(figsize=(12,4))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset')
```




    Text(0.5,1,'Missing value in the dataset')




![png](output_14_1.png)


There is no missing value in the data set


```python
# correlation plot
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff4b52322b0>




![png](output_16_1.png)


Thier no correlation among valiables.
### Plots


```python
f= plt.figure(figsize=(12,4))

ax=f.add_subplot(121)
sns.distplot(df['charges'],bins=50,color='r',ax=ax)
ax.set_title('Distribution of insurance charges')

ax=f.add_subplot(122)
sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
ax.set_title('Distribution of insurance charges in $log$ sacle')
ax.set_xscale('log')
```


![png](output_18_0.png)


If we look at the left plot the charges varies from 1120 to 63500, the plot is right skewed. In right plot we will apply natural log, then plot approximately tends to normal. for further analysis we will apply log on target variable charges. 


```python
f = plt.figure(figsize=(14,6))
ax = f.add_subplot(121)
sns.violinplot(x='sex', y='charges',data=df,palette='Wistia',ax=ax)
ax.set_title('Violin plot of Charges vs sex')

ax = f.add_subplot(122)
sns.violinplot(x='smoker', y='charges',data=df,palette='magma',ax=ax)
ax.set_title('Violin plot of Charges vs smoker')
```




    Text(0.5,1,'Violin plot of Charges vs smoker')




![png](output_20_1.png)


From left plot the insurance charge for male and female is approximatley in same range,it is average around 5000 bucks. In right plot the insurance charge for smokers is much wide range compare to non smokers, the average charges for non smoker is approximately 5000 bucks. For smoker the minimum insurance charge is itself 5000 bucks.


```python
plt.figure(figsize=(14,6))
sns.boxplot(x='children', y='charges',hue='sex',data=df,palette='coolwarm')
plt.title('Violin plot of charges vs children')
```




    Text(0.5,1,'Violin plot of charges vs children')




![png](output_22_1.png)



```python
df.groupby('children').agg(['mean','min','max'])['charges']
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
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>children</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12365.975602</td>
      <td>1121.8739</td>
      <td>63770.42801</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12731.171832</td>
      <td>1711.0268</td>
      <td>58571.07448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15073.563734</td>
      <td>2304.0022</td>
      <td>49577.66240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15355.318367</td>
      <td>3443.0640</td>
      <td>60021.39897</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13850.656311</td>
      <td>4504.6624</td>
      <td>40182.24600</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8786.035247</td>
      <td>4687.7970</td>
      <td>19023.26000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(14,6))
sns.violinplot(x='region', y='charges',hue='sex',data=df,palette='coolwarm',split=True)
plt.title('Violin plot of charges vs children')
```




    Text(0.5,1,'Violin plot of charges vs children')




![png](output_24_1.png)



```python
f = plt.figure(figsize=(14,6))
ax = f.add_subplot(121)
sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker',ax=ax)
ax.set_title('Scatter plot of Charges vs age')

ax = f.add_subplot(122)
sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
ax.set_title('Scatter plot of Charges vs bmi')
plt.savefig('sc.png');
```


![png](output_25_0.png)


>From left plot the minimum age person is insured is 18 year. There is slabs in policy most of non smoker take $1^{st}$ and $2^{nd}$ slab, for smoker policy start at $2^{nd}$ and $3^{rd}$ slab.

>Body mass index (BMI) is a measure of body fat based on height and weight that applies to adult men and women. The minimum bmi is 16$kg/m^2$ and maximum upto 54$kg/m^2$

## Data Preprocessing
### Encoding
Machine learning algorithms cannot work with categorical data directly, categorical data must be converted to number.
 1. Label Encoding
 2. One hot encoding
 3. Dummy variable trap

**Label encoding** refers to transforming the word labels into numerical form so that the algorithms can understand how to operate on them.

A **One hot encoding** is a representation of categorical variable as binary vectors.It allows the representation of categorical data to be more expresive. This first requires that the categorical values be mapped to integer values, that is label encoding. Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.

The **Dummy variable trap** is a scenario in which the independent variable are multicollinear, a scenario in which two or more variables are highly correlated in simple term one variable can be predicted from the others.

By using *pandas get_dummies* function we can do all above three step in line of code. We will this fuction  to get dummy variable for sex, children,smoker,region features. By setting *drop_first =True* function will remove dummy variable trap by droping one variable and original variable.The pandas makes our life easy.


```python
# Dummy variable
categorical_columns = ['sex','children', 'smoker', 'region']
df_encode = pd.get_dummies(data = df, prefix = None, prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
```


```python
# Lets verify the dummay variable process
print('Columns in original data frame:\n',df.columns.values)
print('\nNumber of rows and columns in the dataset:',df.shape)
print('\nColumns in data frame after encoding dummy variable:\n',df_encode.columns.values)
print('\nNumber of rows and columns in the dataset:',df_encode.shape)
```

    Columns in original data frame:
     ['age' 'sex' 'bmi' 'children' 'smoker' 'region' 'charges']
    
    Number of rows and columns in the dataset: (1338, 7)
    
    Columns in data frame after encoding dummy variable:
     ['age' 'bmi' 'charges' 'sex_male' 'children_1' 'children_2' 'children_3'
     'children_4' 'children_5' 'smoker_yes' 'region_northwest'
     'region_southeast' 'region_southwest']
    
    Number of rows and columns in the dataset: (1338, 13)


The original categorical variable are remove and also one of the one hot encode varible column for perticular categorical variable is droped from the column. So we completed all three encoding step by using get dummies function.

### Box -Cox transformation
A Box Cox transformation is a way to transform non-normal dependent variables into a normal shape. Normality is an important assumption for many statistical techniques; if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests. All that we need to perform this transformation is to find lambda value and apply the rule shown below to your variable.  
$$ \begin {cases}\frac {y^\lambda - 1}{\lambda},& y_i\neg=0 \\
 log(y_i) & \lambda = 0 \end{cases}$$
 The trick of Box-Cox transformation is to find lambda value, however in practice this is quite affordable. The following function returns the transformed variable, lambda value,confidence interval


```python
from scipy.stats import boxcox
y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)

#df['charges'] = y_bc  
# it did not perform better for this model, so log transform is used
ci,lam
```




    ((-0.01140290617294196, 0.0988096859767545), 0.043649053770664956)




```python
## Log transform
df_encode['charges'] = np.log(df_encode['charges'])
```

## Polynomial feature


```python
### Polynomial feature
#from sklearn.preprocessing import PolynomialFeatures
#poly_feat = PolynomialFeatures(degree=5,include_bias=False)
#t = poly_feat.fit_transform(df_encode[['age']])
#t = pd.DataFrame(t,columns= poly_feat.get_feature_names())
#t = t.iloc[:,1:]
#df_encode = pd.concat([df_encode,t],axis=1)

#df_encode['age**2'] = df_encode['age'] **2
#df_encode['age**3'] = df_encode['age'] **3
df_encode['age_smoker_yes'] = df_encode['age']* df_encode['smoker_yes']
df_encode['age_sex_male'] = df_encode['age']* df_encode['sex_male']

df_encode.shape
```




    (1338, 15)



## Train Test split
Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. It could over fit and as result our prediction on the test set would be poor. The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than error on any other dataset.


```python
from sklearn.model_selection import train_test_split
X = df_encode.drop('charges',axis=1) # Independet variable
y = df_encode['charges'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)
```

## Model building
In this step build model using our linear regression equation $\theta = (X^T X)^{-1} X^Ty$. In first step we need to add a feature $x_0 =1$ to our original data set. 


```python
# Step 1: add x0 =1 to dataset
X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

# Step2: build model
theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,y_train)) 
```


```python
# The parameters for linear regression model
parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
columns = ['intersect:x_0=1'] + list(X.columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})
```


```python
# Scikit Learn module
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

#Parameter
sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
parameter_df
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
      <th>Parameter</th>
      <th>Columns</th>
      <th>theta</th>
      <th>Sklearn_theta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>theta_0</td>
      <td>intersect:x_0=1</td>
      <td>6.767122</td>
      <td>6.767122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>theta_1</td>
      <td>age</td>
      <td>0.047182</td>
      <td>0.047182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>theta_2</td>
      <td>bmi</td>
      <td>0.012213</td>
      <td>0.012213</td>
    </tr>
    <tr>
      <th>3</th>
      <td>theta_3</td>
      <td>sex_male</td>
      <td>-0.265817</td>
      <td>-0.265817</td>
    </tr>
    <tr>
      <th>4</th>
      <td>theta_4</td>
      <td>children_1</td>
      <td>0.136285</td>
      <td>0.136285</td>
    </tr>
    <tr>
      <th>5</th>
      <td>theta_5</td>
      <td>children_2</td>
      <td>0.263954</td>
      <td>0.263954</td>
    </tr>
    <tr>
      <th>6</th>
      <td>theta_6</td>
      <td>children_3</td>
      <td>0.239013</td>
      <td>0.239013</td>
    </tr>
    <tr>
      <th>7</th>
      <td>theta_7</td>
      <td>children_4</td>
      <td>0.494028</td>
      <td>0.494028</td>
    </tr>
    <tr>
      <th>8</th>
      <td>theta_8</td>
      <td>children_5</td>
      <td>0.477927</td>
      <td>0.477927</td>
    </tr>
    <tr>
      <th>9</th>
      <td>theta_9</td>
      <td>smoker_yes</td>
      <td>2.783048</td>
      <td>2.783048</td>
    </tr>
    <tr>
      <th>10</th>
      <td>theta_10</td>
      <td>region_northwest</td>
      <td>-0.058535</td>
      <td>-0.058535</td>
    </tr>
    <tr>
      <th>11</th>
      <td>theta_11</td>
      <td>region_southeast</td>
      <td>-0.137235</td>
      <td>-0.137235</td>
    </tr>
    <tr>
      <th>12</th>
      <td>theta_12</td>
      <td>region_southwest</td>
      <td>-0.160339</td>
      <td>-0.160339</td>
    </tr>
    <tr>
      <th>13</th>
      <td>theta_13</td>
      <td>age**2</td>
      <td>-0.000123</td>
      <td>-0.000123</td>
    </tr>
    <tr>
      <th>14</th>
      <td>theta_14</td>
      <td>age_smoker_yes</td>
      <td>-0.031380</td>
      <td>-0.031380</td>
    </tr>
    <tr>
      <th>15</th>
      <td>theta_15</td>
      <td>age_sex_male</td>
      <td>0.005123</td>
      <td>0.005123</td>
    </tr>
  </tbody>
</table>
</div>



The parameter obtained from both the model are same.So we succefull build our model using normal equation and verified using sklearn linear regression module. Let's move ahead, next step is prediction and model evaluvation.

## Model evaluation
We will predict value for target variable by using our model parameter for test data set. Then compare the predicted value with actual valu in test set. We compute **Mean Square Error** using formula 
$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}_i - y_i)^2$$

$R^2$ is statistical measure of how close data are to the fitted regression line. $R^2$ is always between 0 to 100%. 0% indicated that model explains none of the variability of the response data around it's mean. 100% indicated that model explains all the variablity of the response data around the mean.

$$R^2 = 1 - \frac{SSE}{SST}$$
**SSE = Sum of Square Error**  
**SST = Sum of Square Total**  
$$SSE = \sum_{i=1}^{m}(\hat{y}_i - y_i)^2$$
$$SST = \sum_{i=1}^{m}(y_i - \bar{y}_i)^2$$
Here $\hat{y}$ is predicted value and $\bar{y}$ is mean value of $y$.


```python
# Normal equation
y_pred_norm =  np.matmul(X_test_0,theta)

#Evaluvation: MSE
J_mse = np.sum((y_pred_norm - y_test)**2)/ X_test_0.shape[0]

# R_square 
sse = np.sum((y_pred_norm - y_test)**2)
sst = np.sum((y_test - y_test.mean())**2)
R_square = 1 - (sse/sst)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse)
print('R square obtain for normal equation method is :',R_square)
```

    The Mean Square Error(MSE) or J(theta) is:  0.155323465604
    R square obtain for normal equation method is : 0.817197889059



```python
# sklearn regression module
y_pred_sk = lin_reg.predict(X_test)

#Evaluvation: MSE
from sklearn.metrics import mean_squared_error
J_mse_sk = mean_squared_error(y_pred_sk, y_test)

# R_square
R_square_sk = lin_reg.score(X_test,y_test)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse_sk)
print('R square obtain for scikit learn library is :',R_square_sk)
```

    The Mean Square Error(MSE) or J(theta) is:  0.155323465604
    R square obtain for scikit learn library is : 0.817197889059


The model returns $R^2$ value of 77.95%, so it fit our data test very well, but still we can imporve the the performance of by diffirent technique. Please make a note that we have transformer out variable by applying  natural log. When we put model into production antilog is applied to the equation.

## Backward elimination
In backward elimination we start with all the feature and remove the least significant feature at each iteration which improves the performance of the model. We repeate this until no improvement is observed on removal of features.


```python
# Backward elimination
import statsmodels.api as sm
X_train_0 = sm.add_constant(X_train)
X_test_0 = sm.add_constant(X_test)
ols = sm.OLS(endog=y_train, exog= X_train_0).fit()
print(ols.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                charges   R-squared:                       0.805
    Model:                            OLS   Adj. R-squared:                  0.802
    Method:                 Least Squares   F-statistic:                     271.0
    Date:                Thu, 06 Sep 2018   Prob (F-statistic):          9.98e-315
    Time:                        11:40:16   Log-Likelihood:                -482.19
    No. Observations:                 936   AIC:                             994.4
    Df Residuals:                     921   BIC:                             1067.
    Df Model:                          14                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const                6.9339      0.089     77.741      0.000       6.759       7.109
    age                  0.0374      0.001     26.165      0.000       0.035       0.040
    bmi                  0.0120      0.002      5.340      0.000       0.008       0.016
    sex_male            -0.2677      0.080     -3.348      0.001      -0.425      -0.111
    children_1           0.1526      0.034      4.485      0.000       0.086       0.219
    children_2           0.2803      0.038      7.330      0.000       0.205       0.355
    children_3           0.2546      0.045      5.720      0.000       0.167       0.342
    children_4           0.5062      0.104      4.859      0.000       0.302       0.711
    children_5           0.4984      0.108      4.613      0.000       0.286       0.710
    smoker_yes           2.7842      0.096     28.948      0.000       2.595       2.973
    region_northwest    -0.0586      0.038     -1.541      0.124      -0.133       0.016
    region_southeast    -0.1370      0.038     -3.597      0.000      -0.212      -0.062
    region_southwest    -0.1602      0.039     -4.119      0.000      -0.237      -0.084
    age_smoker_yes      -0.0314      0.002    -13.668      0.000      -0.036      -0.027
    age_sex_male         0.0052      0.002      2.720      0.007       0.001       0.009
    ==============================================================================
    Omnibus:                      531.247   Durbin-Watson:                   2.017
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3959.551
    Skew:                           2.561   Prob(JB):                         0.00
    Kurtosis:                      11.677   Cond. No.                         476.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The p-value for region_northwest is above significance level $\alpha$ =0.05. so we remove this variable.


```python
# Backward elimination
# Step 2
import statsmodels.api as sm
X_train_1 = X_train.drop('region_northwest',axis=1)
X_test_1 = X_test.drop('region_northwest',axis=1)
X_train_2 = sm.add_constant(X_train_1)
X_test_2 = sm.add_constant(X_test_1)
ols = sm.OLS(endog=y_train, exog= X_train_2).fit()
print(ols.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                charges   R-squared:                       0.804
    Model:                            OLS   Adj. R-squared:                  0.801
    Method:                 Least Squares   F-statistic:                     291.2
    Date:                Thu, 06 Sep 2018   Prob (F-statistic):          1.87e-315
    Time:                        11:40:25   Log-Likelihood:                -483.39
    No. Observations:                 936   AIC:                             994.8
    Df Residuals:                     922   BIC:                             1063.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const                6.9028      0.087     79.395      0.000       6.732       7.073
    age                  0.0374      0.001     26.153      0.000       0.035       0.040
    bmi                  0.0121      0.002      5.371      0.000       0.008       0.017
    sex_male            -0.2661      0.080     -3.326      0.001      -0.423      -0.109
    children_1           0.1511      0.034      4.441      0.000       0.084       0.218
    children_2           0.2780      0.038      7.271      0.000       0.203       0.353
    children_3           0.2526      0.045      5.673      0.000       0.165       0.340
    children_4           0.5089      0.104      4.881      0.000       0.304       0.713
    children_5           0.4998      0.108      4.623      0.000       0.288       0.712
    smoker_yes           2.7852      0.096     28.937      0.000       2.596       2.974
    region_southeast    -0.1086      0.033     -3.256      0.001      -0.174      -0.043
    region_southwest    -0.1315      0.034     -3.849      0.000      -0.198      -0.064
    age_smoker_yes      -0.0314      0.002    -13.652      0.000      -0.036      -0.027
    age_sex_male         0.0051      0.002      2.702      0.007       0.001       0.009
    ==============================================================================
    Omnibus:                      529.594   Durbin-Watson:                   2.022
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3912.168
    Skew:                           2.555   Prob(JB):                         0.00
    Kurtosis:                      11.614   Cond. No.                         475.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Model Validation
In oreder to validated model we need to check few assumption of linear regression model. The common assumption for *Linear Regression* model are following
1. Linear Relationship: In linear regression the relationship between the dependent and independent variable to be *linear*. This can be checked by scatter ploting Actual value Vs Predicted value
2. The residual error plot should be *normally* distributed.
3. The *mean* of *residual error* should be 0 or close to 0 as much as possible
4. The linear regression require all variables to be multivariate normal. This assumption can best checked with Q-Q plot.
5. Linear regession assumes that there is little or no *Multicollinearity in the data. Multicollinearity occurs when the independent variables are too highly correlated with each other. The variance inflation factor *VIF* identifies correlation between independent variables and strength of that correlation. $VIF = \frac {1}{1-R^2}$, If VIF >1 & VIF <5 moderate correlation, VIF < 5 critical level of multicollinearity.
6. Homoscedasticity: The data are homoscedastic meaning the residuals are equal across the regression line. We can look at residual Vs fitted value scatter plot. If heteroscedastic plot would exhibit a funnel shape pattern.


```python
# Prediction for ols model
y_pred_ols = ols.predict(X_test_2)
```


```python
# Check for Linearity
f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test,y_pred_ols,ax=ax,color=['r','g'])
plt.title('Check for Linearity')
plt.xlabel('Actual value')
plt.ylabel('Predicted value')

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test - y_pred_ols),ax=ax,color='b')
plt.axvline((y_test - y_pred_ols).mean(),color='k',linestyle='--')
plt.title('Check for Residual normality & mean')
plt.xlabel('Residual eror')
plt.ylabel('$p(x)$');
```


![png](output_53_0.png)



```python
# Check for Multivariate Normality
# Quantile-Quantile plot 
f,ax = plt.subplots(1,2,figsize=(14,6))
import scipy as sp
_,(_,_,r)= sp.stats.probplot((y_test - y_pred_ols),fit=True,plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')

#Check for Homoscedasticity
sns.scatterplot(y = (y_test - y_pred_ols), x= y_pred_ols, ax = ax[1],color=['r','g']) 
ax[1].set_title('Check for Homoscedasticity')
plt.xlabel('Predicted value')
plt.ylabel('Residual error');
```


![png](output_54_0.png)



```python
# Check for Multicollinearity
#Variance Inflation Factor
VIF = 1/(1- ols.rsquared)
VIF
```




    5.1057746936298756



The model assumption linear regression as follows
1. In our model  the actual vs predicted plot is not linear curve, so linear assumption fails
2. The residual mean is zero and residual error plot right skewed
3. Q-Q plot shows as value log value greater than 1.5 trends to increase
4. The plot is exhibit heteroscedastic, error will insease after certian point.
5. Variance inflation factor value is greater than 5, so bit of multicollearity.

# Thank you for visiting
