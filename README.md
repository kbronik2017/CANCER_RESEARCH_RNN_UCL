
[![GitHub issues](https://img.shields.io/github/issues/kbronik2017/Machine_Learning_Cancer_Research_UCL)](https://github.com/kbronik2017/Machine_Learning_Cancer_Research_UCL/issues)
[![GitHub forks](https://img.shields.io/github/forks/kbronik2017/Machine_Learning_Cancer_Research_UCL)](https://github.com/kbronik2017/Machine_Learning_Cancer_Research_UCL/network)
[![GitHub stars](https://img.shields.io/github/stars/kbronik2017/Machine_Learning_Cancer_Research_UCL)](https://github.com/kbronik2017/Machine_Learning_Cancer_Research_UCL/stargazers)
[![GitHub license](https://img.shields.io/github/license/kbronik2017/Machine_Learning_Cancer_Research_UCL)](https://github.com/kbronik2017/Machine_Learning_Cancer_Research_UCL/blob/master/LICENSE)

# Machine learning and Artificial Intelligence for time series analysis


# Table of contents
1. [Introduction](#introduction)
2. [RNN](#paragraph1)
    1. [Bidirectional LSTM-RNN](#subparagraph1)
3. [Plot_ElasticNet](#paragraph2)
4. [Plot_Lasso](#paragraph3)
5. [Plot_LogisticRegression](#paragraph4)
6. [Plot_RandomForestClassifier](#paragraph5)
7. [Plot_Lowess](#paragraph6)
 1. [ADC](#subparagraph6)
 2. [T2W](#subparagraph6)
8. [Running the Python codes!](#paragraph7)

## Introduction <a name="introduction"></a>

Machine learning algorithms to analyze serial multi-dimensional data to predict prostate cancer progression

## RNN <a name="paragraph1"></a>

### Bidirectional LSTM-RNN <a name="subparagraph1"></a>
<br>
 <img height="540" src="images/Bidirectional_LSTM_RNN.jpg"/>
</br>

## Plot_ElasticNet <a name="paragraph2"></a>
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

Linear regression with combined L1 and L2 priors as regularizer.

Minimizes the objective function:

1 / (2 * n_samples) * ||y - Xw||^2_2
+ alpha * l1_ratio * ||w||_1
+ 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
If you are interested in controlling the L1 and L2 penalty separately, keep in mind that this is equivalent to:

a * ||w||_1 + 0.5 * b * ||w||_2^2
where:

alpha = a + b and l1_ratio = a / (a + b)
The parameter l1_ratio corresponds to alpha in the glmnet R package while alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha.


## Plot_Lasso <a name="paragraph3"></a>
The second paragraph text
## Plot_LogisticRegression <a name="paragraph4"></a>
The second paragraph text
## Plot_RandomForestClassifier <a name="paragraph5"></a>
The second paragraph text
## Plot_Lowess <a name="paragraph6"></a>
The second paragraph text


## Running the Python codes!<a name="paragraph7"></a> 

First, user needs to install Anaconda https://www.anaconda.com/

Then


```sh
  - conda env create -f train_test_environment.yml
  or
  - conda create --name idp --file clone-file.txt
``` 
and 

```sh
  - conda activate idp
``` 
finally

```sh
  - python  (any of the above python code).py
``` 

