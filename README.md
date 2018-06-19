# HomeCreditDefaultRisk

## Project Background

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, 
this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and 
safe borrowing experience. In order to make sure this underserved population has a positive loan experience, 
Home Credit makes use of a variety of alternative data--including telco and transactional information--to 
predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, 
they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure 
that clients capable of repayment are not rejected and that loans are given with a principal, maturity, 
and repayment calendar that will empower their clients to be successful.

## Source Data

Data is stored in `\data` folder. Files are not uploaded to github since they're too large. To get the zip file, use below
kaggle API
```
kaggle competitions download -c home-credit-default-risk
```
## Status

### To-do & Diffculties

- Understand the different data structure and key columns.
- Too many features if we use all dataset, will need to select the features, to do that we have two methods. One is to use the scikit-learn built-in feature selection method. The other one is to use `feature_importance` from model outputs.
- For scikit-learn method, we need to impute the missing values in a logical way, since scikit-learn framework doesn't work with missing values.
- To use `feature_importance` from model outputs, we don't need to impute the missing value since Light-GBM can take care of missing values by design. But we can't directly use it into the Pipeline, or I hadn't figure out how.


### Current Score & Location on LB

_Standing are by the time of submission._

|Submission     |Local AUC|LB AUC|Standing  |Pipeline            |
|:-------------:|:-------:|:----:|:--------:|:------------------:|
|1_20180616_1455|0.7548   |0.745 |1880/2630 |LightGBM            |
|with_bureau    |0.7651   |0.753 |1953/2906 |LightGBM with bureau|


### 2018-06-16

- Explored the base dataset `application_train.csv`.
- Built a simple model as baseline with Light-GBM
- `AUC = 0.7548` on local testing set
- `AUC = 0.745` on Public leaderboard


### 2018-06-19

- Check the share [kernel](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features) by aguiar, the structure seems very clear and could be use within this my own repository.
- Build features from `buearu.csv` and use these features into current pipeline
- `AUC = 0.7651` on local testing set
- `AUC = 0.753` on Public leaderboard

