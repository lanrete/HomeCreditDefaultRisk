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
- To use `feature_importance` from model outputs, we don't need to impute the missing value since Light-GBM can take 
care of missing values by design. But we can't directly use it into the Pipeline, or I hadn't figure out how.
- <del>Genearte features from `previous_application.csv`.</del>
- Feature selection should be performed at different levels. When aggregating the datasets, mutiple features will be 
generated for one single columns. For example, the `min`, `max` and `mean` value will be generated for `AMT_ANNUITY` 
at `SK_ID_CURR` level for dataset `previous_application.csv`. Only one feature should be kept to reduce overfitting
since they all carried similar information. We only need to find the one with highest predictive power.
To do so, we could try:
  - Use `chi2`, `iv` or similar metrics to evalute predictive power. Need to take
    care of missing value in this way.
  - Use `feature_importance` from Light-GBM. Don't need to take care of missing value this way.


### Current Score & Location on LB

_Standing are by the time of submission._

|Submission     |Local AUC|LB AUC|Standing  |Pipeline                   |
|:-------------:|:-------:|:----:|:--------:|:-------------------------:|
|1_20180616_1455|0.7548   |0.745 |1880/2630 |LightGBM                   |
|with_bureau    |0.7651   |0.753 |1953/2906 |LightGBM with bureau       |
|20180621_121117|0.7748   |0.775 |1616/3051 |LightGBM                   |
|20180622_181112|0.7750   |0.776 |1635/3134 |LightGBM + FeatureSelection|


### 2018-06-16

- Explored the base dataset `application_train.csv`.
- Built a simple model as baseline with Light-GBM
- `AUC = 0.7548` on local testing set
- `AUC = 0.745` on Public leaderboard


### 2018-06-19

- Check the share [kernel](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features) by aguiar, 
    the structure seems very clear and could be use within this my own repository.
- Build features from `buearu.csv` and use these features into current pipeline
- `AUC = 0.7651` on local testing set
- `AUC = 0.753` on Public leaderboard

### 2018-06-20

- Adding some basic features generated from the base dataset. Including:
  1. The percentage of ammunity amount in the total income
  2. The payment ratio, or how soon the payment will end
  3. The percentage of total income in the credit amount
- Applied the opt bayes parameters as the hyper-parameters for Light-GBM as starting point
- Best hyper-parameters:
   - __Learning_rate__: 0.01
   - __n_estimators__: 2500
- `AUC = 0.7748` on local testing set
- `AUC = 0.8421` on whole training set
- `AUC = 0.775` on Public leaderboard

We observerd a somewhat heavy over-fitting issues. This could be caused by 
1. Too much estiamtors in the model itself.
2. Too much features.

We need to implement the feature selection framework to reduce the over-fitting issues. 
Right now we only have features from `bureau.csv` and `application_{train|test}.csv`, in future
we will have much more features and much worse over-fitting if no feature selection is applied.


### 2018-06-22

- Write the `ModelBasedFeatureSelector` to select features according to the feature importances from a
  rough model.
- With LGBM as the model for feature selections, use GridSearchCV to find the best number of features to keep
- Reduce the over-fitting problem. But still exist especially if we use the hyper parameters from bayes_opt
- `AUC = 0.7750` on local testing set
- `AUC = 0.776` on PUblic leaderboard


### 2018-06-27

- Explore `previous_application.csv` on previous application.
- For flag features, the dataset is not consistent, meaning some features use `Y` / `N` as the flag, while other use `0` and `1` as the flag. This needs be standardized before aggregating the features.
- For all features (across all dataset), `XAP` and `XNA` are considered as missing.
- For date related features (across all dataset), `365243` should be considered as `np.inf`.


### 2018-06-29

- Refactor the EAD part on `previous_application.csv`, moving the data-cleaning part into production script.
- Data-cleaning includes
  0. Converting numerical flags into categorical flags
  0. Converting missing flag such as `XAP` and `XNA` into `np.nan`
  0. Converting `365243` into `np.nan` for days features
- Adding features into current pipeline, model was fitted roughtly without `GridSearch` for initial result
- `AUC = 0.8115` on whole training set
- `AUC = 0.775` on Public leaderboard
- Do notice a reduce in previous over-fitting issue
