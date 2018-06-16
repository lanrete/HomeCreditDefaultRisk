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

### Current Score & Location on LB

_Standing are by the time of submission._

|Submission     |Local AUC|PB AUC|Standing  |Pipeline|
|---------------|:-------:|:----:|:--------:|:------:|
|1_20180616_1455|0.7548   |0.745 |1880/2630 |LightGBM|


### 2018-06-16

- Explored the base dataset `application_train.csv`.
- Built a simple model as baseline with Light-GBM
- `AUC = 0.7548` on local testing set
- `AUC = 0.745` on PB