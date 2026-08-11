# WIP - propensity_ml_project

* XGBoost
* LGBM
* Decision Tree
* Regularized Logistic Regression - ElasticNet (L1+L2)

# How to use it? - propensity_ml_project

* Create a table on DataBricks - Unity Catalog or add to a volumen the dataset provided. Currently we assume a post-feature engineering dataset is passed as input for the pipeline.

* Change the dataset new path in block:

  ```python
path = "your path here"
df   = spark.table(path).toPandas()
