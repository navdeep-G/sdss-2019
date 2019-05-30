# Interpretable Machine Learning with rsparkling:

# R packages you'll need to install up front if you don't already have them:
#
#install.packages("digest")
#install.packages("devtools")
#install.packages("dplyr")
#
# Install the latest version of h2o:
# The following two commands remove any previously installed H2O packages for R.
#if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
#if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
#pkgs <- c("RCurl","jsonlite")
#for (pkg in pkgs) {
#  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
#}

# Now we download, install and initialize the H2O package for R.
#install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-yates/3/R")
#
# Install rsparkling and sparklyr either from Github:
# devtools::install_github("h2oai/sparkling-water", subdir = "/r/src")
# devtools::install_github("rstudio/sparklyr")
# or from CRAN
#install.packages("sparklyr")
#install.packages("rsparkling")

library(sparklyr)
library(h2o)

# Set version of sparkling water and location of sparkling water jar to use latest 
# Note: At the time of spark_connect sparklyr will call the spark_dependencies function in the rsparkling package.
options(rsparkling.sparklingwater.version = "2.4.11")

library(rsparkling)

# If you don't already have it installed, Spark can be installed via the sparklyr command:
spark_install(version = "2.4.0")

# Create a spark connection
sc <- spark_connect(master = "local", version = "2.4.0")

# Inspect the H2OContext for our Spark connection
# This will also start an H2O cluster
h2o_context(sc)

# We can also view the H2O Flow web UI:
# h2o_flow(sc)

# H2O with Spark DataFrames

# Read in creditcard train dataset to Spark as an example:
library(dplyr)
train_tbl <- spark_read_csv(sc, "../data/creditcard-train.csv")
test_tbl <- spark_read_csv(sc, "../data/creditcard-test.csv")

# Convert the Spark DataFrame's into an H2OFrame's
train_hex <- as_h2o_frame(sc, train_tbl)
test_hex <- as_h2o_frame(sc, test_tbl)

# Modify columns for train
train_hex$SEX = as.factor(train_hex$SEX)
train_hex$MARRIAGE = as.factor(train_hex$MARRIAGE)
train_hex$EDUCATION = as.factor(train_hex$EDUCATION)
train_hex$PAY_0 = as.factor(train_hex$PAY_0)
train_hex$PAY_2 = as.factor(train_hex$PAY_2)
train_hex$PAY_3 = as.factor(train_hex$PAY_3)
train_hex$PAY_4 = as.factor(train_hex$PAY_4)
train_hex$PAY_5 = as.factor(train_hex$PAY_5)
train_hex$PAY_6 = as.factor(train_hex$PAY_6)
train_hex$DEFAULT_PAYMENT_NEXT_MONTH = as.factor(train_hex$DEFAULT_PAYMENT_NEXT_MONTH)

# Modify columns for test
test_hex$SEX = as.factor(test_hex$SEX)
test_hex$MARRIAGE = as.factor(test_hex$MARRIAGE)
test_hex$EDUCATION = as.factor(test_hex$EDUCATION)
test_hex$DEFAULT_PAYMENT_NEXT_MONTH = as.factor(test_hex$DEFAULT_PAYMENT_NEXT_MONTH)
test_hex$PAY_0 = as.factor(test_hex$PAY_0)
test_hex$PAY_2 = as.factor(test_hex$PAY_2)
test_hex$PAY_3 = as.factor(test_hex$PAY_3)
test_hex$PAY_4 = as.factor(test_hex$PAY_4)
test_hex$PAY_5 = as.factor(test_hex$PAY_5)
test_hex$PAY_6 = as.factor(test_hex$PAY_6)

# Split the H2O Frame into train and validation
splits <- h2o.splitFrame(train_hex, ratios = 0.7, seed = 12345)
nrow(splits[[1]])  # nrows in train
nrow(splits[[2]])  # nrows in validation

# Train an H2O Gradient Boosting Machine (GBM)
# And perform 3-fold cross-validation via `nfolds`
y <- "DEFAULT_PAYMENT_NEXT_MONTH"
x <- setdiff(names(train_hex), c("ID", y)) # Do not use ID column
fit <- h2o.gbm(x = x,
               y = y,
               ntrees = 150, # maximum 150 trees in GBM
               max_depth = 4, # trees can have maximum depth of 4
               sample_rate = 0.9, # use 90% of rows in each iteration (tree)
               col_sample_rate = 0.9, # use 90% of variables in each iteration (tree)
               balance_classes = TRUE, # sample to balance 0/1 distribution of target
               stopping_rounds = 5,  # stop if validation error does not decrease for 5 iterations (trees)
               score_tree_interval = 1,  # for reproducibility, set higher for bigger data
               training_frame = splits[[1]], # training frame
               validation_frame = splits[[2]], # validation frame
               seed = 12345 # Seed for reproducibility,
               ) 

# Evaluate model performance on validation:
h2o.performance(fit, valid = TRUE)

# As a comparison, we can evaluate performance on a test set
h2o.performance(fit, newdata = test_hex)


# Now, generate the predictions (as opposed to metrics)
pred_hex <- h2o.predict(fit, newdata = test_hex)
pred_hex

# If we want these available in Spark:
pred_sdf <- as_spark_dataframe(sc, pred_hex)
pred_sdf


# Other useful functions:

# Inspect Spark log directly
spark_log(sc, n = 20)

# H2O-3 machine learning interpretability (MLI):
# PDP of model for PAY_0
pdp_pay_0 <- h2o.partialPlot(fit, test_hex, cols = "PAY_0")

# Shapley
shap <- h2o.predict_contributions(fit, newdata = test_hex)

# Now we disconnect from Spark, this will result in the H2OContext being stopped as
# well since it's owned by the spark shell process used by our Spark connection:
spark_disconnect(sc)
