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
h2o_flow(sc)


# H2O with Spark DataFrames

# Read in creditcard train dataset to Spark as an example:
library(dplyr)
train_tbl <- spark_read_csv(sc, "../data/CreditCard_Cat-train.csv")
test_tbl <- spark_read_csv(sc, "../data/CreditCard_Cat-test.csv")
train_tbl
test_tbl

# Convert the Spark DataFrame's into an H2OFrame's
train_hex <- as_h2o_frame(sc, train_tbl)
test_hex <- as_h2o_frame(sc, test_tbl)
train_hex
test_hex

# Split the H2O Frame into train & test sets
#splits <- h2o.splitFrame(train_hex, ratios = 0.7, seed = 1)
#nrow(splits[[1]])  # nrows in train
#nrow(splits[[2]])  # nrows in test

# Train an H2O Gradient Boosting Machine (GBM)
# And perform 3-fold cross-validation via `nfolds`
y <- "DEFAULT_PAYMENT_NEXT_MONTH"
x <- setdiff(names(train_hex), y)
fit <- h2o.gbm(x = x,
               y = y,
               training_frame = train_hex,
               nfolds = 3,
               min_rows = 1,
               seed = 1)

# Evaluate 3-fold cross-validated model performance:
h2o.performance(fit, xval = TRUE)

# As a comparison, we can evaluate performance on a test set
h2o.performance(fit, newdata = test_hex)

# Note: Since this is a very small data problem,
# we see a reasonable difference between CV and
# test set metrics


# Now, generate the predictions (as opposed to metrics)
pred_hex <- h2o.predict(fit, newdata = test_hex)
pred_hex

# If we want these available in Spark:
pred_sdf <- as_spark_dataframe(sc, pred_hex)
pred_sdf


# Other useful functions:

# Inspect Spark log directly
spark_log(sc, n = 20)


# Now we disconnect from Spark, this will result in the H2OContext being stopped as
# well since it's owned by the spark shell process used by our Spark connection:
spark_disconnect(sc)
