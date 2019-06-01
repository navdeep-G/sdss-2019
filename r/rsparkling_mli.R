# Interpretable Machine Learning with rsparkling:

# R packages you'll need to install up front if you don't already have them:
#
#install.packages("digest")
#install.packages("devtools")
#install.packages("dplyr")
#install.packages("data.tree") # For surrogate models
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

# Set up input and output for modelling
y <- "DEFAULT_PAYMENT_NEXT_MONTH"
x <- setdiff(names(train_hex), c("ID", y)) # Do not use ID column

# Train an H2O Gradient Boosting Machine (GBM)
fit_gbm <- h2o.gbm(x = x,
               y = y,
               ntrees = 150, # maximum 150 trees in GBM
               max_depth = 4, # trees can have maximum depth of 4
               sample_rate = 0.9, # use 90% of rows in each iteration (tree)
               col_sample_rate = 0.9, # use 90% of variables in each iteration (tree)
               stopping_rounds = 5,  # stop if validation error does not decrease for 5 iterations (trees)
               score_tree_interval = 1,  # for reproducibility, set higher for bigger data
               training_frame = splits[[1]], # training frame
               validation_frame = splits[[2]], # validation frame
               seed = 12345 # seed for reproducibility,
               ) 

# Evaluate model performance on validation set:
h2o.performance(fit_gbm, valid = TRUE)

# As a comparison, we can evaluate performance on a test set
h2o.performance(fit_gbm, newdata = test_hex)

# Now, generate the predictions (as opposed to metrics)
pred_gbm_hex <- h2o.predict(fit_gbm, newdata = test_hex)
pred_gbm_hex

# If we want these available in Spark:
pred_gbm_sdf <- as_spark_dataframe(sc, pred_gbm_hex)
pred_gbm_sdf

######################################################################################################################
# H2O-3 machine learning interpretability (MLI):
#
# Obtain and display global variable importance
# 
# During training, the h2o GBM aggregates the improvement in error caused by each split in each 
# decision tree across all the decision trees in the ensemble classifier. These values are 
# attributed to the input variable used in each split and give an indication of the contribution 
# each input variable makes toward the model's predictions. The variable importance ranking should 
# be parsimonious with human domain knowledge and reasonable expectations. 
# In this case, a customer's most recent payment behavior, PAY_0, is by far the most important variable 
# followed by their second most recent payment, PAY_2, and third most recent payment, PAY_3, behavior. 
# This result is well-aligned with business practices in credit lending: people who miss their most recent 
# payments are likely to default soon.
h2o.varimp_plot(fit_gbm)

# Train a decision tree surrogate model to describe GBM
# 
# A surrogate model is a simple model that is used to explain a complex model. One of the original references 
# for surrogate models is available here: https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf.
# In this example, a single decision tree will be trained on the original inputs and predictions of the h2o
# GBM model and the tree will be visualized. The variable importance, interactions, and decision paths displayed in 
# the directed graph of the trained decision tree surrogate model are then assumed to be indicative of the 
# internal mechanisms of the more complex GBM model, creating an approximate, overall flowchart for the GBM.
# There are few mathematical guarantees that the simple surrogate model is highly representative of the more 
# complex GBM, but a recent preprint article has put forward ideas on strenghthening the theoretical relationship 
# between surrogate models and more complex models: https://arxiv.org/pdf/1705.08504.pdf.

# Bind predictions from test set to test_hex
y_surrogate <- "p1"
x_surrogate <- setdiff(names(train_hex), c("ID", y, "DEFAULT_PAYMENT_NEXT_MONTH")) # Do not use ID column
test_hex_yhat <- h2o.cbind(test_hex, pred_gbm_hex$p1)

rf_surrogate <- h2o.randomForest(x=x_surrogate,
                                 y=y_surrogate,
                                 ntrees=1,          # use only one tree
                                 sample_rate=1,     # use all rows in that tree
                                 mtries=-2,         # use all columns in that tree
                                 max_depth=3,       # shallow trees are easier to understand
                                 seed=12345,        # random seed for reproducibility,
                                 training_frame = test_hex_yhat
)
source("viz_tree.R")
rf_h2o_surrogate_tree = h2o.getModelTree(model = rf_surrogate, tree_number = 1)
rf_data_tree = createDataTree(rf_h2o_surrogate_tree)
GetEdgeLabel <- function(node) {return (node$edgeLabel)}
GetNodeShape <- function(node) {switch(node$type, 
                                       split = "diamond", leaf = "oval")}
GetFontName <- function(node) {switch(node$type, 
                                      split = 'Palatino-bold', 
                                      leaf = 'Palatino')}
SetEdgeStyle(rf_data_tree, fontname = 'Palatino-italic', 
             label = GetEdgeLabel, labelfloat = TRUE,
             fontsize = "26", fontcolor='royalblue4')
SetNodeStyle(rf_data_tree, fontname = GetFontName, shape = GetNodeShape, 
             fontsize = "26", fontcolor='royalblue4',
             height="0.75", width="1")

SetGraphStyle(rf_data_tree, rankdir = "LR", dpi=70.)
plot(rf_data_tree, output = "graph")

# Calculating partial dependence and ICE to validate and explain monotonic behavior
# 
# Partial dependence plots are used to view the global, average prediction behavior of a variable under 
# the monotonic model. Partial dependence plots show the average prediction of the monotonic model as a 
# function of specific values of an input variable of interest, indicating how the monotonic GBM predictions 
# change based on the values of the input variable of interest, while taking nonlinearity into consideration 
# and averaging out the effects of all other input variables. Partial dependence plots enable increased 
# transparency into the monotonic GBM's mechanisms and enable validation and debugging of the monotonic GBM 
# by comparing a variable's average predictions across its domain to known standards and reasonable expectations. 
# Partial dependence plots are described in greater detail in The Elements of Statistical Learning, 
# section 10.13: https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf.
#
# Individual conditional expectation (ICE) plots, a newer and less well-known adaptation of partial dependence plots, 
# can be used to create more localized explanations for a single observation of data using the same basic ideas as 
# partial dependence plots. ICE is also a type of nonlinear sensitivity analysis in which the model predictions for 
# a single observation are measured while a feature of interest is varied over its domain. ICE increases understanding
# and transparency by displaying the nonlinear behavior of the monotonic GBM. ICE also enhances trust, accountability,
# and fairness by enabling comparisons of nonlinear behavior to human domain knowledge and reasonable expectations. 
# ICE, as a type of sensitivity analysis, can also engender trust when model behavior on simulated or extreme data
# points is acceptable. A detailed description of ICE is available in this arXiv preprint: https://arxiv.org/abs/1309.6392.
#
# Because partial dependence and ICE are measured on the same scale, they can be displayed in the same line plot to 
# compare the global, average prediction behavior for the entire model and the local prediction behavior for certain 
# rows of data. Overlaying the two types of curves enables analysis of both global and local behavior simultaneously
# and provides an indication of the trustworthiness of the average behavior represented by partial dependence. 
# (Partial dependence can be misleading in the presence of strong interactions or correlation.
# ICE curves diverging from the partial dependence curve can be indicative of such problems.) 
# Histograms are also presented with the partial dependence and ICE curves, to enable a rough measure of 
# epistemic uncertainty for model predictions: predictions based on small amounts of training data are 
# likely less dependable.

# PDP for variable "PAY_0"
pdp_gbm_pay0 <- h2o.partialPlot(fit_gbm, test_hex, cols = "PAY_0")

# ICE
source("ice.R")
# ICE for first row in test set and "PAY_0"
ice_pay0_row1_frame <- get_ice_frame(test_hex[1,], h2o.unique(test_hex$PAY_0), "PAY_0")
ice_gbm_pay0_row1 <- h2o.partialPlot(fit_gbm, ice_pay0_row1_frame, cols = "PAY_0")

# Generate reason codes using the Shapley method
# 
# Shapley explanations will be used to calculate the local variable importance for any one prediction: 
# http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions. 
# Shapley explanations are the only possible consistent local variable importance values. (Here consistency means 
# that if a variable is more important than another variable in a given prediction, the more important 
# variable's Shapley value will not be smaller in magnitude than the less important variable's Shapley value.) 
# Very crucially Shapley values also always sum to the actual prediction of the H2O GBM/XGBoost model. 
# When used in a model-specific context for decision tree models, Shapley values are likely the
# most accurate known local variable importance method available today. 
#
# The numeric Shapley values in each column are an estimate of how much each variable contributed to each prediction. 
# Shapley contributions can indicate how a variable and its values were weighted in any given decision by the model. 
# These values are crucially important for machine learning interpretability and are related to "local feature 
# importance", "reason codes", or "turn-down codes." The latter phrases are borrowed from credit scoring.
# Credit lenders in the U.S. must provide reasons for automatically rejecting a credit application. 
# Reason codes can be easily extracted from Shapley local variable contribution values by ranking the 
# variables that played the largest role in any given decision.

# You should be able to use the function `predict_contributions` to explain the predictions of the GBM model:
contributions <- h2o.predict_contributions(fit_gbm, test_hex)

# Some checks to ensure Shapley matches the predictions
# Helper function to ensure sum of Shapley match prediction after applying inverse link function for binomial 
# classification
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Check if sum of Shapley values after sigmoid transform match predicted outcome for same test set
p1_using_contributions <- sigmoid(as.data.frame(h2o.sum(contributions, axis=1, return_frame = T)))
head(pred_gbm_hex$p1)
head(p1_using_contributions)

# Train interpretable model(s) with H2O-3
# 
# The previous methods are quite helpful in interpreting complex models (tree based models, neural nets, etc.).
# However, one can still rely on directly interpretable models for their machine learning endeavors (GLM, GAM, etc.)
# Below we show a way to build an interpretable model in H2O-3

# Train an H2O Generalized Linear Model (GLM)
fit_glm <- h2o.glm(x = x,
                   y = y,
                   family = "binomial", # need to set "family" for H2O-3 GLM, which is "binomial" for classification
                   training_frame = splits[[1]], # training frame
                   validation_frame = splits[[2]], # validation frame
                   seed = 12345 # seed for reproducibility,
)

# Evaluate model performance on validation set:
h2o.performance(fit_glm, valid = TRUE)

# As a comparison, we can evaluate performance on a test set
h2o.performance(fit_glm, newdata = test_hex)

# Now, generate the predictions (as opposed to metrics)
pred_glm_hex <- h2o.predict(fit_glm, newdata = test_hex)
pred_glm_hex

# If we want these available in Spark:
pred_glm_sdf <- as_spark_dataframe(sc, pred_glm_hex)
pred_glm_sdf

# We can still look at PDP/ICE for GLM to view model behavior globally (PDP) and locally (ICE)

# PDP for variable "PAY_0"
pdp_glm_pay0 <- h2o.partialPlot(fit_glm, test_hex, cols = "PAY_0")

# ICE
source("ice.R")
# ICE for first row in test set and "PAY_0"
ice_pay0_row1_frame <- get_ice_frame(test_hex[1,], h2o.unique(test_hex$PAY_0), "PAY_0")
ice_glm_pay0_row1 <- h2o.partialPlot(fit_glm, ice_pay0_row1_frame, cols = "PAY_0")

# Inspect Spark log directly
spark_log(sc, n = 20)

# Now we disconnect from Spark, this will result in the H2OContext being stopped as
# well since it's owned by the spark shell process used by our Spark connection:
spark_disconnect(sc)
