# Lime Water Example: Regression
# Regression Example: Boston Housing

library(mlbench) # for dataset
data("BostonHousing")
dim(BostonHousing)

# Define features
features = setdiff(colnames(BostonHousing), "medv")
features

# Pick four random samples for test dataset
set.seed(1234)
row_test_samp = sample(1:nrow(BostonHousing), 4)

# Train
x_train = BostonHousing[-row_test_samp, features]
y_train = BostonHousing[-row_test_samp, "medv"]

# Test
x_test = BostonHousing[row_test_samp, features]
y_test = BostonHousing[row_test_samp, "medv"]


library(caret) # ML framework
library(doParallel) # parallelisation

# Train a Random Forest using caret
cl = makePSOCKcluster(8)
registerDoParallel(cl)
set.seed(1234)
model_rf =
  caret::train(
    x = x_train,
    y = y_train,
    method = "rf",
    tuneLength = 3,
    trControl = trainControl(method = "cv")
  )
stopCluster(cl)

# Print model summary
model_rf

# Using the Random Forest model to make predictions on test set
yhat_test = predict(model_rf, x_test)

# Create a new data frame to compare target (medv) and predictions
d_test = data.frame(x_test,
                    medv = y_test,
                    predict = yhat_test,
                    row.names = NULL)


# RF: LIME Steps 1 and 2
# Step 1: Create an 'explainer' object using training data and model
explainer = lime::lime(x = x_train, model = model_rf)

# Step 2: Turn 'explainer' into 'explainations' for test set
explainations = lime::explain(x = x_test,
                              explainer = explainer,
                              n_permutations = 5000,
                              feature_select = "auto",
                              n_features = 5)

# RF: LIME Explainations
head(explainations, 5) #LIME Pred: 36.59, Random Forest Pred: 31.64, R^2 = 0.65


# RF: LIME Visualisation
# Step 3: Visualise explainations
lime::plot_features(explainations, ncol = 4)




# H2O AutoML
# Start a local H2O cluster (JVM)
library(h2o)
h2o.init(nthreads = -1)

# Prepare H2O Data Frames
h_train = as.h2o(BostonHousing[-row_test_samp,])
h_test = as.h2o(BostonHousing[row_test_samp,])

# Train multiple H2O models with a simple API
# Stacked Ensembles will be created from those H2O models
# You tell H2O 1) how much time you have and/or 2) how many models do you want
model_automl = h2o.automl(x = features,
                          y = "medv",
                          training_frame = h_train,
                          nfolds = 5,
                          max_runtime_secs = 120, # time #<<
                          max_models = 20,        # max models #<<
                          stopping_metric = "RMSE",
                          seed = 1234)

# H2O: AutoML Model Leaderboard
# Print out leaderboard
model_automl@leaderboard

# H2O: Model Leader
# Best Model (either an individual model or a stacked ensemble)
model_automl@leader

# Using the best model to make predictions on test set
yhat_test = h2o.predict(model_automl@leader, h_test)

# Create a new data frame to compare target (medv) and predictions
d_test = data.frame(x_test,
                    medv = y_test,
                    predict = as.data.frame(yhat_test),
                    row.names = NULL)



# H2O: LIME Steps 1 and 2
# Step 1: Create an 'explainer' object using training data and model
explainer = lime::lime(x = as.data.frame(h_train[, features]),
                       model = model_automl@leader)

# Step 2: Turn 'explainer' into 'explainations' for test set
explainations = lime::explain(x = as.data.frame(h_test[, features]),
                              explainer = explainer,
                              n_permutations = 5000,
                              feature_select = "auto",
                              n_features = 5) # look at top 5 features only

# H2O: LIME Explainations
head(explainations, 5)

# H2O: LIME Visualisation
# Step 3: Visualise explainations
lime::plot_features(explainations, ncol = 2)
