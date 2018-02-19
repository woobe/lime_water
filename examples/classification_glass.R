# Lime Water Example: Classification
# Classification Example: Glass

library(mlbench) # for dataset
data("Glass")

# Rename columns
colnames(Glass) = c("Refractive_Index", "Sodium", "Magnesium", "Aluminium",
                    "Silicon", "Potassium", "Calcium", "Barium", "Iron", "Type")
dim(Glass)
str(Glass)

# Define Features
features = setdiff(colnames(Glass), "Type")
features

# Pick four random samples for test dataset
set.seed(1234)
row_test_samp = sample(1:nrow(Glass), 4)

# H2O AutoML
# Start a local H2O cluster (JVM)
library(h2o)
h2o.init(nthreads = -1)

# Prepare H2O Data Frames
h_train = as.h2o(Glass[-row_test_samp,])
h_test = as.h2o(Glass[row_test_samp,])

# Train multiple H2O models with a simple API
# Stacked Ensembles will be created from those H2O models
# You tell H2O 1) how much time you have and/or 2) how many models do you want
model_automl = h2o.automl(x = features,
                          y = "Type",
                          training_frame = h_train,
                          nfolds = 5,
                          max_runtime_secs = 120, # time #<<
                          max_models = 20,        # max models #<<
                          stopping_metric = "mean_per_class_error",
                          seed = 1234)

# H2O: AutoML Model Leaderboard
# Print out leaderboard
model_automl@leaderboard

# H2O: Model Leader
# Best Model (either an individual model or a stacked ensemble)
model_automl@leader

# H2O: Making Prediction
# Using the best model to make predictions on test set
yhat_test = h2o.predict(model_automl@leader, h_test)
head(yhat_test)


# H2O: LIME Steps 1 and 2
# Step 1: Create an 'explainer' object using training data and model
explainer = lime::lime(x = as.data.frame(h_train[, features]),
                       model = model_automl@leader)

# Step 2: Turn 'explainer' into 'explainations' for test set
explainations = lime::explain(x = as.data.frame(h_test[, features]),
                              explainer = explainer,
                              n_permutations = 5000,
                              feature_select = "auto",
                              n_labels = 1, # Explain top prediction only #<<
                              n_features = 5)

# H2O: LIME Visualisation
# Step 3: Visualise explainations
lime::plot_features(explainations, ncol = 2)
