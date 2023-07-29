# Statistical Learning and Data Mining Challenge
# Third binary classification problem to discriminate Mild Cognitive Impairment vs. Control patients

# Libraries
library(purrr)
library(leaps)
library(cellWise)
library(rrcovHD)
library(mltools)
library(modelgrid)
library(rrcov)
library(caret)
library(magrittr)
library(dplyr)
library(recipes)

# Set seed for the reproducibility 
set.seed(42)

# Set the file directory
setwd("E:/Data Mining Challenge/")

# Read training data
train_data<- read.csv("MCICTLtrain.csv")

# Read Test data 
test_data<- read.csv("MCICTLtest.csv")

# Display the first few rows of the training data
head(train_data)

# Display the first few rows of the test data
head(test_data)

# Remove the patient ID column from the training data
train_data <- subset(train_data, select = -ID )

# Updated training data
head(train_data)

# Remove the patient ID column from the test data
test_ids = test_data$ID
test_data <- subset(test_data, select = -ID )

# Updated test data
head(test_data)

# Separating features from the labels in the training data
train_labels <- train_data$Label
train_predictors <- subset(train_data, select = 1:593)

# Display the first few rows of the training features
head(train_predictors)

# Checking the dimensions of the training features
dim(train_predictors)

# Encoding labels using a two level factor
train_data$Label <- factor(train_data$Label)
train_labels <- factor(train_labels)

# Checking if the data is balanced
summary(train_labels)

# Training and evaluation of the classification model

# Function to calculate the MCC scores 
calculate_mcc <- function (data, lev = NULL, model = NULL) {
  mcc_metric <- mcc(data$obs, data$pred)  
  names(mcc_metric) <- "MCC"
  mcc_metric
}

newSummary <- function(...) c(twoClassSummary(...), calculate_mcc(...))

# ------------------------------ LOGISTIC REGRESSION ------------------------------ #

# Feature selection methods to be used  
# 1. Recursive feature elimination
# 2. Removing correlated features
# 3. Principal component analysis

caretFuncs$summary <- twoClassSummary

rfe_ctrl <- rfeControl(functions=caretFuncs, method = "repeatedcv", number = 5, repeats =5)

train_ctrl <- trainControl(classProbs= TRUE, summaryFunction =  twoClassSummary) 

set.seed(42)

rfe_result <- rfe(train_predictors, 
                  train_labels,
                  sizes=c(1, 5, 10, 25, 50, 100, 250),
                  rfeControl= rfe_ctrl,
                  trControl = train_ctrl,
                  preProcess=c("center", "scale"), 
                  method = "glm", 
                  family = binomial(link = "logit")
)

# Analyzing the results 
rfe_result

# Display the RFE results
ggplot(data = rfe_result, metric = "ROC") + theme_bw()

# Create new train data with the features suggested by the recursive feature elimination method
optimal_preds <- append(predictors(rfe_result), 'Label')
optimal_train_data <- train_data[ ,optimal_preds]

# Create the pre-processing pipeline
initial_recipe <- recipe(train_data) %>%
  update_role(Label, new_role = "outcome") %>%
  update_role(-Label, new_role = "predictor") %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

# Pre-processing pipeline for new train data suggested by RFE
rfe_recipe <- recipe(optimal_train_data) %>%
  update_role(Label, new_role = "outcome") %>%
  update_role(-Label, new_role = "predictor") %>%
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())

# Models to be trained
models <- 
  model_grid() %>%
  share_settings(
    data = train_data,
    trControl = trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 5,
                             summaryFunction = newSummary,
                             classProbs = TRUE),
    metric = "ROC",
    method = "glm",
    family = binomial(link = "logit")
  )

models <- models %>%
  add_model(model_name = "baseline",
            x = initial_recipe)%>%
  add_model(model_name = "rfe",
            x = rfe_recipe) %>%
  add_model(model_name = "corr_.6",
            x = initial_recipe %>%
              step_corr(all_predictors(), threshold = .6)) %>%
  add_model(model_name = "corr_.7",
            x = initial_recipe %>%
              step_corr(all_predictors(), threshold = .7)) %>%
  add_model(model_name = "corr_.8",
            x = initial_recipe %>%
              step_corr(all_predictors(), threshold = .8)) %>%
  add_model(model_name = "pca_.75",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .75)) %>%
  add_model(model_name = "pca_.8",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .8)) %>%
  add_model(model_name = "pca_.85",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .85)) %>%
  add_model(model_name = "pca_.9",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .9)) %>%
  add_model(model_name = "pca_.95",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .95))

# Retrain and plot re-sampled results 
set.seed(777)
models <- train(models)
models$model_fits %>% caret::resamples(.) %>% bwplot(.)

models$model_fits %>%
  map(pluck(c("recipe", "term_info", "role"))) %>%
  map_int(~ sum(.x == "predictor"))

resamps <- caret::resamples(models$model_fits)
summary(resamps)

# --------------------------------- RANDOM FOREST --------------------------------- #

# Feature selection methods to be used  
# 1. Recursive feature elimination
# 2. Removing correlated features
# 3. Principal component analysis

caretFuncs$summary <- twoClassSummary

rfe_ctrl <- rfeControl(functions=caretFuncs, method = "repeatedcv", number = 5, repeats =5)

train_ctrl <- trainControl(classProbs= TRUE, summaryFunction = twoClassSummary) 

set.seed(777)

rfe_result <- rfe(train_predictors, 
                  train_labels,
                  sizes=c(1, 10, 20, 30, 40, 50, 60),
                  rfeControl= rfe_ctrl,
                  trControl = train_ctrl,
                  preProcess=c("center", "scale"),
                  method = "rf",
                  tuneLength = 3
)

# Analyzing the results
rfe_result

# Display the RFE results
ggplot(data = rfe_result, metric = "ROC") + theme_bw()


# Create new train data with the features suggested by the recursive feature elimination method 
optimal_preds <- append(predictors(rfe_result), 'Label')
optimal_train_data <- train_data[ ,optimal_preds]

# Create the pre-processing pipeline
initial_recipe <- recipe(train_data) %>%
  update_role(Label, new_role = "outcome") %>%
  update_role(-Label, new_role = "predictor") %>%
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())

# Pre-processing pipeline for new train data suggested by RFE
rfe_recipe <- recipe(optimal_train_data) %>%
  update_role(Label, new_role = "outcome") %>%
  update_role(-Label, new_role = "predictor") %>%
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())

# Models to be trained
models <- 
  model_grid() %>%
  share_settings(
    data = train_data,
    trControl = trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 5,
                             summaryFunction = newSummary,
                             classProbs = TRUE),
    metric = "ROC",
    method = "rf",
    tuneLength = 10
  )

models <- models %>%
  add_model(model_name = "baseline", 
            x = initial_recipe)  %>%
  add_model(model_name = "rfe", 
            x = rfe_recipe) %>%
  add_model(model_name = "corr_.6",
            x = initial_recipe %>%
              step_corr(all_predictors(), threshold = .6)) %>%
  add_model(model_name = "corr_.7",
            x = initial_recipe %>%
              step_corr(all_predictors(), threshold = .7)) %>%
  add_model(model_name = "corr_.8", 
            x = initial_recipe %>%
              step_corr(all_predictors(), threshold = .8)) %>%
  add_model(model_name = "pca_.75",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .75)) %>%
  add_model(model_name = "pca_.8",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .8)) %>%
  add_model(model_name = "pca_.85",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .85)) %>%
  add_model(model_name = "pca_.9",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .9)) %>%
  add_model(model_name = "pca_.95",
            x = initial_recipe %>%
              step_pca(all_predictors(), threshold = .95))



# Retrain and plot re-sampled results 
set.seed(777)
models <- train(models)
models$model_fits %>% caret::resamples(.) %>% bwplot(.)

models$model_fits %>%
  map(pluck(c("recipe", "term_info", "role"))) %>%
  map_int(~ sum(.x == "predictor"))

resamps <- caret::resamples(models$model_fits)
summary(resamps)

# --------------------------------- RESULTS --------------------------------- #

# Cross Validation Approach
train_samples <- train_data[1:150, ]
valid_samples <- train_data[151:172, ]

preds <- train_samples[, -594]
labels <- train_samples[, 594]

new_preds <- train_samples[-c(71, 125, 132, 87), -594]
new_labels <- train_samples[-c(71, 125, 132, 87), 594]


# Including extreme values

# Training control
trControl <- trainControl(savePredictions = TRUE, 
                          preProcOptions  = list(thresh = 0.85),
                          classProbs = TRUE, 
                          verboseIter = FALSE,
                          summaryFunction = newSummary)

set.seed(777)

# Train Logistic Regression Model
model_outs <- train(x= preds , y= labels, 
                    method = 'glm',
                    trControl = trControl,
                    metric = 'ROC',
                    preProcess=c("center", "scale", "pca"))

val_pred_out <- predict(model_outs, valid_samples[, -594])
acc1 <- sum(val_pred_out == valid_samples[, 594])/length(val_pred_out)
acc1

# Excluding extreme values

# Training control
trControl <- trainControl(savePredictions = TRUE, 
                          preProcOptions  = list(thresh = 0.85),
                          classProbs = TRUE, 
                          verboseIter = FALSE,
                          summaryFunction = newSummary)
set.seed(777)

# Train Logistic Regression Model
model_no_outs <- train(x= new_preds , y= new_labels, 
                       method = 'glm',
                       trControl = trControl,
                       metric = 'ROC',
                       preProcess=c("center", "scale", "pca"))

val_pred_no_out <- predict(model_no_outs, valid_samples[, -594])
acc2 <- sum(val_pred_no_out == valid_samples[, 594])/length(val_pred_no_out)
acc2

# Final Model Predictions

# Training control
trControl <- trainControl(savePredictions = TRUE, 
                          preProcOptions  = list(thresh = 0.85),
                          classProbs = TRUE, 
                          verboseIter = FALSE,
                          summaryFunction = newSummary)

# Train Logistic Regression Model
final_model <- train(x= train_predictors , y= train_labels, 
                     method = 'glm',
                     trControl = trControl,
                     metric = 'ROC',
                     preProcess=c("center", "scale", "pca"))

# Test set predictions
test_preds <- predict(final_model, test_data)
test_probs <- predict(final_model, test_data, type = "prob")
test_preds <- data.frame(test_ids, test_probs, test_preds)

# Features
features = 2:593

# Display the test set predictions
test_preds
features

# Save the predictions and the feature indices in the csv file
write.csv(test_preds, file = "0075715_Amin_MCICTLres.csv", row.names = FALSE)
write.csv(features, file = "0075715_Amin_MCICTLfeat.csv", row.names = FALSE)

# Save predictions 
save(test_preds,  file = "0075715_Amin_MCICTLres.RData")
# Save feature indices 
save(features,  file = "0075715_Amin_MCICTLfeat.RData")
