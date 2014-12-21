---
title: "HAR ML Modeling"
author: "Shabbir Suterwala"
date: "Sunday, December 21, 2014"
output: html_document
---

# Predictive Modeling for Human Activity Recognition (HAR)
**Shabbir Suterwala**

## Executive Summary
The purpose of this exercise it to predict whether if people are performing Biceps Curl exercises properly. Using the Weight Lifting Exercises Dataset available at http://groupware.les.inf.puc-rio.br/har, a prediction model based on Random Forest is trained and tested with 98% and 100% accuracies respectively. The dataset is captured using various devices such as Jawbone Up, Nike FuelBand, and Fitbit and are worn by the subjects when performing the exercise.

## Data Loading and Cleaning
The training and testing data was downloaded from https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv and https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv respectively. Training data was analyzed and cleaned as follows:

   1. All columns that had  > 95% NA values were discarded. There were 100 such columns.
   2. The following columns were removed:
      + X: This contained the row number and is sot useful for prediction
      + user_name: This contained the subject names and is not useful for prediction
      + raw_timestamp_part_1/raw_timestamp_part_2: These were redundant due to cvtd_timestamp
      + new_window: It had near zero variation

After cleaning as above, there were no columns that had any NA.


```r
library(caret)
load <- function() {
    pml <- read.csv("pml-training.csv", na.strings=c("#DIV/0!", "NA", ""))
    pml$cvtd_timestamp <- as.numeric(strptime(pml$cvtd_timestamp, format="%d/%m/%Y %H:%M"))
    
    #### Convert no(=1) / yes(=2) to 0/1
    pml$new_window <- as.integer(pml$new_window) - 1
    
    ##### Remove unwanted columns and those with at least 95% NA values ############    
    pml.unwanted <- names(pml)[sapply(pml, function(x) sum(is.na(x))/length(x)) > 0.95]
    
    #The following variables are also removed:
    #X: Just a row number
    #user_name: Is the name of the subject. This info is not important and will introduce a bias if used.
    #raw_timestamp_part_1/raw_timestamp_part_2: Contained in cvtd_timestamp
    #new_window: it was nearzero
    pml.unwanted <- c(pml.unwanted, c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2"))
    pml <- pml[, setdiff(names(pml), pml.unwanted)]

    #remove near zero values
    nzv <- nearZeroVar(pml, saveMetrics=F)
    pml <- pml[, -nzv]
    
    ###### Build Train and Test Sets ##################
    set.seed(3333)
    inTrain <- createDataPartition(pml$classe, p=.75, list=FALSE)
    train.data <- pml[inTrain, ]
    test.data <- pml[-inTrain, ]
    
    ######### Checking if any factor variables are left ##########
    #train.types <- sapply(train.data, class)
    #train.types.factor <- train.types[train.types == "factor"]
    #print(train.types.factor)
    
    ######## return the list #################
    list(pml=pml, train.data=train.data, test.data=test.data)
} 

pml.data <- load()
```

There are 54 variables in the cleaned dataset

## Data Modeling

Three predictive models were built (details below). Each model was cross validated with testing data that was extracted and set aside from the training data. The actual testing was used only when the final model was ready. 

### Linear Discriminant Analysis Model


```r
doLDA <- function (pml.data, method="lda") {
    set.seed(4545)

    ppo <- trainControl()$preProcOptions
    ppo$thresh = 0.975
    trc <- trainControl(preProcOptions=ppo)
    
    model <- train(classe ~ ., data=pml.data$train.data, method=method, preProcess = "pca", trControl = trc)
    pred <-  predict(model, pml.data$test.data)
    conf.mx <- confusionMatrix(pred, pml.data$test.data[, ncol(pml.data$test.data)])
    modelFit <- list(model = model, pred = pred, conf.mx=conf.mx)        
}

start.tm <- proc.time()
lda.modelFit <- doLDA(pml.data, "lda")
cfx <- lda.modelFit$conf.mx
cfx
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 942 176 153  46  54
##          B  92 437  82  58 177
##          C 142 196 523 150 120
##          D 208  79  82 478 101
##          E  11  61  15  72 449
## 
## Overall Statistics
##                                         
##                Accuracy : 0.577         
##                  95% CI : (0.563, 0.591)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.466         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.675   0.4605    0.612   0.5945   0.4983
## Specificity             0.878   0.8966    0.850   0.8854   0.9603
## Pos Pred Value          0.687   0.5165    0.462   0.5042   0.7385
## Neg Pred Value          0.872   0.8738    0.912   0.9176   0.8948
## Prevalence              0.284   0.1935    0.174   0.1639   0.1837
## Detection Rate          0.192   0.0891    0.107   0.0975   0.0916
## Detection Prevalence    0.280   0.1725    0.231   0.1933   0.1240
## Balanced Accuracy       0.777   0.6785    0.731   0.7399   0.7293
```

```r
proc.time() - start.tm
```

```
##    user  system elapsed 
##   30.28    0.09   30.42
```

LDA's out of sample error rate (1 - Accuracy) is 42.31%

### Random Forest with bootstrap sampling

```r
doRF <- function(pml.data, method="boot") {
    set.seed(4444)
    
    #preproc <- preProcess(train.data[, -ncol(train.data)], method="pca", thresh = 0.975)
    ppo <- trainControl()$preProcOptions
    ppo$thresh = 0.975
    trc <- trainControl(method=method, number=5, preProcOptions=ppo)
    
    rf.model <- train(classe ~ ., data=pml.data$train.data, method="rf", preProcess = "pca", trControl = trc)
    rf.pred <- predict(rf.model, pml.data$test.data)
    conf.mx <- confusionMatrix(rf.pred, pml.data$test.data[, ncol(pml.data$test.data)])
    rf.modelFit <- list(model = rf.model, pred = rf.pred, conf.mx=conf.mx)
    
}

if (!exists("rf.modelFit.boot")) {
    start.tm <- proc.time()
    rf.modelFit.boot <- doRF(pml.data, "boot")
    proc.time() - start.tm
}
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```
##    user  system elapsed 
##   749.8     5.0   757.5
```

```r
cfx <- rf.modelFit.boot$conf.mx
cfx
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1383    6    0    2    1
##          B    7  933   15    0    3
##          C    2   10  828   28    5
##          D    3    0   11  774    4
##          E    0    0    1    0  888
## 
## Overall Statistics
##                                         
##                Accuracy : 0.98          
##                  95% CI : (0.976, 0.984)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.975         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.991    0.983    0.968    0.963    0.986
## Specificity             0.997    0.994    0.989    0.996    1.000
## Pos Pred Value          0.994    0.974    0.948    0.977    0.999
## Neg Pred Value          0.997    0.996    0.993    0.993    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.190    0.169    0.158    0.181
## Detection Prevalence    0.284    0.195    0.178    0.162    0.181
## Balanced Accuracy       0.994    0.988    0.979    0.979    0.993
```
Random Forest with bootstraping's out of sample error rate (1 - Accuracy) is 2%

### Random Forest with cross validation

```r
#### Using RF with method CV #################
if (!exists("rf.modelFit.cv")) {
    start.tm <- proc.time()
    rf.modelFit.cv <- doRF(pml.data, "cv")
    proc.time() - start.tm
}
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```
##    user  system elapsed 
##  553.99    2.17  556.77
```

```r
cfx <- rf.modelFit.cv$conf.mx
cfx
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1383    6    0    2    1
##          B    7  930   14    0    3
##          C    2   13  833   27    6
##          D    3    0    7  775    4
##          E    0    0    1    0  887
## 
## Overall Statistics
##                                         
##                Accuracy : 0.98          
##                  95% CI : (0.976, 0.984)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.975         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.991    0.980    0.974    0.964    0.984
## Specificity             0.997    0.994    0.988    0.997    1.000
## Pos Pred Value          0.994    0.975    0.946    0.982    0.999
## Neg Pred Value          0.997    0.995    0.995    0.993    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.190    0.170    0.158    0.181
## Detection Prevalence    0.284    0.195    0.180    0.161    0.181
## Balanced Accuracy       0.994    0.987    0.981    0.980    0.992
```
Random Forest with cross validation's out of sample error rate (1 - Accuracy) is 1.96%

## Final Model
From the above confusion matrix outputs, we can see that Random Forest based models are far superior; not only do they give better accuracy, the kappa is high too. The Random Forest with cross validation has a slighly lower out of sample error, so we will select this model.

## Test Execution
Execution of selected model against the test data yeilded 100% accurate results. The results of this execution is not provided in this report since doing that may voilate Coursera.org's honor code.

# Conclusion
A prediction model based on Random Forest alogrithm was built that indicates with 98% accuracy whether someone is performing the Biceps Curl exercise properly or not. The alogrithm was tested on test data and provided 100% accurate results.


# References
   1. http://groupware.les.inf.puc-rio.br/har
   2. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
 
