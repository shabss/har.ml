
library(caret)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

test.model <- function (model) {
    pml <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!", "NA", ""))
    pml$cvtd_timestamp <- as.numeric(strptime(pml$cvtd_timestamp, format="%d/%m/%Y %H:%M"))
    pred.test <- predict(model, newdata=pml)
}


load <- function() {
    pml <- read.csv("pml-training.csv", na.strings=c("#DIV/0!", "NA", ""))
    pml$cvtd_timestamp <- as.numeric(strptime(pml$cvtd_timestamp, format="%d/%m/%Y %H:%M"))
    
    #### Convert no(=1) / yes(=2) to 0/1
    pml$new_window <- as.integer(pml$new_window) - 1
    
    ###### Factor Values ############
    #user_name
    #new_window -- yes/no
    #classe
    
    ##### Remove unwanted columns and those with at least 95% NA values ############
    pml.unwanted <- names(pml)[sapply(pml, function(x) sum(is.na(x))/length(x)) > 0.95]
    
    #The following variables are also removed:
    #X: Just a row number
    #user_name: Is the name of the subject. This info is not important and will introduce a bias if used.
    #raw_timestamp_part_1/raw_timestamp_part_2: Contained in cvtd_timestamp
    #new_window: it was nearzero
    pml.unwanted <- c(pml.unwanted, c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2"))
    print(pml.unwanted)
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
    train.types <- sapply(train.data, class)
    train.types.factor <- train.types[train.types == "factor"]
    print(train.types.factor)
    
    ######## return the list #################
    list(pml=pml, train.data=train.data, test.data=test.data)
} 


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

doGLM <- function(pml.data) {
    set.seed(1234)
    
    ppo <- trainControl()$preProcOptions
    ppo$thresh = 0.975
    trc <- trainControl(preProcOptions=ppo)
    
    model <- train(classe ~ ., data=pml.data$train.data, method="glm", preProcess = "pca", trControl = trc)
    pred <-  predict(model, pml.data$test.data)
    conf.mx <- confusionMatrix(pred, pml.data$test.data[, ncol(pml.data$test.data)])
    modelFit <- list(model = model, pred = pred, conf.mx=conf.mx)    
}

doNB <- function (pml.data, method="lda") {
    set.seed(4545)

    ppo <- trainControl()$preProcOptions
    ppo$thresh = 0.975
    trc <- trainControl(preProcOptions=ppo)
    
    model <- train(classe ~ ., data=pml.data$train.data, method=method, preProcess = "pca", trControl = trc)
    pred <-  predict(model, pml.data$test.data)
    conf.mx <- confusionMatrix(pred, pml.data$test.data[, ncol(pml.data$test.data)])
    modelFit <- list(model = model, pred = pred, conf.mx=conf.mx)        
}


###### Start ###############
if (!exists("pml.data")) {
    pml.data <- load()
}

#### Using RF with method boot #################
start.tm <- proc.time()
if (!exists("rf.modelFit.method_boot")) {
    rf.modelFit.method_boot <- doRF(pml.data)
}
print(rf.modelFit.method_boot$conf.mx)
print(proc.time() - start.tm)

#### Using RF with method CV #################
start.tm <- proc.time()
if (!exists("rf.modelFit.method_cv")) {
    rf.modelFit.method_cv <- doRF(pml.data, "cv")
}
print(rf.modelFit.method_cv$conf.mx)
print(proc.time() - start.tm)

#### Using lda  #################
start.tm <- proc.time()
if (!exists("lda.modelFit")) {
    lda.modelFit <- doNB(pml.data, "lda")
}
print(lda.modelFit$conf.mx)
print(proc.time() - start.tm)

outcome <- test.model(rf.modelFit.method_cv$model)
print(outcome)
