---
title: "Practical Machine-Learning"
author: "Bellemin Ronan"
date: "2/25/2022"
output: 
  html_document:
    keep_md: true
editor_options: 
  chunk_output_type: inline
---

# Assignement project: Week 4

## Setup

```r
library(readr)
library(dplyr)
library(caret)
library(corrplot)

training_set <- read_csv("pml-training.csv")
test_set <- read_csv("pml-testing.csv")

# removing useless columns for our ML models
training_set = training_set %>% select(-c(1:7))
test_set = test_set %>% select(-c(1:7))
```


## Data Pre-Processing
We check our outcome variable

```r
table(training_set$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

We convert 'classe' as factor

```r
# converting 'classe' as factor
training_set$classe <- as.factor(training_set$classe)
```

Dealing with NAs by removing columns that have more than 90% of NAs among the observations

```r
# training set
na_threshold <- nrow(training_set)/100*90
handle_na <- colSums(is.na(training_set)) < na_threshold
clean_training_set <- training_set[,handle_na]
rm(training_set)

# test set
na_threshold <- nrow(test_set)/100*90
handle_na <- colSums(is.na(test_set)) < na_threshold
clean_test_set <- test_set[,handle_na]
rm(test_set)
```


## Exploratory Analysis
We create a correlation plot to see correlations between our variables.

```r
#palette
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corr_matrix = cor(select(clean_training_set, c(1:52)), use = "complete.obs")
corrplot(corr_matrix, method = 'color', 
                  tl.col = "black", 
                  tl.cex = 0.7, 
                  col=col(10), type="upper")    
```

![](projectML_files/figure-html/unnamed-chunk-3-1.png)<!-- -->


## Splitting Data (Cross-Validation)
We split our training data into a test set (40%) and a training set (60%).

```r
set.seed(158)

inTrainIndex <- createDataPartition(clean_training_set$classe, p=0.6, list = F)

training_training_set <- clean_training_set[inTrainIndex,]

training_test_set <- clean_training_set[-inTrainIndex,]
```


## Regression/Decision Trees Model
We create a decision trees model and check the accuracy of this model.

```r
rpart_model_training <- train(classe ~., method='rpart', data=training_training_set)

rpart_model_predict <- predict(rpart_model_training, training_test_set)

confusionMatrix(training_test_set$classe, rpart_model_predict)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2041   29  157    0    5
##          B  632  510  376    0    0
##          C  659   38  671    0    0
##          D  582  224  480    0    0
##          E  207  178  389    0  668
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4958          
##                  95% CI : (0.4847, 0.5069)
##     No Information Rate : 0.5252          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3406          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4953   0.5209  0.32369       NA  0.99257
## Specificity            0.9487   0.8532  0.87927   0.8361  0.89210
## Pos Pred Value         0.9144   0.3360  0.49050       NA  0.46325
## Neg Pred Value         0.6295   0.9259  0.78358       NA  0.99922
## Prevalence             0.5252   0.1248  0.26421   0.0000  0.08578
## Detection Rate         0.2601   0.0650  0.08552   0.0000  0.08514
## Detection Prevalence   0.2845   0.1935  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7220   0.6871  0.60148       NA  0.94233
```


## Random Forest Model
We create a random forest model and check the accuracy of this model.

```r
rf_model_training <- train(classe ~., method='ranger', data=training_training_set)

rf_model_predict <- predict(rf_model_training, training_test_set)
confusionMatrix(training_test_set$classe, rf_model_predict)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    4 1514    0    0    0
##          C    0    8 1360    0    0
##          D    0    0   12 1271    3
##          E    0    0    2    5 1435
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9957         
##                  95% CI : (0.9939, 0.997)
##     No Information Rate : 0.285          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9945         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9947   0.9898   0.9961   0.9979
## Specificity            1.0000   0.9994   0.9988   0.9977   0.9989
## Pos Pred Value         1.0000   0.9974   0.9942   0.9883   0.9951
## Neg Pred Value         0.9993   0.9987   0.9978   0.9992   0.9995
## Prevalence             0.2850   0.1940   0.1751   0.1626   0.1833
## Detection Rate         0.2845   0.1930   0.1733   0.1620   0.1829
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9991   0.9971   0.9943   0.9969   0.9984
```


## Conclusion
We performed two models above. A Decision Trees Model with an accuracy of 49.58% and a Random Forest Model with an accuracy of 99.57%. The Random Forest Model is way better than the Decision Trees Model. Therefore, we chose the RF model to predict the manner in which the 20 different test cases did the exercise.


## Prediction in the Test Set

```r
rf_model_final_prediction <- predict(rf_model_training, clean_test_set)
rf_model_final_prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
