# AMLProject
Algorithms for machine learning
1. K nearest neighbour
2. Logistic Regression

Datasets:
1. RedWine
2. WhiteWine
3. Bank

#Usage
python driver.py <dataset dir> <training file name> <testing file name> <algo> [options]
<dataset dir> - directory where the training and testing files reside
<training file name> - a CSV file used for training. Our algorithms assume that the class label is the last attribute of the file
<testing file name> - a CSV file used for testing.
<algo> - param takes values as knn or lr
        knn - for kNN algorithm
        lr - for Logistic Regression
[options] -
    if <algo> is knn -
        options take value as
            <option1> - k value (this value is mandatory of knn to run)
            [option2] ... [option n] - rest are the indices for the best predictors
     if <algo> is lr -
        options take value as
            [option1] ... [option n] - the indices for the best predictors
For running knn:
python driver.py data/ trainingRedWineNorm.csv testingRedWineNorm.csv knn 5

Here the value for k is set to 5, with no best predictors given.

For runnning lr:
python driver.py data/ trainingRedWineNorm.csv testingRedWineNorm.csv lr 2 4
Here the best predictor indices are given as 2 4 (algo follow 0th based indexing)
To consider all the features use,
python driver.py data/ trainingRedWineNorm.csv testingRedWineNorm.csv lr


#Note :
The data is preprocessed using the scripts provided in directory "preprocessing scripts". And this preprocessed data is used for the algorithm.
The preprocessed data for all three datasets is present in "data" directory