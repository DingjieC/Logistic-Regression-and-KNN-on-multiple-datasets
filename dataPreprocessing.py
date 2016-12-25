import pandas as pd
from collections import Counter
filepath = "D:\CD\FA16\AML\datasets\wine quality\winequality-red.csv"
redwine = pd.DataFrame(pd.read_csv(filepath, sep=";"))
redwine['label'] = [0]*len(redwine)
for i,r in redwine.iterrows():
	if r.quality >= 6:
		redwine.set_value(i,'label',1)
redwine.drop('quality', axis=1, inplace=True)
redwine.to_csv("D:\CD\FA16\AML\datasets\wine quality\winequality-red-2.csv",sep=',',index=False)

#print(redwine.head())
#print(sum(redwine.label))
redwine_2 = pd.DataFrame(pd.read_csv("D:\CD\FA16\AML\datasets\wine quality\winequality-red-2.csv",sep=','))
redwine_2_class1 = redwine_2[redwine_2.label==1]
redwine_2_class0 = redwine_2[redwine_2.label==0]
print(len(redwine_2_class1))
print(len(redwine_2_class0))
print("Training data set size for class 1: ",int(len(redwine_2_class1)*0.9))
print("Training data set size for class 0: ",int(len(redwine_2_class0)*0.9))

# sampling the data
#http://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe
redwine_train1 = redwine_2_class1.sample(frac=0.9)
redwine_train0 = redwine_2_class0.sample(frac=0.9)
redwine_test1 = redwine_2_class1.loc[~redwine_2_class1.index.isin(redwine_train1.index)]
print("Length of test: (1)-",len(redwine_test1))
print("Length of train: (1)-",len(redwine_train1))
redwine_test0 = redwine_2_class0.loc[~redwine_2_class0.index.isin(redwine_train0.index)]
print("Length of test: (1)-",len(redwine_test0))
print("Length of train: (1)-",len(redwine_train0))
trainingRedWine = redwine_train0.append(redwine_train1,ignore_index=True)
testingRedWine = redwine_test0.append(redwine_test1,ignore_index = True)

# for logistic regression
trainingRedWine.to_csv("D:/CD/FA16/AML/datasets/wine quality/trainingRedWine.csv",sep=',',index=False)
testingRedWine.to_csv("D:/CD/FA16/AML/datasets/wine quality/testingRedWine.csv",sep=',',index=False)

# for kNN, normalize the features

#normalize the training dataset
trainingRedWineNorm = (trainingRedWine-trainingRedWine.mean())/(trainingRedWine.max()-trainingRedWine.min())
# this is for the class label
trainingRedWineNorm.label.unique()
#resetting the class label
for i,r in trainingRedWineNorm.iterrows():
	if r.label > 0:
		trainingRedWineNorm.set_value(i,'label',int(1))
	else:
		trainingRedWineNorm.set_value(i,'label',int(0))

#normalized the testing dataset
testingRedWineNorm = (testingRedWine-testingRedWine.mean())/(testingRedWine.max() - testingRedWine.min())
# this is for the class label
testingRedWineNorm.label.unique()
#resetting the class label
for i,r in testingRedWineNorm.iterrows():
	if r.label > 0:
		testingRedWineNorm.set_value(i,'label',int(1))
	else:
		testingRedWineNorm.set_value(i,'label',int(0))

# for logistic regression
trainingRedWineNorm.to_csv("D:/CD/FA16/AML/datasets/wine quality/trainingRedWineNorm.csv",sep=',',index=False)
testingRedWineNorm.to_csv("D:/CD/FA16/AML/datasets/wine quality/testingRedWineNorm.csv",sep=',',index=False)
