
# coding: utf-8

# In[5]:

import pandas as pd
from collections import Counter
filepath = "C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Wine Corpus/winequality-white.csv"
whitewine = pd.DataFrame(pd.read_csv(filepath, sep=";"))
whitewine['label'] = [0]*len(whitewine)
for i,r in whitewine.iterrows():
    if r.quality >= 6:
        whitewine.set_value(i,'label',1)
whitewine.drop('quality', axis=1, inplace=True)
whitewine.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Wine Corpus/winequality-white-2.csv",sep=',',index=False)
#print(redwine.head())
#print(sum(redwine.label))


# In[6]:

whitewine_2 = pd.DataFrame(pd.read_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Wine Corpus/winequality-white-2.csv",sep=','))
whitewine_2_class1 = whitewine_2[whitewine_2.label==1]
whitewine_2_class0 = whitewine_2[whitewine_2.label==0]
print(len(whitewine_2_class1))
print(len(whitewine_2_class0))
print("Training data set size for class 1: ",int(len(whitewine_2_class1)*0.9))
print("Training data set size for class 0: ",int(len(whitewine_2_class0)*0.9))


# In[9]:

# sampling the data
#http://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe
whitewine_train1 = whitewine_2_class1.sample(frac=0.9)
whitewine_train0 = whitewine_2_class0.sample(frac=0.9)
whitewine_test1 = whitewine_2_class1.loc[~whitewine_2_class1.index.isin(whitewine_train1.index)]
print("Length of test: (1)-",len(whitewine_test1))
print("Length of train: (1)-",len(whitewine_train1))
whitewine_test0 = whitewine_2_class0.loc[~whitewine_2_class0.index.isin(whitewine_train0.index)]
print("Length of test: (0)-",len(whitewine_test0))
print("Length of train: (0)-",len(whitewine_train0))
trainingWhiteWine = whitewine_train0.append(whitewine_train1,ignore_index=True)
testingWhiteWine = whitewine_test0.append(whitewine_test1,ignore_index = True)


# In[11]:

# for logistic regression
trainingWhiteWine.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Wine Corpus/trainingWhiteWine.csv",sep=',',index=False)
testingWhiteWine.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Wine Corpus/testingWhiteWine.csv",sep=',',index=False)


# In[12]:

# for kNN, normalize the features

#normalize the training dataset
trainingWhiteWineNorm = (trainingWhiteWine-trainingWhiteWine.mean())/(trainingWhiteWine.max()-trainingWhiteWine.min())
# this is for the class label
trainingWhiteWineNorm.label.unique()
#resetting the class label
for i,r in trainingWhiteWineNorm.iterrows():
    if r.label > 0:
        trainingWhiteWineNorm.set_value(i,'label',int(1))
    else:
        trainingWhiteWineNorm.set_value(i,'label',int(0))


# In[15]:

#normalized the testing dataset
testingWhiteWineNorm = (testingWhiteWine-testingWhiteWine.mean())/(testingWhiteWine.max() - testingWhiteWine.min())
# this is for the class label
testingWhiteWineNorm.label.unique()
#resetting the class label
for i,r in testingWhiteWineNorm.iterrows():
    if r.label > 0:
        testingWhiteWineNorm.set_value(i,'label',int(1))
    else:
        testingWhiteWineNorm.set_value(i,'label',int(0))


# In[16]:

# for logistic regression
trainingWhiteWineNorm.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Wine Corpus/trainingWhiteWineNorm.csv",sep=',',index=False)
testingWhiteWineNorm.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Wine Corpus/testingWhiteWineNorm.csv",sep=',',index=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



