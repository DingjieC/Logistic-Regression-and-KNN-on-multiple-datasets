
# coding: utf-8

# In[17]:

import pandas as pd
from collections import Counter
filepath = "C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Bank marketing/bank.csv"
bank = pd.DataFrame(pd.read_csv(filepath, sep=";"))
bank['label'] = [0]*len(bank)
for i,r in bank.iterrows():
    if r.y == "yes":
        bank.set_value(i,'label',1)
bank.drop('y', axis=1, inplace=True)
bank.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Bank marketing/bank-2.csv",sep=',',index=False)
#print(redwine.head())
#print(sum(redwine.label))


# In[40]:

bank_2 = pd.DataFrame(pd.read_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Bank marketing/bank-2.csv",sep=','))
job = list(set(bank_2.job))
marital = list(set(bank_2.marital))
education = list(set(bank_2.education))
default = list(set(bank_2.default))
housing = list(set(bank_2.housing))
loan = list(set(bank_2.loan))
month = list(set(bank_2.month))
poutcome = list(set(bank_2.poutcome))
bank_2['job_id'] = [job.index(v) for v in bank_2['job']]
bank_2['marital_id'] = [marital.index(v) for v in bank_2['marital']]
bank_2['education_id'] = [education.index(v) for v in bank_2['education']]
bank_2['default_id'] = [default.index(v) for v in bank_2['default']]
bank_2['housing_id'] = [housing.index(v) for v in bank_2['housing']]
bank_2['loan_id'] = [loan.index(v) for v in bank_2['loan']]
bank_2['poutcome_id'] = [poutcome.index(v) for v in bank_2['poutcome']]
bank_2['month_id'] = [month.index(v) for v in bank_2['month']]
bank_2.drop('job', axis=1, inplace=True)
bank_2.drop('marital', axis=1, inplace=True)
bank_2.drop('education', axis=1, inplace=True)
bank_2.drop('default', axis=1, inplace=True)
bank_2.drop('housing', axis=1, inplace=True)
bank_2.drop('loan', axis=1, inplace=True)
bank_2.drop('poutcome', axis=1, inplace=True)
bank_2.drop('day', axis=1, inplace=True)
bank_2.drop('contact', axis=1, inplace=True)
bank_2.drop('month', axis=1, inplace=True)
cols = bank_2.columns.tolist()
cols = cols[0:6] + cols[7:] + [cols[6]]
bank_2 = bank_2[cols]
print(bank_2.columns)
bank_2_class1 = bank_2[bank_2.label==1]
bank_2_class0 = bank_2[bank_2.label==0]
print(len(bank_2_class1))
print(len(bank_2_class0))
print("Training data set size for class 1: ",int(len(bank_2_class1)*0.9))
print("Training data set size for class 0: ",int(len(bank_2_class0)*0.9))


# In[41]:

# sampling the data
#http://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe
bank_train1 = bank_2_class1.sample(frac=0.9)
bank_train0 = bank_2_class0.sample(frac=0.9)
bank_test1 = bank_2_class1.loc[~bank_2_class1.index.isin(bank_train1.index)]
print("Length of test: (1)-",len(bank_test1))
print("Length of train: (1)-",len(bank_train1))
bank_test0 = bank_2_class0.loc[~bank_2_class0.index.isin(bank_train0.index)]
print("Length of test: (0)-",len(bank_test0))
print("Length of train: (0)-",len(bank_train0))
trainingBank = bank_train0.append(bank_train1,ignore_index=True)
testingBank = bank_test0.append(bank_test1,ignore_index = True)
print(trainingBank)


# In[42]:

# for logistic regression
trainingBank.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Bank marketing/trainingBank.csv",sep=',',index=False)
testingBank.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Bank marketing/testingBank.csv",sep=',',index=False)


# In[43]:

# for kNN, normalize the features

#normalize the training dataset
trainingBankNorm = (trainingBank-trainingBank.mean())/(trainingBank.max()-trainingBank.min())
# this is for the class label
trainingBankNorm.label.unique()
#resetting the class label
for i,r in trainingBankNorm.iterrows():
    if r.label > 0:
        trainingBankNorm.set_value(i,'label',int(1))
    else:
        trainingBankNorm.set_value(i,'label',int(0))


# In[44]:

#normalized the testing dataset
testingBankNorm = (testingBank-testingBank.mean())/(testingBank.max() - testingBank.min())
# this is for the class label
testingBankNorm.label.unique()
#resetting the class label
for i,r in testingBankNorm.iterrows():
    if r.label > 0:
        testingBankNorm.set_value(i,'label',int(1))
    else:
        testingBankNorm.set_value(i,'label',int(0))


# In[45]:

# for logistic regression
trainingBankNorm.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Bank marketing/trainingBankNorm.csv",sep=',',index=False)
testingBankNorm.to_csv("C:/Users/ROHAN/Desktop/MS/MS Fall 2016/Applied Machine Learning/Project/Bank marketing/testingBankNorm.csv",sep=',',index=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



