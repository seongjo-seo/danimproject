import surprise
# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import pickle


raw=pd.read_csv('./travel_records.csv')
# raw.drop_duplicates(inplace=True)

#swapping columns
raw=raw[['userid','place_id','point']] 
raw.columns = ['n_users','n_items','rating']

rawTrain,rawholdout = train_test_split(raw, test_size=0.15)
# when importing from a DF, you only need to specify the scale of the ratings.
reader = surprise.Reader(rating_scale=(1,5)) 
#into surprise:
data = surprise.Dataset.load_from_df(rawTrain,reader)
holdout = surprise.Dataset.load_from_df(rawholdout,reader)


### CF

kSplit = surprise.model_selection.split.KFold(n_splits=10, shuffle=True) # split data into folds. 


rmseKNN = []

sim_options = {'name': 'pearson',
               'user_based': False  # compute  similarities between items
               }
collabKNN = surprise.KNNBasic(k=20, sim_options=sim_options) #try removing sim_options. You'll find memory errors.

for trainset, testset in kSplit.split(data): #iterate through the folds.
    collabKNN.fit(trainset)
    predictionsKNN = collabKNN.test(testset)
    rmseKNN.append(surprise.accuracy.rmse(predictionsKNN,verbose=True))#get root means squared error
print(sum(rmseKNN)/len(rmseKNN))

with open('models/collabKNN.pkl', 'wb') as f:
    pickle.dump(collabKNN, f)

### MF

funkSVD = surprise.prediction_algorithms.matrix_factorization.SVD(n_factors=40, n_epochs=200, biased=True)


rmseSVD = []

min_error = 1
for trainset, testset in kSplit.split(data): #iterate through the folds.
    funkSVD.fit(trainset)
    predictionsSVD = funkSVD.test(testset)
    rmseSVD.append(surprise.accuracy.rmse(predictionsSVD,verbose=True))#get root means squared error

print(sum(rmseSVD)/len(rmseSVD))

with open('models/funkSVD.pkl', 'wb') as f:
    pickle.dump(funkSVD, f)

## co-clustering

rmseCo = []

coClus = surprise.prediction_algorithms.co_clustering.CoClustering(n_cltr_u=4,n_cltr_i=8,n_epochs=25)
for trainset, testset in kSplit.split(data): #iterate through the folds.
    coClus.fit(trainset)
    predictionsCoClus = coClus.test(testset)
    rmseCo.append(surprise.accuracy.rmse(predictionsCoClus,verbose=True))#get root means squared error
print(sum(rmseCo)/len(rmseCo))

with open('models/coClus.pkl', 'wb') as f:
    pickle.dump(coClus, f)

### Slope One Collaborative Filtering Algorithm

slopeOne = surprise.prediction_algorithms.slope_one.SlopeOne()


rmseSlope = []
for trainset, testset in kSplit.split(data): #iterate through the folds.
    slopeOne.fit(trainset)
    predictionsSlope = slopeOne.test(testset)
    rmseSlope.append(surprise.accuracy.rmse(predictionsSlope,verbose=True))#get root means squared error
print(sum(rmseSlope)/len(rmseSlope))

with open('models/slopeOne.pkl', 'wb') as f:
    pickle.dump(slopeOne, f)