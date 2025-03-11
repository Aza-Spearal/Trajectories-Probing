import pandas as pd
import numpy as np
import torch
from lazypredict.Supervised import LazyClassifier
import os
from sklearn.model_selection import train_test_split

def lazy(layer, x_train, y_train, x_test, y_test, title):

    title = 'CM_'+title+'_'+str(layer)
    xtrain, xtest, ytrain, ytest = np.array(x_train[layer].cpu()), np.array(x_test[layer].cpu()), np.array(y_train.cpu()), np.array(y_test.cpu())
    clf = LazyClassifier(verbose=0, ignore_warnings=True, title=title)
    models,predictions = clf.fit(xtrain, xtest, ytrain, ytest)
    print(models)
    return models


def fill_df(layer, df, dazy, name):
    name = name+'.csv'
    if not os.path.exists(name):
        df[str(layer)] = dazy['F1 Score']
    else:
        df[str(layer)] = dazy.loc[df.index, 'F1 Score']
    
    df.to_csv(name)
    return df


name = 'human_activation'

dt1 = name+'_model'
dt2 = name+'_rand'

[xtrain, xtest, ytrain, ytest] = torch.load(dt1+'.pt') #[samples, layers, dmodel]
[xtrain_rand, xtest_rand, ytrain_rand, ytest_rand] = torch.load(dt2+'.pt')

#print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
#print(xtrain_rand.shape, xtest_rand.shape, ytrain_rand.shape, ytest_rand.shape)

'''
size = 3
xtrain = torch.chunk(xtrain, size, dim=0)[0]
xtest = torch.chunk(xtest, size, dim=0)[0]
ytrain = torch.chunk(ytrain, size, dim=0)[0]
ytest = torch.chunk(ytest, size, dim=0)[0]
xtrain_rand = torch.chunk(xtrain_rand, size, dim=0)[0]
xtest_rand = torch.chunk(xtest_rand, size, dim=0)[0]
ytrain_rand = torch.chunk(ytrain_rand, size, dim=0)[0]
ytest_rand = torch.chunk(ytest_rand, size, dim=0)[0]
'''

xtrain = xtrain.permute(1, 0, 2)
xtest = xtest.permute(1, 0, 2)
xtrain_rand = xtrain_rand.permute(1, 0, 2)
xtest_rand = xtest_rand.permute(1, 0, 2)

n_layers = len(xtrain)

print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
print(xtrain_rand.shape, xtest_rand.shape, ytrain_rand.shape, ytest_rand.shape)


mazy_df = pd.DataFrame()
razy_df = pd.DataFrame()
mazy_razy_df = pd.DataFrame()

for layer in range(n_layers):
    
    mazy = lazy(layer, xtrain, ytrain, xtest, ytest, name)
    mazy_df = fill_df(layer, mazy_df, mazy, 'mazy_'+name)

    razy = lazy(layer, xtrain_rand, ytrain_rand, xtest_rand, ytest_rand, name+'_random')
    razy_df = fill_df(layer, razy_df, razy, 'razy_'+name)
    
    mazy_razy = mazy.subtract(razy, fill_value=0).sort_values(['F1 Score'], ascending=[False])
    mazy_razy_df = fill_df(layer, mazy_razy_df, mazy_razy, 'mazy_razy_'+name)