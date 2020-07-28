from tqdm import tqdm
from dataset import QuoraDataset
from torch.utils.data import DataLoader
import config
import torch
import numpy as np
from sklearn import svm

def get_features(model,train=True):
    if train==True:
        dataset = QuoraDataset()
        n_samples = len(dataset)
        train_dataloader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=False)
        labels = []
        features = []
        with torch.no_grad():
            for ind,d in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                input_ids = d['input_ids']
                attention_mask = d['attention_mask']
                label = np.array(d['label'])
                labels.extend(label)
                feature = model(input_ids,attention_mask)
                features.extend(feature)
                del ind
                del d
        return np.array(features), np.array(labels)
    
    else:
        dataset = QuoraDataset(train=False)
        n_samples = len(dataset)
        test_dataloader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=False)
        features = []
        with torch.no_grad():
            for ind,d in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                input_ids = d['input_ids']
                attention_mask = d['attention_mask']

                feature = model(input_ids,attention_mask)
                features.extend(feature)
        return np.array(features)

def train_fn(X_train,y_train):
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    return clf

def eval_fn(clf,X_test):
    return clf.predict(X_test)

def test_results(clf,test_data):
    return clf.predict(test_data)