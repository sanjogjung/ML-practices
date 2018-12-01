import pandas as pd 
msg_train=pd.read_csv('train.csv') #for training
msg_train.head()
msg_train['message']
msg_train['label'] 
msg_test=pd.read_csv('test.csv') # for testing
msg_test.head() 
msg_test['message']
msg_test['label']
import classification as clf
clf.train_classifiers(msg_train['message'],msg_train['label'],msg_test['message'],msg_test['label'])
