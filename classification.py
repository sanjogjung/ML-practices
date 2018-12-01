from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import preprocessing as prp
def train_classifiers(message_train,label_train,message_test,label_test):
    pipeline1 = Pipeline([
    ('bow', CountVectorizer(analyzer=prp.message_process)),  # tokenizing strings and taking count
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier',LogisticRegression()),  # train on TF-IDF vectors with Logistic Regression classifier
    ])
    pipeline1.fit(message_train,label_train)
    pred1=pipeline1.predict(message_test)
    print('Using Logistic regression classifier we got the following results')
    print('Accuracy score in percentage')
    print(accuracy_score(label_test, pred1)*100)
    print('Confusion matrix :')
    print(confusion_matrix(label_test,pred1))
    print('\n')
    
    pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=prp.message_process)),  # tokenizing strings and taking count
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier',DecisionTreeClassifier()),  # train on TF-IDF vectors on decision tree classifier
    ])
    pipeline2.fit(message_train,label_train)
    pred2=pipeline2.predict(message_test)
    print('Using Decision Tree classifier we got the following results')
    print('Accuracy score in percentage')
    print(accuracy_score(label_test, pred2)*100)
    print('Confusion matrix :')
    print(confusion_matrix(label_test,pred2))
    print('\n')
    pipeline3 = Pipeline([
    ('bow', CountVectorizer(analyzer=prp.message_process)),  # tokenizing strings and taking count
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB()),  # train on TF-IDF vectors wuth Baiyes classifier
    ])
    pipeline3.fit(message_train,label_train)
    pred3=pipeline3.predict(message_test)
    print('Using Naive Bayes classifier we got the following results')
    print('Accuracy score in percentage')
    print(accuracy_score(label_test, pred3)*100)
    print('Confusion matrix :')
    print(confusion_matrix(label_test,pred3))
    print('\n')

    pipeline4 = Pipeline([
    ('bow', CountVectorizer(analyzer=prp.message_process)),  # tokenizing strings and taking count
    ('tfidf', TfidfTransformer()),  # calculating  TF-IDF scores for everystrings
    ('classifier',SVC(kernel='linear')),  # train on TF-IDF vectors with Support vector machine classifier
    ])
    pipeline4.fit(message_train,label_train)
    pred4=pipeline4.predict(message_test)
    print('Using Support Vector machine classifier we got the following results')
    print('Accuracy score in percentage')
    print(accuracy_score(label_test, pred4)*100)
    print('Confusion matrix :')
    print(confusion_matrix(label_test,pred4))
    print('For the given test data, following are the corresponding classifications :\n')
    for label in pred4:
        print(label)


    