# spamclassifier
Spam and ham detetor
Modules used.
1. preprocessing : for preprocessing the text and converting them to tokens
2. classification : for training and testing the classifiers using different classifiers
              like logistic regression,decision tree, naive bayes and support vector machine
3.main : for starting the training and testing

APPROACH USED:
Here we analyze the text using Tfidf weight for each word for each document(message) and we pass those weight as a feature to
all the classifiers and on the basis of these features the classifiers get trained and 
tries to predict the appropriate output

For converting the words to tfidf score .
we use countvectorizer from sklearn to construct a bag of words for the different document
and them we transform them with the help of tfidf vectorizer into their corresponding
tfidf weight .
This process is common among all the classifiers so for the sake of simplicity we use a pipeline
feature to construct a stepwise procedure for doing the above mentioned task.

About how do i use it?
As i have already cross-validate the data into training and testing set (via jupyter)
You just have to run (main.py module) and things will get started.
The model accuracy with confusion matrix will be printed for everyclassifiers 


