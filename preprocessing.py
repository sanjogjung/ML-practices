#importing dependencies
import nltk                  
import pandas as pd 
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def message_process(message):
    nopunc=[char.lower() for char in message if char not in string.punctuation] #removing punctuation
    nopunc=''.join(nopunc) #joining characters to a form
    words=nopunc.split() #splitting string to form a token
    ps=nltk.PorterStemmer() 
    stem_words=[ps.stem(word) for word in words] #stemming tokens
    return [word for word in stem_words if word not in stopwords.words('english')]#removing stop words
 