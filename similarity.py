"""
Author: Ratnesh Chandak
versions:
    Python 3.7.4
    pandas==0.25.1
    scikit-learn==0.21.3
    numpy==1.17.2
"""
# get the stop words
from sklearn.feature_extraction import text 
stop_words = text.ENGLISH_STOP_WORDS

#give stop words to constructor so that it will ignore those while vectorization of sentences
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,1), stop_words=stop_words)

#read input file
import pandas as pd

#dataset contains Unique_Id, text1 and text2 in 3 columns
data = pd.read_csv("Text_Similarity_Dataset.csv") 

train_set = data.loc[:,'text1']
test_set = data.loc[:,'text2']

#simDf is for saving similarity score
import numpy
a = numpy.zeros(shape=(len(data),1))
simDf = pd.DataFrame(a,columns=['similarity'])

#iterate each row and get cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(data)):
    X = vectorizer.fit_transform([train_set[i]])
    y = vectorizer.transform([test_set[i]])
    simDf.iloc[i]=cosine_similarity(X,y)[0]

#output the result into submission.csv
output=pd.DataFrame({'Unique_ID':data.loc[:,'Unique_ID'],'Similarity_Score':simDf.similarity})
output.to_csv('submission.csv',index=False)