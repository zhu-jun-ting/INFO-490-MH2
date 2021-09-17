#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# import zhu_david_week2_tfidf_template as tf_idf


# In[12]:


get_ipython().run_cell_magic('capture', '', '%run "zhu_david_week2_tfidf_template.ipynb"\n\n%matplotlib inline')


# In[3]:


def cv_demo1():
    corpus = get_corpus()
    # normalize all the words to lowercase
    cvec = CountVectorizer(lowercase=True)
    # convert the documents into a document-term matrix
    doc_term_matrix = cvec.fit_transform(corpus)
    # 
    print(cvec.get_feature_names())
    # get the counts
    print(doc_term_matrix.toarray())
    
cv_demo1()


# In[4]:


def cv_demo2():
    corpus = get_corpus()
    # pass in our own tokenizer
    cvec = CountVectorizer(tokenizer=split_into_tokens)
    # convert the documents into a document-term matrix
    doc_term_matrix = cvec.fit_transform(corpus)
    # get the terms found in the corpus
    tokens = cvec.get_feature_names()
    return doc_term_matrix, tokens

dtm, tokens = cv_demo2()


# In[5]:


def word_matrix_to_df(wm, feature_names):
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx+1) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names, columns=feature_names)
    return df
def cv_demo3():
    doc_term_matrix, tokens = cv_demo2()
    df = word_matrix_to_df(doc_term_matrix, tokens)
    return df
df = cv_demo3()
print(df.head())


# In[6]:


def cv_demo_idf():
    # get the data from the CountVectorizer
    doc_term_matrix, tokens = cv_demo2()
    # create the tf•idf transformer
    tfidf_transformer=TfidfTransformer()
    # transform the doc_term_matrix into TF•IDF
    tfidf_transformer.fit(doc_term_matrix)
    # make it a dataframe for easy viewing
    df = pd.DataFrame(tfidf_transformer.idf_, index=tokens, columns=["idf_weights"])
#     sort descending
    df.sort_values(by=['idf_weights'], inplace=True, ascending=False)
    return df

df = cv_demo_idf()
print(df.head(30))


# In[7]:


def cv_demo_tf_idf():
    doc_term_matrix, tokens = cv_demo2()
    tfidf_transformer=TfidfTransformer(smooth_idf=True)
    # learn the IDF vector
    tfidf_transformer.fit(doc_term_matrix)
    idf = tfidf_transformer.idf_
    # transform the count matrix to tf-idf
    tf_idf_vector = tfidf_transformer.transform(doc_term_matrix)
    print(tf_idf_vector)

cv_demo_tf_idf()


# In[8]:


def cv_demo_pd_tf_idf():
    doc_term_matrix, tokens = cv_demo2()
    tfidf_transformer = TfidfTransformer(smooth_idf=True)
#     tfidf_transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, norm=None)
    # learn the IDF vector
    tfidf_transformer.fit(doc_term_matrix)
    idf = tfidf_transformer.idf_
    # transform the count matrix to tf-idf
    tf_idf_vector = tfidf_transformer.transform(doc_term_matrix)
    # print out the values
    # for the token 'i' in the second document
    token = 'i'
    doc = 1
    df_idf = pd.DataFrame(idf, index=tokens, columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'], inplace=True, ascending=False)
    idf_token = df_idf.loc[token]['idf_weights']
    doc_vector = tf_idf_vector[doc]
    df_tfidf = pd.DataFrame(doc_vector.T.todense(), index=tokens, columns=["tfidf"])
    df_tfidf.sort_values(by=["tfidf"], ascending=False, inplace=True)
    tfidf_token = df_tfidf.loc[token]['tfidf']
    # tfidf = tf * idf
    tf_token = tfidf_token / idf_token
    print('TF    {:s} {:2.4f}'.format(token, tf_token))
    print('IDF   {:s} {:2.4f}'.format(token, idf_token))
    print('TFIDF {:s} {:2.4f}'.format(token, tfidf_token))

cv_demo_pd_tf_idf()

# Outputs:

# TfidfTransformer(smooth_idf=True, sublinear_tf=True, norm=None): 
# TF    i 2.0986
# IDF   i 1.5108
# TFIDF i 3.1706

# tfidf_transformer = TfidfTransformer(smooth_idf=True)
# TF    i 0.3848
# IDF   i 1.5108
# TFIDF i 0.5814


# In[34]:


corpus = get_corpus()
cv = TfidfVectorizer(smooth_idf=True, use_idf=True, tokenizer=split_into_tokens, norm=None)
tfidf = cv.fit_transform(corpus)
tokens = cv.get_feature_names()
idf = cv.idf_
values = tfidf.todense().tolist()

# print(tfidf)
# print('*' * 100)
# print(tfidf.todense())
# print('*' * 100)
# print(values)

# print(pd.DataFrame(tfidf.todense(), columns=tokens, index=np.arange(1, 5)).plot(kind='bar', figsize=(20, 10)))
df = pd.DataFrame(tfidf.todense(), columns=tokens, index=np.arange(1, 5))
print(df)
ax = df.plot.bar()


# In[10]:


def dump_sparse_matrix():
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(use_idf=True)
    corpus = ["another day of rain; rain rain go away, comeback another day"]
    matrix = vec.fit_transform(corpus)
    print(matrix.shape)
    print(vec.idf_)  # all 1's (there's only 1 document)
    coo_format = matrix.tocoo()
    print(coo_format.col)
    print(coo_format.data)
    tuples = zip(coo_format.col, coo_format.data)
    in_order = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    features = vec.get_feature_names() # the unique words
    print(features)
    for score in in_order:
        idx = score[0]
        word = features[idx]
        print("{:10s} tfidf:".format(word), score)
dump_sparse_matrix()


# In[11]:


# ***********************************************
# Review Questions
# Q1)What does CountVectorizer do? CountVectorizer gives a interpreter to convert a corpus to document term matrix
# Q2)What does TfidfTransformer do? TfidfTransformer transforms a document to tfidf values based on terms
# Q3)What does sklearn's fit function do? Fit function trys to build the model based on the incoming dataset
# Q4)What does sklearn's transform function do? Transform applys the trained model to the given dataset
# ##
# ##
# ##
# ***********************************************

