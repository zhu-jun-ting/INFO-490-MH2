#!/usr/bin/env python
# coding: utf-8

# In[9]:


##TF

def get_corpus():
    c1 = "Do you like Green eggs and ham"
    c2 = "I do not like them Sam I am I do not like Green eggs and ham"
    c3 = "Would you like them Here or there"
    c4 = "I would not like them Here or there I would not like them Anywhere"

    return [c1, c2, c3, c4]


def split_into_tokens(data, normalize=True, min_length=0):
    tokens = data.split(" ")

    if (normalize):
        tokens = map(lambda token: str.lower(token), tokens)

    if (min_length != None):
        tokens = filter(lambda token: len(token) >= min_length, tokens)

    return list(tokens)


# build TF

import collections
from functools import reduce
import math


def build_tf(corpus, min_length=0):
    vocabs = []
    tfs = []

    for document in corpus:
        vocab = collections.Counter(split_into_tokens(document, normalize=True, min_length=min_length))
        vocabs.append(vocab)

        tf = dict()
        for item in list(vocab):
#             tf[item] = vocab[item] / sum(vocab.values())
            tf[item] = 1 + math.log(vocab[item])

        tfs.append(tf)

    corpus_counter = reduce(lambda x, y: x + y, vocabs)

    # return Counter for all words at [0] and the dictionary of Term Frequency for each document at [1]

    return corpus_counter, tfs


##IDF




def build_idf(vocabulary, corpus_tf):
    idf = dict()

    for term in list(vocabulary):
        term_count = 0
        for doc_tf in corpus_tf:
            if term in doc_tf:
                term_count += 1
        # print(term_count)
        idf[term] = math.log(len(corpus_tf) / term_count)

    return idf


# '''
# ***********************************************
# This part is required for grading
# Lesson Assignment
# ***********************************************
# '''
def compute_TFIDF():
    return


def build_tf_idf():
    return


# test_tfidf()
# '''
# ***********************************************
# This part is required for grading
# Short Answers to the Questions
# Q1)What is Term Frequency (TF)
# Q2)What is Inverse Document Frequency (IDF)
# Q3)What does TF???IDF measure?

# ##Please write your observations here about the words that would
# help differentiate the documents
# ##
# ##
# ##
# ***********************************************
# '''

split_into_tokens(get_corpus()[0], normalize=True, min_length=4)

vocab, tfs = build_tf(get_corpus(), min_length=0)
# vocab

vocab

tfs

math.log(4 / 1)

build_idf(vocab, tfs)


def compute_TFIDF(doc_tf, idf):
    # returns a collection counter for the document tf and the corpus idf
    tf = collections.Counter()
    for vocab in doc_tf:
        tf[vocab] = doc_tf[vocab] * idf[vocab]
    return tf


def build_tf_idf(tfs, idf):
    tfidf = [collections.Counter() for x in tfs]
    for idx, doc_tf in enumerate(tfs):
        tfidf[idx] = compute_TFIDF(doc_tf, idf)
    return tfidf


def test_tfidf():
    corpus = get_corpus()
    vocab, tf = build_tf(corpus)
    idf = build_idf(vocab, tf)
    tfidf = build_tf_idf(tf, idf)
    return tfidf


test_tfidf()
print("hello")


# the tf*idf value indicates larger valued words can differentiate paragraphs better, such as: "ham", "I", "eggs", etc.

# ***********************************************
# This part is required for grading
# Short Answers to the Questions
# Q1)What is Term Frequency (TF): term frequency determines how often a word occurs in a document
# Q2)What is Inverse Document Frequency (IDF): inverse document frequency determines how many documents there contains a certain word
# Q3)What does TF???IDF measure?: TF*IDF measures what are the more important words (with higher TF*IDF) and what are the more common stop words


# In[ ]:




