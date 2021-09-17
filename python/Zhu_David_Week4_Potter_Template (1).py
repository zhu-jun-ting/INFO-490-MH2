#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install Wordcloud


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traitlets 
import ipywidgets
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# In[3]:


'''
*********************************************************
This part is required for grading                       *
Practice Problems && Lesson Assignment                  *
Fill in all the functions and compare your results      *
You should run your file before submission              *
Only files with .py extension are accepted              *
*********************************************************
'''

def read_data_file(filename):
    with open(filename, 'r') as fd:
        return fd.read()

def load_stopwords(extra=[]):
    return extra + ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


def get_harry_potter():
  return read_data_file('hp1.txt')

#print(get_harry_potter()[0:17])

def build_corpus(count=1):
  corpus = []
  return corpus

def test_corpus():
  return None
#test_corpus()


# def build_tf_idf_model(docs, stopwords=[]):
#   return vectorizer, matrix


def test_build(corpus):
  pass

def print_tfidf(vec, matrix, n=0):
  pass

def test_print():
  pass

#test_print()

def prepare_query_v1(corpus=None, query=''):
  return None

def dump_sparse_vector(v):
  pass

def test_single_query(query):
  pass
#test_single_query('harry potter')

def print_matching_document(matrix, q_vector):
  pass

def find_matching_document(matrix, q_vector):
  
  #print_matching_document(matrix, q_vector)
  return (None, None)

def find_match(query='', corpus_size=3):
  return None

#print(find_match(query="harry potter pretend", corpus_size=3))
#print(find_match(query="muggle",corpus_size=3)) 


def show_image(path):
  return None

def test_UI(vec=None, matrix=None, query='', debug=False):
  return None

#don't forget to test and compare your results
# corpus = build_corpus(7)
# (vec, matrix) = build_tf_idf_model(corpus)
# print(test_UI(vec, matrix, 'muggle', debug = True))
# test_UI(vec, matrix, 'winky')
# test_UI(vec, matrix, 'lupin')
# test_UI(vec, matrix, 'lupin champions')

'''
***********************************************
********************END************************
***********************************************
'''


# In[4]:


hp_book_paths = ["hp/hp{}.txt".format(x) for x in range(0, 8)]
hp_book_paths[2]


# In[5]:


#read_data_file function is already written in the template file
def get_harry_potter():
  return read_data_file('hp/hp1.txt')
print(get_harry_potter()[0:17])


# In[6]:


def build_corpus(count=1):
    result = []
    for i in range(1, count + 1):
        result.append(read_data_file(hp_book_paths[i]))
    return result
# len(build_corpus(2)) == 2


# In[7]:


def get_corpus(index=1):
    if index < 1 or index > 7:
        return "no such file"
    return read_data_file(hp_book_paths[index])
get_corpus(9)


# In[8]:


def test_corpus():
  c = build_corpus(2)
  print(len(c) == 2)
  print(c[0][0:17])
  print(c[1][0:31])
  doc1 = c[0]
  print(len(doc1.split()), len(set(doc1.split())))
test_corpus()


# In[9]:


wordcloud = WordCloud().generate(str(build_corpus(7)))
get_ipython().run_line_magic('pylab', 'inline')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[10]:


def build_tf_idf_model(docs, stopwords=[]):
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True, norm=None, stop_words=stopwords)
    tfidf_table = vectorizer.fit_transform(docs)
#     print(vectorizer.get_feature_names()[0: 30])
    return vectorizer, tfidf_table

print(build_tf_idf_model(build_corpus(2), load_stopwords()))


# In[11]:


def test_build(corpus):
    vec, matrix = build_tf_idf_model(corpus)
#     print(matrix.shape)
test_build(build_corpus(1))


# In[12]:


def print_tfidf(vec,matrix, n=0):
    features = vec.get_feature_names() # the unique words
    doc_vector = matrix[n]
    df = pd.DataFrame(doc_vector.T.todense(), index=features, columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False, inplace=True)
    print(df.head(20))
def test_print():
    corpus = build_corpus(1)
    vec, matrix = build_tf_idf_model(corpus, load_stopwords())
    print_tfidf(vec,matrix)
test_print()
plt.imshow(wordcloud, interpolation='bilinear')


# In[13]:


def prepare_query_v1(corpus=build_corpus(3), query=''):
    vectorizer, matrix = build_tf_idf_model(corpus, load_stopwords())
    query_matrix = vectorizer.transform([query])
    return query_matrix


def dump_sparse_vector(v):
  coo_m = v.tocoo()
  for r,c,d in zip(coo_m.row, coo_m.col, coo_m.data):
    print('non zero at', r,c,d)
    
def test_single_query(query):
  q_vec = prepare_query_v1(query=query)
  dump_sparse_vector(q_vec)
    
# test_single_query('harry potter')

def test_my_input_query():
    @ipywidgets.interact(query=ipywidgets.Text("harry potter"))
    def _query_that(query):
        print(test_single_query(query))

# test_my_input_query()

prepare_query_v1(query='harry potter')


# In[14]:


def print_matching_document(matrix, q_vector):
    assert q_vector.shape[0] == 1, "bad query vector (wrong size)"
    result = []
    for m_idx, m in enumerate(matrix):
        for q_idx, q in enumerate(q_vector):
            result.append((m_idx + 1, cosine_similarity(m,q)[0][0]))
#             print((m_idx + 1, cosine_similarity(m,q)[0][0]))
    return result

def print_matching_document_with_vec_and_matrix(vectorizer, matrix, q_vector):
    assert q_vector.shape[0] == 1, "bad query vector (wrong size)"
    result = []
    for m_idx, m in enumerate(matrix):
        for q_idx, q in enumerate(q_vector):
            result.append((m_idx + 1, cosine_similarity(m,q)[0][0]))
#             print((m_idx + 1, cosine_similarity(m,q)[0][0]))
    return result

search_range = build_corpus(7)
vectorizer, matrix = build_tf_idf_model(search_range, load_stopwords())
query_matrix = prepare_query_v1(corpus=search_range, query='harry potter')
print(matrix.shape, query_matrix.shape)
tuples = print_matching_document(matrix, query_matrix)
sorted(tuples, key=lambda x: x[1], reverse=True)[0]


# In[15]:


def find_match(volumns_number=7, debug=False, show_best_match=True):
    @ipywidgets.interact(volumns=ipywidgets.IntSlider(value=3, min=1, max=7, step=1), query=ipywidgets.Text("harry potter pretend"))
    def _show_search_results(volumns, query):
        search_range = build_corpus(volumns)
        vectorizer, source_matrix = build_tf_idf_model(search_range, load_stopwords())
        query_matrix = prepare_query_v1(corpus=search_range, query=query)
        assert source_matrix.shape[1] == query_matrix.shape[1], "search range does not match in query."
        result_tuples = print_matching_document(source_matrix, query_matrix)
        if debug:
            print(result_tuples)
        if show_best_match:
            best_volumn = sorted(result_tuples, key=lambda x: x[1], reverse=True)[0]
#             print(sorted(result_tuples, key=lambda x: x[1], reverse=True)[0])
            print('*'*100)
            print("THE BEST MATCHING VOLUMN OF QUERY STRING '{}' IN HARRY POTTER SERIES IS NO.{} WITH COSINE SIMILARITY OF {}.".format(query, best_volumn[0], best_volumn[1]))
    return _show_search_results

find_match(volumns_number=7, debug=True, show_best_match=True)


# In[16]:


def build_query_matrix(vectorizer, matrix, query):
#     vectorizer, matrix = build_tf_idf_model(corpus, load_stopwords())
    query_matrix = vectorizer.transform([query])
    return query_matrix

def get_result_tuples(vectorizer, matrix, q_vector):
    assert q_vector.shape[0] == 1, "bad query vector (wrong size)"
    result = []
    for m_idx, m in enumerate(matrix):
        for q_idx, q in enumerate(q_vector):
            result.append((m_idx + 1, cosine_similarity(m,q)[0][0]))
#             print((m_idx + 1, cosine_similarity(m,q)[0][0]))
    return result


# In[17]:


def show_image(path):
    with open(path, 'rb') as fd:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        imgdata = plt.imread(fd)
        im = ax.imshow(imgdata)
        return fig

def get_hp_image_pathlib():
    return {
        1: 'hp/hp1.png',
        2: 'hp/hp2.png',
        3: 'hp/hp3.png',
        4: 'hp/hp4.png',
        5: 'hp/hp5.png',
        6: 'hp/hp6.png',
        7: 'hp/hp7.png'
    }
    
def test_UI(vec=None, matrix=None, query='', debug=False):
  # build vec, matrix if either parameter is None
  # use the full harry potter corpus if you need to build
    if vec == None or matrix == None:
        volumns = 7
        vec, matrix = build_tf_idf_model(build_corpus(volumns), load_stopwords())
  # transform the query string
    query_matrix = build_query_matrix(vec, matrix, query)
  #
  # find the matching document
    result_tuples = get_result_tuples(vec, matrix, query_matrix)
    best_volumn = sorted(result_tuples, key=lambda x: x[1], reverse=True)[0]
  #
  # if no document matches, use the first document
    if best_volumn[1] == 0.0:
        best_volumn = (1, 0.0)
  #
  # get path for the image
    path_lib = get_hp_image_pathlib()
    
    
  # display the image

    show_image(path_lib[best_volumn[0]])

  # show_image(path)
  # return the winning index
#     if debug then print out all similarities
    
    if debug:
        for volumn, similarity in result_tuples:
            print('{}: {}'.format(volumn, similarity))
    
    return best_volumn[0]

test_UI(vec=None, matrix=None, query='death', debug=True)


# In[18]:


def interactive_UI(volumns_number=7, debug=False, show_best_match=True):
    @ipywidgets.interact(volumns=ipywidgets.IntSlider(value=3, min=1, max=7, step=1), query=ipywidgets.Text("harry potter pretend"))
    def _show_search_results(volumns, query):
        search_range = build_corpus(volumns)
        vectorizer, source_matrix = build_tf_idf_model(search_range, load_stopwords())
        query_matrix = prepare_query_v1(corpus=search_range, query=query)
        assert source_matrix.shape[1] == query_matrix.shape[1], "search range does not match in query."
        result_tuples = print_matching_document(source_matrix, query_matrix)
        if debug:
            print(result_tuples)
        if show_best_match:
            best_volumn = sorted(result_tuples, key=lambda x: x[1], reverse=True)[0]
#             print(sorted(result_tuples, key=lambda x: x[1], reverse=True)[0])
            print('*'*100)
            print("THE BEST MATCHING VOLUMN OF QUERY STRING '{}' IN HARRY POTTER SERIES IS NO.{} WITH COSINE SIMILARITY OF {}.".format(query, best_volumn[0], best_volumn[1]))
            path_lib = get_hp_image_pathlib()
            show_image(path_lib[best_volumn[0]])
            print('*'*100)
            
    return _show_search_results

interactive_UI(volumns_number=7, debug=False, show_best_match=True)

