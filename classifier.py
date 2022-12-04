import string

from urllib import request
from urllib.request import FancyURLopener

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize , sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams , bigrams
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends
from nltk import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import bs4 as bs
from bs4 import BeautifulSoup as bsoup
from bs4 import SoupStrainer

# nltk.download('universal_tagset')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
import scipy
from scipy import spatial

import numpy
from numpy import dot

import sqlite3
connection = sqlite3.connect('database.db')

import warnings
warnings.filterwarnings('ignore')

from sklearn . model_selection import train_test_split
from sklearn . feature_extraction . text import CountVectorizer
from sklearn . feature_extraction . text import TfidfTransformer
from sklearn . linear_model import LogisticRegression

import os

import wikipedia

# result = wikipedia.search("Lung cancer")
# page = wikipedia.page(result[0])
# title = page.title
# categories = page.categories
# # print("categories: "+ categories)
# content = page.content
# # print("content: " + content)
# links = page.links
# refereces = page.references
# summary = page.summary
bow = {}
import numpy as np

def extract_content(page_name):
    r = wikipedia.search(page_name)
    if len(r) > 0:
        return wikipedia.page(r[0]).content
    else:
        print("error - none of the books on my shelf have any info on that!")

wikipages = {
    'Lung cancer': extract_content('Lung cancer'),
    'Kidney disease': extract_content('Kidney disease'),
    'The heart': extract_content('The heart'),
}

def calculate(): ##need to recalculate information
    process_documents()
    calc_matrix()
    b_bow()
    calc_bow()


## tokenisation
def process_documents():
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # only keep words
    tok_documents = {}
    for page in wikipages:
        tok_documents[page] = tokenizer.tokenize(wikipages[page])

    #lowercase
    lowered_doc = {}
    for page in tok_documents:
        lowered_doc[page] = [word.lower() for word in tok_documents[page]]

    #removing stopwords
    filtered_pages = {}
    english_stopwords = stopwords.words('english')
    for page in lowered_doc:
        filtered_pages[page] = [word for word in lowered_doc[page] if word not in english_stopwords]

    #change to lemmentisation if time
    sb_stemmer = SnowballStemmer('english')
    stemmed_documents = {}
    for book in filtered_pages:
        stemmed_documents[book] = [sb_stemmer.stem(word) for word in filtered_pages[book]]

    global vocabulary
    vocabulary = []
    for book in stemmed_documents:
        for stem in stemmed_documents[book]:
            if stem not in vocabulary:
                vocabulary.append(stem)

    global bow
    bow = {}
    for book in stemmed_documents:
        bow[book] = np.zeros(len(vocabulary))
        for stem in stemmed_documents[book]:
            index = vocabulary.index(stem)
            bow[book][index] += 1
        # print(f'{book} bag-of-word: {bow[book]}')

def calc_matrix():
    matrix = np.vstack([bow[key] for key in bow])
    # print(f"Document-term matrix {matrix.shape}:\n{matrix}")
    # print(f"Term-document matrix {matrix.transpose().shape}:\n{matrix.transpose()}")

    from scipy import sparse
    from sys import getsizeof
    sparse_matrix = sparse.csr_matrix(matrix)
    # print(f"Size of array: {getsizeof(matrix)} B")
    # print(f"Size of sparse matrix: {getsizeof(sparse_matrix)} B")

    np.save('./td_matrix.npy', matrix)  #saving matrix using pickle

    loaded_matrix = np.load('./td_matrix.npy') #loading matrix

def binary_weighting(vector):
    b_vector = np.array(vector, dtype=bool)  # convert into bool (True/False)
    b_vector = np.array(b_vector, dtype=int)  # convert bool into int (True/False become 1/0)
    return b_vector

def b_bow():
    binary_bow = {}
    for book in bow:
        binary_bow[book] = binary_weighting(bow[book])
        # print(f'{book} bag-of-word (binary weighted): {binary_bow[book]}')


from math import log10

def logfreq_weighting(vector):
    lf_vector = []
    for frequency in vector:
        lf_vector.append(log10(1+frequency))
    return np.array(lf_vector)


def calc_bow():
    global logfreq_bow
    logfreq_bow = {}
    for book in bow:
        logfreq_bow[book] = logfreq_weighting(bow[book])
        # print(f'{book} bag-of-word (logfreq weighted): {logfreq_bow[book]}')


def tfidf_weighting(vector_1, vector_2):
    N = 2
    tfidf_vector_1 = np.zeros(len(vector_1))
    tfidf_vector_2 = np.zeros(len(vector_2))
    for i in range(len(vector_1)):
        # Get n
        term_booleans = [vector_1[i]!=0, vector_2[i]!=0]
        n = sum(term_booleans)
        # Compute TF-IDF for each vector
        frequency_1 = vector_1[i]
        tfidf_1 = log10(1+frequency_1) * log10(N/n)
        tfidf_vector_1[i] = tfidf_1
        frequency_2 = vector_2[i]
        tfidf_2 = log10(1+frequency_2) * log10(N/n)
        tfidf_vector_2[i] = tfidf_2
    return tfidf_vector_1, tfidf_vector_2

from scipy import spatial

def sim_cosine(vector_1, vector_2):
    similarity = 1 - spatial.distance.cosine(vector_1, vector_2)
    return similarity

import itertools

for pair in itertools.combinations(bow.keys(), 2):
    similarity = sim_cosine(bow[pair[0]], bow[pair[1]])
    # print(f'{pair}: {similarity}')

logfreq_vector_query = 0
def processQuery(query):

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tok_query = tokenizer.tokenize(query)

    # Lower casing
    lowered_tok_query = [word.lower() for word in tok_query]

    # Remove stopwords and lower casing
    english_stopwords = stopwords.words('english')
    filtered_query = [word for word in lowered_tok_query
                    if word not in english_stopwords]

    # Stemming
    sb_stemmer = SnowballStemmer('english')
    stemmed_query = [sb_stemmer.stem(word) for word in filtered_query]

    vector_query = np.zeros(len(vocabulary))
    for stem in stemmed_query:
        if stem in vocabulary:
            index = vocabulary.index(stem)
            vector_query[index] += 1
    # print(f"Query bag-of-word:\n{vector_query}")
    # print(sparse.csr_matrix(vector_query))
    global logfreq_vector_query
    logfreq_vector_query = logfreq_weighting(vector_query)
    # print(f"Query bag-of-word (logfreq weighted):\n{logfreq_vector_query}")
    # print(sparse.csr_matrix(logfreq_vector_query))

def most_similar(query):
    processQuery(query=query)
    similarities = {}
    for book in bow.keys():
        similarity = sim_cosine(logfreq_bow[book], logfreq_vector_query)
        if(similarity > 0):
            similarities[book] = similarity
        # print(f'Similarity with {book}: {similarity}')
    return similarities

def get_summary(topic):
    result = wikipedia.search(topic)
    page = wikipedia.page(result[0])
    return page.summary

calculate()