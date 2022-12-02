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

#change the urls - maybe use wikipedia pages
# urls = {'Lung cancer': 'http://gutenberg.org/files/1661/1661-0.txt',
#         'Kidney disiease': 'http://gutenberg.org/files/834/834-0.txt',
#         'Body parts': 'http://www.gutenberg.org/files/84/84-0.txt'}
# raw_documents = {}
# for book in urls:
#     content = request.urlopen(urls[book]).read().decode('utf8')
#     raw_documents[book] = content

# tokenizer = nltk.RegexpTokenizer(r"\w+")  # only keep words
# tok_documents = {}
# for book in raw_documents:
#     tok_documents[book] = tokenizer.tokenize(raw_documents[book])

# lowered_tok_documents = {}
# for book in tok_documents:
#     lowered_tok_documents[book] = [word.lower() for word in tok_documents[book]]

# filtered_documents = {}
# english_stopwords = stopwords.words('english')
# for book in lowered_tok_documents:
#     filtered_documents[book] = [word for word in lowered_tok_documents[book]
#                                 if word not in english_stopwords]
   
# from nltk.stem.snowball import SnowballStemmer

# sb_stemmer = SnowballStemmer('english')
# stemmed_documents = {}
# for book in filtered_documents:
#     stemmed_documents[book] = [sb_stemmer.stem(word)
#                                for word in filtered_documents[book]]
    # print(f'{book}: {stemmed_documents[book][500:510]}')

vocabulary = []
for book in stemmed_documents:
    for stem in stemmed_documents[book]:
        if stem not in vocabulary:
            vocabulary.append(stem)
# print(f'VOCABULARY: {vocabulary[500:510]}, total length: {len(vocabulary)}')

import numpy as np

bow = {}
for book in stemmed_documents:
    bow[book] = np.zeros(len(vocabulary))
    for stem in stemmed_documents[book]:
        index = vocabulary.index(stem)
        bow[book][index] += 1
    print(f'{book} bag-of-word: {bow[book][500:510]}')

matrix = np.vstack([bow[key] for key in bow])
print(f"Document-term matrix {matrix.shape}:\n{matrix}")
print(f"Term-document matrix {matrix.transpose().shape}:\n{matrix.transpose()}")


from scipy import sparse
from sys import getsizeof

sparse_matrix = sparse.csr_matrix(matrix)
print(f"Size of array: {getsizeof(matrix)} B")
print(f"Size of sparse matrix: {getsizeof(sparse_matrix)} B")

print(matrix)
np.save('./td_matrix.npy', matrix)

loaded_matrix = np.load('./td_matrix.npy')
print(loaded_matrix)

def binary_weighting(vector):
    b_vector = np.array(vector, dtype=bool)  # convert into bool (True/False)
    b_vector = np.array(b_vector, dtype=int)  # convert bool into int (True/False become 1/0)
    return b_vector

binary_bow = {}
for book in bow:
    binary_bow[book] = binary_weighting(bow[book])
    print(f'{book} bag-of-word (binary weighted): {binary_bow[book]}')

from math import log10

def logfreq_weighting(vector):
    lf_vector = []
    for frequency in vector:
        lf_vector.append(log10(1+frequency))
    return np.array(lf_vector)

logfreq_bow = {}
for book in bow:
    logfreq_bow[book] = logfreq_weighting(bow[book])
    print(f'{book} bag-of-word (logfreq weighted): {logfreq_bow[book]}')

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
    print(f'{pair}: {similarity}')

query = input("Enter your query: ")

tokenizer = nltk.RegexpTokenizer(r"\w+")
tok_query = tokenizer.tokenize(query)

lowered_tok_query = [word.lower() for word in tok_query]

english_stopwords = stopwords.words('english')
filtered_query = [word for word in lowered_tok_query
                  if word not in english_stopwords]

             
lemmentiser = WordNetLemmatizer()
def lemmed_words (text):
    tokenized_query = word_tokenize(text)
    tokens_wo_sw= [word.lower() for word in tokenized_query if not word in stopwords.words()]
    # print(tokens_wo_sw)
    
    postmap = {
    'ADJ': 'j',
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v'
    }
    
    #lemmentised and tagged query
    lemm_q = []
    lemmentiser = WordNetLemmatizer()
    tagged_query = nltk.pos_tag(tokens_wo_sw, tagset='universal')
    for token in tagged_query: 
        word = token[0]
        tag = token[1]
        if tag in postmap.keys():
            lemm_q.append(lemmentiser.lemmatize(word, postmap[tag]))
        else:
            lemm_q.append(lemmentiser.lemmatize(word))
    return lemm_q

words = []
for word in filtered_query:
    words.append(lemmed_words(word))
lemmed_query = words

vector_query = np.zeros(len(vocabulary))
for stem in lemmed_query:
    if stem in vocabulary:
        index = vocabulary.index(stem)
        vector_query[index] += 1
logfreq_vector_query = logfreq_weighting(vector_query)


for subject in bow.keys():
    similarity = sim_cosine(logfreq_bow[book], logfreq_vector_query)
    # print(f'Similarity with {book}: {similarity}')

from sklearn.model_selection import train_test_split

label_dir = {
"history": "Datasets/training/history",
"medicine": "Datasets/training/medicine",
"greetings": "Datasets/training/greetings"
}

data = []
labels = []

for label in label_dir.keys() :
    for file in os.listdir ( label_dir [ label ]) :
        filepath = label_dir[ label ] + os.sep + file
        with open(filepath, encoding ='utf8', errors ='ignore', mode ='r') as review:
            content = review.read ()
            data.append ( content )
            labels.append ( label )

X_train , X_test , y_train , y_test = train_test_split ( data , labels , stratify = labels, test_size =0.25 , random_state =1)

lemmentiser = WordNetLemmatizer()
analyser = CountVectorizer().build_analyzer()



count_vect = CountVectorizer (stop_words = stopwords . words ('english'))
X_train_counts = count_vect . fit_transform ( X_train )

tfidf_transformer = TfidfTransformer ( use_idf = True , sublinear_tf = True ). fit (
X_train_counts )
X_train_tf = tfidf_transformer . transform ( X_train_counts )

clf = LogisticRegression ( random_state =0) . fit ( X_train_tf , y_train )

from sklearn.metrics import accuracy_score , f1_score , confusion_matrix

X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

X_train_lf = np.apply_along_axis(logfreq_weighting, 0, X_train_counts.toarray())
X_train_lf = sparse.csr_matrix(X_train_lf)
# print(X_train_lf)

from sklearn.svm import SVC

new_clf = SVC(C=1.0, random_state=0).fit(X_train_lf, y_train)

X_new_lf = np.apply_along_axis(logfreq_weighting, 0, X_new_counts.toarray())
X_new_lf = sparse.csr_matrix(X_new_lf)


