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

import classifier
warnings.filterwarnings('ignore')

import os

import intent_matcher as im

import python_weather


def find_name(query):
    tokenized_query = word_tokenize(query)
    tagged_query= nltk.pos_tag(tokenized_query, tagset='universal')
    # tagged_query_lem = nltk.pos_tag(lemm_q, tagset='universal')
    
    sent = nltk.ne_chunk(tagged_query, binary=False)
    return sent

    # for token in tagged_query:
    #     if token[1] == 'NOUN':
    #         return token[0]
        
    # return 'friend'

print(find_name("My name is Ella"))