import string

from urllib import request

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

#personal information
isUnderstood = False
username = "user"
subject = ""
topic = ""
def generateText(query):
    response = process_query(query)
    return response
    
    

def process_query(query):
    #tokenization
    tokenized_query = word_tokenize(query)

    #removing stop words
    #maybe make quotes a search prompt
    tokens_wo_sw= [word.lower() for word in tokenized_query if not word in stopwords.words()]
    filtered_query = ("").join(tokens_wo_sw)


    #stemming
    #can change type of stemmer if this isnt working
    # q_stemmer = PorterStemmer()
    sb_stemmer = SnowballStemmer('english')
    stemmed_q = []
    for token in tokens_wo_sw:
        stemmed_q.append(sb_stemmer.stem(token))
    

    #lemmentisation (without stemming or tagging)
    # lemm_q = []
    # lemmentiser = WordNetLemmatizer()
    # for token in tokens_wo_sw:
    #     lemm_q.append(lemmentiser.lemmatize(token))
    # print(lemm_q)
    
    #stemmed and tagged query
    tagged_query_stem = nltk.pos_tag(stemmed_q, tagset='universal')
    # tagged_query_lem = nltk.pos_tag(lemm_q, tagset='universal')
    
    postmap = {
    'ADJ': 'j',
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v'
    }
    
    #lemmentised and tagged query
    lemm_q = []
    lemmentiser = WordNetLemmatizer()
    tagged_query = nltk.pos_tag(lemm_q, tagset='universal')
    for token in tagged_query: 
        word = token
        tag = token[0]
        tag = token[1]
        if tag in postmap.keys():
            lemm_q.append(lemmentiser.lemmatize(word, postmap[tag]))
        else:
            lemm_q.append(lemmentiser.lemmatize(word))

    #only return false if no similar paper is found
    return falleback_response(query)


fallbackResponses = {1: "sorry, could you rephrase that",
                    2: "Could you please be more clear about what you're asking. Try and include some key words",
                    3: "Could you please be more specific",
                    4: "Would you like me to make a web search about ", #followed by query
                    5: "Sorry, but I do not think that I can help you with that. Please feel free to ask me about something else!"
}

fbr_index = 1
def falleback_response(query):
    global fbr_index
    fbr_index = 1 if fbr_index > 5 else fbr_index
    
    response = fallbackResponses[fbr_index]
    if(fbr_index == 4):
        response += "'" + query + "', yes/no?"
        print(response)
        answer = input()
        if(answer == 'yes'):
            searchWeb(query)
            response = "I hope that was helpful. Is there anything else that you'd like to ask?"
        else:
            response = "Sorry, but I'm not sure if I can help you with that query"
        
    fbr_index += 1
    return response

def searchWeb(query):
    return -1

def findName(name):
    return "unknown" #fix this to find the name within the query

#this is the main while loop which eveything else comes from. the program will stop when the user says bye
user = 2 #user 2 is the user, and 1 is the bot
BOT = 1
USER = 2

welcome_message = 'hello, I am studybot. How can I help you?'
print(welcome_message)
words = ["hello", "apple", "banannas", "chocolate"]

query = ""
done = False
while(done == False):
    if(user == BOT):
        print(generateText(query))
        user = USER
    elif(user == USER):
        query = input()
        if("bye" in query):
            print("I hope I was of good use! Goodbye! :)")
            done = True
        user = BOT
    else:
        print('error')