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

#personal information
username = "user"
subject = ""
topic = ""

# keepwords = ["how", "what", "when", "where", "why"]
# new_words = list(filter(lambda w: w in keepwords, stopwords))


#to do list:
# - pre-process text
# - cosine simimarity to find 
# - 
def generateText(query):
    return -1
    
def download_documents(keywords):
    url = "https://scholar.google.com/?q="
    class MyOpener(FancyURLopener):
        version = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.152 Safari/537.36'
    openurl = MyOpener().open
    raw = request.urlopen(url)

def find_topic(subject):
    path = 'Datasets\\topic_list\\' + subject
    for file in os.listdir(path):
        with open(path, encoding='utf-8', errors='ignore', mode='r') as document:
            content = document.read()
            topic = document.name
        print(topic)

def find_keywords(query):
    #tokenization
    tokenized_query = word_tokenize(query)

    #removing stop words
    #maybe make quotes a search prompt
    tokens_wo_sw= [word.lower() for word in tokenized_query if not word in stopwords.words()]
    # filtered_query = ("").join(tokens_wo_sw)

    tagged_query = nltk.pos_tag(tokens_wo_sw, tagset='universal')

    keywords = ""
    for token in tagged_query:
         if token[1] == "NOUN" or token[1] == "VERB":
            keywords += token[0] + " "
    
    return keywords


subject = "medicine"
def process_documents(subject):
    corpus = {}
    #datasets\\has all of the files related to that topic
    path = 'Datasets\\' + subject

    string.punctuation = string.punctuation + "'"+"-"+"'"+"-"

    string.punctuation = string.punctuation.replace(".", "")

    for file in os.listdir(path):
        filepath = path + os.sep + file
        with open(filepath, encoding='utf-8', errors='ignore', mode='r') as document:
            content = document.read()
            document_id = file
            corpus[document_id] = content
            #entire folder of documents held in this corpus
    
    #now we have a corpus with all of the files for that subject. what do we do next?

    
    file = open("Datasets\\TestReadingQu.txt", encoding = "utf8").read()

    file_nl_removed = ""
    for line in file:
        line_nl_removed = line.replace("\n", " ") #removing newline characters
        file_nl_removed += line_nl_removed
    file_p = "".join([char for char in file_nl_removed if char not in string.punctuation])
    tokenised_file = process_text(file_p)
    return tokenised_file

def process_text(query):
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
        word = token[0]
        tag = token[1]
        if tag in postmap.keys():
            lemm_q.append(lemmentiser.lemmatize(word, postmap[tag]))
        else:
            lemm_q.append(lemmentiser.lemmatize(word))

    return lemm_q
    # #only return false if no similar paper is found
    # return falleback_response(query)

def add_to_dataset(keywords):
    content = classifier.extract_content(keywords)
    classifier.wikipages[keywords] = content
    classifier.calculate()
    

fallbackResponses = {
        1: "Could you please be more clear about what you're asking. Try and include some key words.",
        2: "Sorry, but I do not know much about ",#followed by query
        3: "Could you please be more specific?", 
        4: "Sorry, but I'm having trouble understanding your question."
}


fbr_index = 1

#handle fallback responses in here
def fallback_response(query):
    global fbr_index
    fbr_index = 1 if fbr_index > len(fallbackResponses) else fbr_index
    
    response = fallbackResponses[fbr_index]
    if(fbr_index == 2):
        response += "'" + query + "', but let me read about it!"
        print(response)
        add_to_dataset(find_keywords(query)) ##returns string of information about the topic and adds page to database
        chop_response(find_keywords(query)) ##takes in entire string
        print("This is the best information that I could find from my research")
        print("I hope that this was helpful!")
        response = ""
    
    fbr_index += 1
    return response

def no_inp(s):
    s.lower()
    if s == "no" or s == "n":
        return True
    else:
        return False

def respond(query):
    #match intent, and if intent is a search question (not small talk or any other intent) then conduct search
    results = classifier.most_similar(query)
    answer_list = [] #l1
    similarity_list = [] #l2
    for key in results:
        answer_list.append(key)
        similarity_list.append(results[key]) 
    
    zipped_pairs = zip(similarity_list, answer_list)
    answer_list = [x for _, x in sorted(zipped_pairs, reverse=True)]
    sorted(similarity_list, reverse=True)

    print("sorted !!")
    print(answer_list)

    resp = False
    for response in range(len(answer_list)):
        ans = input("Are you asking about "+ answer_list[response] + "? (y/n)\n")
        if no_inp(ans) == False: #if they say yes
            chop_response(answer_list[response])
                
            resp = True #has responded, and therefore we do not need a fallback response after exiting the loop
            print()
            cont = input("I hope this was helpful! Is there anything else that you'd like to ask?\n")
            if no_inp(cont):
                print("So glad that I could help! Bye!")
                quit()
            else:
                return
    if(resp == False):
        print(fallback_response(query))

def chop_response(topic):
    st = classifier.get_summary(topic)
    resp_list = st.split("\n")

    for i in range(len(resp_list)):
        print(resp_list[i])
        print()
        if i < len(resp_list)-1:
            inp = input("Should I continue giving you information about "+ topic + "?\n").lower()
            if no_inp(inp):
                return -1

def searchWeb(query):
    return -1

def findName(query):
    print("Checking!!!!!")
    tokenized_query = word_tokenize(query)
    tagged_query= nltk.pos_tag(tokenized_query, tagset='universal')
    # tagged_query_lem = nltk.pos_tag(lemm_q, tagset='universal')
    
    for token in tagged_query:
        print(token)
        if token[1] == 'NOUN':
            return token[0]

def intent_matcher(query):
    for word in query:
        if word.lower() == "name":
            return "Name"
    else: 
        return "UNKNOWN"

async def weather_response(query):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    city_name = "Nottingham"

    complete_url = 

#this is the main while loop which eveything else comes from. the program will stop when the user says bye
user = 2 #user 2 is the user, and 1 is the bot
BOT = 1
USER = 2

welcome_message = 'hello, I am studybot. How can I help you?'
print(welcome_message)

query = ""
done = False
while(done == False):
    if(user == BOT):
        intent = im.find_intent("hello!")
        if intent == "greetings":
            greetings_response(query)
        elif intent == "medicine":
            medical_response(query)
        elif intent == "source":
            find_source(query)
        elif intent == "weather":
            weather_response(query)
        elif intent == "exit":
            print("I hope that I was of good use! Bye! :)")
            done == True
        # respond(query)
        user = USER
    elif(user == USER):
        query = input("What would you like to know?\n")
        #determine intent
        user = BOT
    else:
        print('Whoops! I think I am broken :(')