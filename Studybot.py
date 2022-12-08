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

# keepwords = ["how", "what", "when", "where", "why"]
# new_words = list(filter(lambda w: w in keepwords, stopwords))


#to do list:
# - pre-process text
# - cosine simimarity to find 
# - 
def generateText(query):
    return -1

def find_topic(subject):
    path = 'Datasets\\topic_list\\' + subject
    for file in os.listdir(path):
        with open(path, encoding='utf-8', errors='ignore', mode='r') as document:
            content = document.read()
            topic = document.name

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
        1: "Wow, thats a tough question. Give me some time to think... ",
        2: "Sorry, but I do not know much about ",#followed by query
        3: "I guess there are some things that I dont know. Time to hit the books!", 
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
    else:
        print(response)
        add_to_dataset(find_keywords(query)) ##returns string of information about the topic and adds page to database
        chop_response(find_keywords(query)) ##takes in entire string
        print("This is the best information that I could find from my research")
        print("I hope that this was helpful!")
        response = ""

    fbr_index += 1
    return response

def medical_response(query):
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

    resp = False
    for response in range(len(answer_list)):
        ans = input("Are you asking about "+ answer_list[response] + "? (y/n)\n")
        if im.find_intent(ans) == "yes": #if they say yes
            # print(answer_list[response])
            chop_response(answer_list[response])
            resp = True #has responded, and therefore we do not need a fallback response after exiting the loop
            return
            
    if(resp == False):
        print(fallback_response(query))

def chop_response(topic):
    st = classifier.get_summary(topic)
    resp_list = st.split("\n")

    resp_list = list(filter(None, resp_list))

    for i in range(len(resp_list)):
        banned_word = 
        resp_list = [resp_list[i].replace(banned_word, '****') for word in resp_list[i] if word in banned_list] 
        print(resp_list[i])
        print()
        if i < len(resp_list)-1:
            inp = input("Should I continue giving you information about "+ topic + "?\n").lower()
            if im.find_intent(inp) == "no":
                return


def find_name(query):
    tokenized_query = word_tokenize(query)
    
    clean_tokens = [token.capitalize() for token in tokenized_query if token not in string.punctuation]
    
    pos = nltk.pos_tag(clean_tokens)
    chunks = nltk.ne_chunk(pos)

    person = []
    for subtree in chunks.subtrees():
        if subtree.label() == "PERSON":
            for leaf in subtree.leaves():
                person.append(leaf[0])
    
    
    
    username = "".join(person)
    if len(username) == 0:
        return 'friend'
    else:
        return username

def find_source(query):
    inp = input("Are you asking for a source or are you asking a medical question?\nPress s for source and m for medical information")
    if inp.lower() == 's':
        print('finding source...')
    else:
        medical_response(query)

gr_index = 1
def greetings_response(query):
    username = find_name(query)
    greetings = {
        1: "Hi " + username +  "! I hope you are doing well!",
        2: "Hello " + username +  ". I am studybot ",#
        3: "Beep bop. Hello " + username + ". I am Studybot and I am ready to help", 
        4: "Hello " + username + ". Let me help you with my research!"
    }
    global gr_index
    gr_index = 1 if gr_index > len(greetings) else gr_index
    
    response = greetings[gr_index]

    gr_index +=1
    return response


def weather_response(query):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    city_name = "Nottingham"
    print("It is currently sunny...")
    complete_url = ''

#this is the main while loop which eveything else comes from. the program will stop when the user says bye
 #user 2 is the user, and 1 is the bot
BOT = 1
USER = 2

from pygame import mixer

query = input('hello, I am studybot. How can I help you?\n')
done = False

user = BOT
while(done == False):
    if(user == BOT):
        intent = im.find_intent(query)
        if intent == "greetings":
            print(greetings_response(query))
        elif intent == "medicine":
            medical_response(query)
        elif intent == "source":
            find_source(query)
        elif intent == "weather":
            weather_response(query)
        elif intent == "music":
            i = True
            while i:
                print('press p to pause and r to resume')
                mixer.init()
                mixer.music.load("song1.wav")
                mixer.music.set_volume(0.5)
                mixer.music.play()
                ch = input()
                if ch == 'p':
                    mixer.music.pause()
                    i = False
                elif ch == 'r':
                    mixer.music.play()
                    i = False
                print('playing music...')
        elif intent == "exit":
            print("I hope that I was of good use! Bye! :)")
            done == True
            exit()
        # respond(query)
        user = USER
    elif(user == USER):
        query = input("What else would you like to ask?\n")
        user = BOT
    else:
        print('Whoops! I think I am broken :(')