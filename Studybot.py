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


#personal information
isUnderstood = False
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


    

fallbackResponses = {1: "sorry, could you rephrase that",
                    2: "Could you please be more clear about what you're asking. Try and include some key words",
                    3: "Could you please be more specific",
                    4: "Would you like me to make a web search about ", #followed by query
                    5: "Sorry, but I do not think that I can help you with that. Please feel free to ask me about something else!"
}

fbr_index = 1
def fallback_response(query):
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

def no_inp(s):
    s.lower()
    if s == "no" or s == "n":
        return True
    else:
        return False

def respond(query):
    #match intent, and if intent is a search question (not small talk or any other intent) then conduct search
    results = classifier.most_similar(query)
    answer_list = []
    similarity_list = []
    for key in results:
        answer_list.append(key)
        similarity_list.append(results[key])
    [x for _, x in sorted(zip(similarity_list, answer_list))]
    
    resp = False
    for response in range(len(answer_list)):
        ans = input("Are you asking about "+ answer_list[response] + "? (y/n)\n")
        if no_inp(ans) == False:
            chop_response(answer_list[response])
                
            resp = True #has responded, and therefore we do not need a fallback response after exiting the loop
            print()
            cont = input("I hope this was helpful! Is there anything else that you'd like to ask?\n")
            if no_inp(cont):
                print("So glad that I could help! Bye!")
                quit()
        
    if resp == False:
        fallback_response(query)

def chop_response(topic):
    st = classifier.get_summary(topic)
    resp_list = st.split("\n")

    for i in range(len(resp_list)):
        print(resp_list[i])
        print()
        if i < len(resp_list):
            inp = input("Should I continue giving you information about "+ topic + "?\n").lower()
            if no_inp(inp):
                return -1

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

query = ""
done = False
while(done == False):
    if(user == BOT):
        respond(query)
        user = USER
    elif(user == USER):
        query = input()
        if("bye" in query):
            print("I hope I was of good use! Goodbye! :)")
            done = True
        user = BOT
    else:
        print('error')