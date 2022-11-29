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
nltk.download('universal_tagset')

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

    # if 'name' in query:
    #     username = findName(query)
    #     return "hello "+ username
    # else:
    response = process_query(query)
    print(response)
    
    

def process_query(query):
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    #tokenizing and tagging the query
   
    tokenized_query = word_tokenize(query)
    tokenized_query = [word.upper() for word in tokenized_query if not word in
    stopwords.words()]
    filtered_query = (" ").join(tokenized_query)

    #query is not tokenized, put in uppercase, stopwords removed,
    #and joined into one filtered query
    nltk.download('wordnet')
    lemmentiser = WordNetLemmatizer()
    for token in tokenized_query:
        #lemmentising all of the tokens
        # tokenized_query[token] = 
        lemmentiser.lemmatize(token)


    tagged_query = nltk.pos_tag(tokenized_query, tagset='universal')

    print(tagged_query)

    #only return false if no similar paper is found
    return False
    


filepath = "Datasets\\brain-injury.txt"
def pre_process_corpus(filepath):
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE Corpus)
    ( documentId text , documentContent text , documentTopic text ) ''')

    cursor.execute()
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().upper()
    
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tok_query = tokenizer.tokenize(query.upper())
    tok_corpus = tokenizer.tokenize(content)

    #removing stopwords
    english_stopwords = stopwords.words('english')
    filtered_query = tok_query.remove(english_stopwords)
    filtered_corpus = tok_corpus.remove(english_stopwords)
    
    # stemming
    stemmer = SnowballStemmer('english')
    stemmed_query = [stemmer.stem(word) for word in filtered_query]
    stemmed_corpus = [stemmer.stem(word) for word in filtered_corpus]
   
    ## bag of words for query

    # v_q = np.zeros(len(vocabulary))
    # for stem in stemmed_query:
    # if(stem in vocabulary):
    #     index = vocabulary.index(stem)
    #     vector_query[index] += 1
    # print(f"Query bag-of-word:\n{vector_query}")
    # print(sparse.csr_matrix(vector_query))
    # logfreq_vector_query = logfreq_weighting(vector_query)
    # print(f"Query bag-of-word (logfreq weighted):\n{logfreq_vector_query}")
    # print(sparse.csr_matrix(logfreq_vector_query))
    # return -1


fallbackResponses = {1: "sorry, could you rephrase that",
                    2: "Could you please be more clear about what you're asking. Try and include some key words",
                    3: "Could you please be more specific",
                    4: "Would you like me to make a web search about ", #followed by query
                    5: "Sorry, but I do not think that I can help you with that. Please feel free to ask me about something else!"
}
fbr_index = 1
def default_response(query):
    global fbr_index
    fbr_index = 1 if fbr_index > 5 else fbr_index
    
    response = fallbackResponses[fbr_index]
    if(fbr_index == 4):
        response += "'" + query + "'"
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
query = ""
done = False
while(done == False):
    if(user == BOT):
        print(generateText(query))
        user = USER
    elif(user == USER):
        query = input()
        split = query.split
        if("bye" in query):
            print("I hope I was of good use! Goodbye! :)")
            done = True
        user = BOT
    else:
        print('error')