##intent matcher

import classifier as c1
import classifier as c2
import json
import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

lemmentiser = WordNetLemmatizer()

intents_json = json.loads(open("intents.json").read())
yes_no = json.loads(open("yes_no.json").read())

intent_dict = {}
keywords = ""

def calc_intent():
    for intent in intents_json["intents"]:
        keywords = " ".join(intent["patterns"])
        intent_dict[intent["tag"]] = keywords
        
    c1.wikipages = intent_dict
    c1.calculate()

def find_intent(q):
    calc_intent()
    print(list(c1.most_similar(query=q))[0])
    return list(c1.most_similar(query=q))[0]

yn_dict = {}
yn_keywords = ""

def y_n():
    for intent in yes_no["intents"]:
        yn_keywords = " ".join(intent["patterns"])
        yn_dict[intent["tag"]] = yn_keywords
            
    c2.wikipages = yn_dict
    c2.calculate()

def find_yn(q):
    y_n()
    print(list(c2.most_similar(query=q))[0])
    return list(c2.most_similar(query=q))[0]