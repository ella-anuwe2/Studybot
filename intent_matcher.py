##intent matcher

import classifier
import json
import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

lemmentiser = WordNetLemmatizer()

intents_json = json.loads(open("intents.json").read())

ignore = ["?", "!", ".", ","]

intent_dict = {}
keywords = ""
for intent in intents_json["intents"]:
    keywords = " ".join(intent["patterns"])
    intent_dict[intent["tag"]] = keywords
        
classifier.wikipages = intent_dict
classifier.calculate()

def find_intent(q):
    return list(classifier.most_similar(query=q))[0]