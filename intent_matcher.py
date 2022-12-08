##intent matcher

import classify_with_sw as c1
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

for intent in intents_json["intents"]:
    keywords = " ".join(intent["patterns"])
    intent_dict[intent["tag"]] = keywords
        
    c1.wikipages = intent_dict
    c1.calculate()

def find_intent(q):

    results = c1.most_similar(q)
    answer_list = [] #l1
    similarity_list = [] #l2
    for key in results:
        answer_list.append(key)
        similarity_list.append(results[key]) 
    
    zipped_pairs = zip(similarity_list, answer_list)
    answer_list = [x for _, x in sorted(zipped_pairs, reverse=True)]
    sorted(similarity_list, reverse=True)

    # print(answer_list[0] + ": " + str(similarity_list[0]))
    # print(list(c1.most_similar(query=q)))
    return answer_list[0]
