import urllib
from urllib import request
from urllib.request import FancyURLopener
import webbrowser, sys
import bs4
from bs4 import BeautifulSoup
import requests

#source of code https://www.youtube.com/watch?v=dyUhGZ6iNTc
#https://stackoverflow.com/questions/47928608/how-to-use-beautifulsoup-to-parse-google-search-results-in-python

def search(query):
    text = urllib.parse.quote_plus(query)
    url = "https://scholar.google.com/scholar?q=" + text
   
    results = requests.get(url)
    
    soup = bs4.BeautifulSoup(results.text, "html.parser")
    #print(soup)
    for g in soup.find_all(class_ ='gs_rt'):
        print(g)
        print(g.text)
        print('-----')

search('hello world')