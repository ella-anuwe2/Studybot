# from urllib import request
# from urllib.request import FancyURLopener
import webbrowser, sys
import bs4
import requests
#source of code https://www.youtube.com/watch?v=dyUhGZ6iNTc

def search(query):
    res = requests.get("https://google.com/search?q="+''.join(sys.argv[1:]))
    res.raise_for_status()

    soup = bs4.BeautifulSoup(res.text, "html.parser")
    linkElements = soup.select('.r a')
    linkToOpen = min(15, len(linkElements))
    for i in range(linkToOpen):
        webbrowser.open('https://google.com'+linkElements[i].get('href'))

# def download_documents(keywords, query):
#     url = "https://scholar.google.com/"
#     class MyOpener(FancyURLopener):
#         version = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.152 Safari/537.36'
#     openurl = MyOpener().open
#     raw = request.urlopen(url)
#     return -1