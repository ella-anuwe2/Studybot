import string
isUnderstood = False
username = "user"
subject = ""
topic = ""
def generateText(query):

    if 'name' in query:
        username = findName(query)
    if(isUnderstood):
        return "hello"+ username
    else:

        return defaultResponse(query)



fallbackResponses = {1: "sorry, could you rephrase that",
                    2: "What topic are you asking about",
                    3: "Could you please be more specific",
                    4: "Would you like me to make a web search about ", #followed by query
                    5: "Sorry, but I do not think that I can help you with that. Please feel free to ask me about something else!"
}
fbr_index = 1
def defaultResponse(query):
    global fbr_index
    fbr_index = 1 if fbr_index > 5 else (fbr_index += 1)
    response = fallbackResponses[fbr_index]
    # fbr_index += 1
    return response

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