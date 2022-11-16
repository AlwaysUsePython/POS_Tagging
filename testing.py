import os
import random

def change(sentence):

    os.system("python hmm.py \"" + sentence + "\" -m ./train_model/custom_pos_model.h5")

    file = open("test.txt", 'r')

    text = file.read()

    toUse = ""

    stop = False
    currentIndex = 0
    nextWord = ""
    pos = ""
    phrase = ""
    startingTag = False

    while not stop:
        try:
            if text[currentIndex] == "/":
                startingTag = True
                found = False

            if not startingTag:
                nextWord += text[currentIndex]

            if startingTag and text[currentIndex] == " ":
                found = True
                startingTag = False
                print(pos)
                if pos[1] == "." or pos[1] == "," or pos[2] == "C" or pos[1:] == "IN":
                    stop = True
                    print("stopped")
                else:
                    if nextWord != "I" and nextWord != "am":
                        phrase += nextWord + " "
                    nextWord = ""
                    pos = ""
            
            elif startingTag:
                pos += text[currentIndex]
            
            currentIndex += 1
            
            if currentIndex == len(text):
                stop = True
                phrase += nextWord + " "
        except:
            stop = True
            if nextWord != "." and nextWord != ",":
                phrase += nextWord + " "
    
    print("Hi " + phrase[0:-1] + ", I'm dad.")

    if phrase == " ":
        return " "

    action = random.randint(1, 3)
    if action == 1:
        retStr = "Hi " + phrase[0:-1] + ", I'm dad."
    elif action == 1:
        retStr =  "Hola " + phrase[0:-1] + ", soy Dora! Can you say \"No me importa?\" That means \"I don't give a sh*t\" in Spanish!"
    elif action == 2:
        retStr = "Your mom was " + phrase[0:-1] + " last night."
    elif action == 3:
        retStr = "It's great that you're " + phrase[0:-1] + " but literally who asked"
    
    print(retStr)
    return retStr