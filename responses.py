import testing

def handle_response(message):

    if message.startswith("I'm ") or message.startswith("i'm "):
        newMessage = ""
        for letter in message:
            if letter == "." or letter == ",":
                newMessage += " " + letter
            else:
                newMessage += letter

        message = testing.change(newMessage[4:])

        print(message)
        return message

    elif message.startswith("I am ") or message.startswith("i am"):
        newMessage = ""
        for letter in message:
            if letter == "." or letter == ",":
                newMessage += " " + letter
            else:
                newMessage += letter

        message = testing.change(newMessage[5:])
        print(message)
        return message
    elif message.startswith("Im ") or message.startswith("im"):
        newMessage = ""
        for letter in message:
            if letter == "." or letter == ",":
                newMessage += " " + letter
            else:
                newMessage += letter

        message = testing.change(newMessage[3:])
        print(message)
        return message