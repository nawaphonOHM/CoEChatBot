import json

class BagsOfWords:
    bag = None
    bag_reverse = None
    next_token = 0

    def __init__(self):
        spacial_token_dir = json.load(
            open("./setting.json", "r", encoding="utf-8")
        )
        self.bag = {}
        self.bag_reverse = {}

        tokens = spacial_token_dir["spacial_token"]

        for word in tokens.keys():
            self.addWord(word, id=tokens[word])

        
    
    def getWord(self, token):
        if type(token) is not int:
            raise TypeError(
                    "Expected 'token' as int but got as {0}"
                    .format(type(token))
                )

        return self.bag[token]

    def addWord(self, word, id=None):
        if type(word) is not str:
            raise TypeError(\
                    "Expected 'word' as str but got as {0}"
                    .format(type(word))
                )

        if id is None:

            while self.next_token in self.bag:
                self.next_token = self.next_token + 1

            self.bag_reverse[word] = self.next_token
            self.bag[self.next_token] = word
            self.next_token = self.next_token + 1
        else:
            if type(id) is not int:
                raise TypeError(
                    "Expected 'id' as int but got as {0}"
                    .format(type(id))
                )
            if id < 0:
                raise ValueError("An ID must more than 0!")
            if id in self.bag:
                raise ValueError("There is an ID already!")
    
            self.bag_reverse[word] = id
            self.bag[id] = word
    
    def getToken(self, word):
        if type(word) is not str:
            raise TypeError(
                    "Expected 'word' as int but got as {0}"
                    .format(type(id))
                )
                
        return self.bag_reverse[word]

