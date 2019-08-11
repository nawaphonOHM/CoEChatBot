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
    
    def getWord(self, token):
        return self.bag[token]

    def addWord(self, word):
        self.bag_reverse[word] = self.next_token
        self.bag[self.next_token] = word
        self.next_token = self.next_token + 1
    
    def getToken(self, word):
        return self.bag_reverse[word]

