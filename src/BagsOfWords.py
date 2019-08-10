import json

class BagsOfWords:
    bag = None
    bag_reverse = None
    next_word = 0
    spacial_token_dir = None

    def __init__(self):
        self.bag = {}
        self.bag_reverse = {}
        self.spacial_token_dir = json.load(
            open("./setting.json", "r", encoding="utf-8")
        )
    
    def getWord(self, token):
        return self.bag[token]

    def addWord(self, word):
