import json

class BagsOfWords:
    bag = None
    bag_reverse = None
    next_token = 0
    max_token = 0

    def __init__(self):
        setting = json.load(
            open("./setting.json", "r", encoding="utf-8")
        )
        self.bag = {}
        self.bag_reverse = {}

        tokens = setting["spacial_token"]

        for word in tokens.keys():
            self.addWord(word, id=tokens[word])
        
        self.max_token = setting["max_sentence"]

        
    
    def getWord(self, token):
        if token not in self.bag:
            raise TypeError("Unknown this token.")

        return self.bag[token]

    def addWord(self, word, id=None):
        if id is None:
            while self.next_token in self.bag:
                self.next_token = self.next_token + 1

            self.bag_reverse[word] = self.next_token
            self.bag[self.next_token] = word
            self.next_token = self.next_token + 1
        else:
            if id < 0:
                raise ValueError("An ID must more than 0!")
            if id in self.bag:
                raise ValueError("There is an ID already!")
    
            self.bag_reverse[word] = id
            self.bag[id] = word
    
    def getToken(self, word):
        if word not in self.bag_reverse:
            raise TypeError("Unknown this word {0}".format(word))
        return self.bag_reverse[word]
    
    def has(self, word):
        return word in self.bag_reverse

    def length(self):
        return len(self.bag)

