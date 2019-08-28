import json

class Bag:
    word_token = {}

    def __init__(self):
        self.word_token = {}
        self.token_word = {}
        self.amount_word = 0
        self.next_token = 0
        self.intention_classes = []
        self.amount_class = 0
        self.intention_response = {}
        self.intention_map = {}


    def getWord(self, token):
        if type(token) is not int:
            raise TypeError(\
                    "Expected token as int but got as {0}"
                    .format(type(token))
                )
        if token not in self.token_word:
            raise ValueError("Unknown this token {0}.".format(token))

        return self.token_word[token]

    def addWord(self, word):
        if type(word) is not str:
            raise TypeError(\
                    "Expected word as str but got as {0}"
                    .format(type(word))
                )
        if word not in self.word_token:
            self.word_token[word] = self.next_token
            self.token_word[self.next_token] = word
            self.next_token += 1
            self.amount_word += 1
    
    def getToken(self, word):
        if type(word) is not str:
            raise TypeError(\
                    "Expected word as str but got as {0}"
                    .format(type(word))
                )
        if word not in self.word_token:
            raise ValueError("Unknown this word {0}.".format(word))
        return self.word_token[word]

    def length(self):
        return self.amount_word

    def addIntentionType(self, intention):
        if type(intention) is not str:
            raise TypeError(\
                    "Expected intention as str but got as {0}"
                    .format(type(intention))
                )
        if intention not in self.intention_classes:
            self.intention_classes.append(intention)
            self.amount_class += 1

    def sort_em(self):
        self.intention_classes.sort()

    def has(self, word):
        return word in self.word_token

    def getEntriedWords(self):
        return self.word_token.keys()
    
    def amountOfIntention(self):
        return self.amount_class

    def getEntiredItemsIntention(self):
        return self.intention_classes.copy()

    def getIntention(self, id):
        if type(id) is not int:
            raise TypeError(
                    "Expected id as int but got as {0}"
                    .format(type(id))
                )
        return self.intention_classes[id]

    def getResponseSentence(self, intention):
        if type(intention) is int:
            return self.intention_response[self.intention_classes[intention]]
        if type(intention) is not str:
            raise TypeError(\
                    "Expected an intention as str or int but got as {0}"
                    .format(type(intention))
                )
        if intention not in self.intention_classes:
            raise ValueError(\
                    "Unknown this class {0}"
                    .format(intention)
                )
        return self.intention_response[intention]

    def setResponseSentence(self, intention, response_sentence):
        if type(intention) is not str:
            raise TypeError(\
                    "Expected an intention as str but got as {0}"
                    .format(type(intention))
                )
        if type(response_sentence) is not str:
            raise TypeError(\
                    "Expected a response sentence as str but got as {0}"
                    .format(type(response_sentence))
                )
        self.intention_response[intention] = response_sentence
    
    def setIntentionMap(self, class_name, new_class_name):
        if type(class_name) is not str:
            raise TypeError(\
                "Expected a class_name as str but got as {0}"
                .format(type(class_name))
            )
        if type(new_class_name) is not str:
            raise TypeError(\
                "Expected an new_class_name as str but got as {0}"
                .format(type(new_class_name))
            )
        self.intention_map[class_name] = new_class_name
    
    def getIntentionMap(self, class_name):
        if type(class_name) is not str:
            raise TypeError(\
                "Expected a class_name as str but got as {0}"
                .format(type(class_name))
            )
        if class_name not in self.intention_map.keys():
            raise TypeError(\
                    "This class_name has no intention set -> {0}"
                    .format(class_name)
                )
        return self.intention_map[class_name]
    
    def classNameHasMapAnother(self, class_name):
        if type(class_name) is not str:
            raise TypeError(\
                    "Expected a class_name as str but got as {0}"
                    .format(type(class_name))
                )
        return class_name in self.intention_map.keys()