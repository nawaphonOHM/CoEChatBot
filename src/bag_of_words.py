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
        self.intention_set = {}


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
        if not self.has(word):
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
        if not self.has(word):
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
        if not self.hasClass(intention):
            self.intention_classes.append(intention)
            self.amount_class += 1

    def sort_em(self):
        self.intention_classes.sort()

    def has(self, word):
        return word in self.word_token

    def hasClass(self, class_name):
        return class_name in self.intention_classes

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
        if not self.hasClass(intention):
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
    
    def setIntentionMap(self, class_name_pattern, map_class_name):
        if type(class_name_pattern) is not str:
            raise TypeError(\
                "Expected a class_name_pattern as str but got as {0}"
                .format(type(class_name_pattern))
            )
        if type(map_class_name) is not str:
            raise TypeError(\
                "Expected an map_class_name as str but got as {0}"
                .format(type(map_class_name))
            )
        self.intention_map[class_name_pattern] = map_class_name
    
    def getIntentionMap(self, class_name_pattern):
        if type(class_name_pattern) is not str:
            raise TypeError(\
                "Expected a class_name_pattern as str but got as {0}"
                .format(type(class_name_pattern))
            )
        if not self.classNameHasMapAnother(class_name_pattern):
            raise TypeError(\
                    "This class_name_pattern has no intention map -> {0}"
                    .format(class_name_pattern)
                )
        return self.intention_map[class_name_pattern]
    
    def classNameHasMapAnother(self, class_name_pattern):
        if type(class_name_pattern) is not str:
            raise TypeError(\
                    "Expected a class_name_pattern as str but got as {0}"
                    .format(type(class_name_pattern))
                )
        return class_name_pattern in self.intention_map.keys()
    
    def classNameHasIntentionSet(self, class_name):
        if type(class_name) is not str:
            raise TypeError(\
                    "Expected a class_name as str but got as {0}"
                    .format(type(class_name))
                )
        return class_name in self.intention_set.keys()

    def getIntentionSet(self, class_name):
        if type(class_name) is not str:
            raise TypeError(\
                    "Expected a class_name as str but got as {0}"
                    .format(type(class_name))
                )
        if not self.classNameHasIntentionSet(class_name):
            raise TypeError(\
                    "This class_name has no intention set -> {0}"
                    .format(class_name)
                )
        return self.intention_set[class_name]

    def setIntentionSet(self, class_name, set_class_name):
        if type(class_name) is not str:
            raise TypeError(\
                    "Expected a class_name as str but got as {0}"
                    .format(type(class_name))
                )
        self.intention_set[class_name] = set_class_name