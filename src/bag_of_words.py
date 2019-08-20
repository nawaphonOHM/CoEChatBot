import json

class BagsOfWords:
    word_token = {}
    token_word = {}
    amount_word = 0
    next_token = 0
    intention_classes = []

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

    def sort_em(self):
        self.intention_classes.sort()

