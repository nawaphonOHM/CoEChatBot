import json

class BagsOfWords:
    word_token = {}
    token_word = {}
    amount = 0
    next_token = 0

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

