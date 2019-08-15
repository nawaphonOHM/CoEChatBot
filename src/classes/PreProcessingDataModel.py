

class PreProcessingDataModel:
    intension = None, 
    inquery_sentence = None
    response_sentence = None
    changed_intension = None

    def setIntension(self, intension):
        if type(intension) is not str:
            raise TypeError(
                    "Expected intension as str but got as {0}"
                    .format(type(intension))
                )
        self.intension = intension

    def getIntension(self):
        return self.intension

    def setInquerySentence(self, inquery_sentence):
        if type(inquery_sentence) is not str:
            raise TypeError(
                    "Expected inquery_sentence as str but got as {0}"
                    .format(type(inquery_sentence))
                )
        self.inquery_sentence = inquery_sentence

    def getInquerySentence(self):
        return self.inquery_sentence

    def setResponseSentence(self, response_sentence):
        if type(response_sentence) is not str:
            raise TypeError(
                    "Expected response_sentence as str but got as {0}"
                    .format(type(response_sentence))
                )
        self.response_sentence = response_sentence

    def getResponseSentence(self):
        return self.response_sentence
    def setChangedIntension(self, changed_intension):
        if type(changed_intension) is not str:
            raise TypeError(
                    "Expected changed_intension as str but got as {0}"
                    .format(type(changed_intension))
                )
        self.changed_intension = changed_intension
        
    def getChangedIntension(self):
        return self.changed_intension