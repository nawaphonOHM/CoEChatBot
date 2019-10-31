import json

class Bag:
    word_token = {}

    def __init__(self):
        self.word_token = {}
        self.token_word = {}
        self.next_token = 0
        self.intention_classes = {}
        self.intention_classes_reverse = {}
        self.amount_intention_classes = 0
        self.response_sentence = {}
        self.amount_response_sentence = 0
        self.response_classes = []
        self.amount_response_classes = 0
        self.intention_contextual = {}
        self.intention_contextual_reverse = {}
        self.amount_intention_contextual = 0
        self.state_contextual = []
        self.amount_state_contextual = 0
        self.excluded_stop_words = None


    def get_word(self, token: int) -> str:
        if token not in self.token_word:
            raise ValueError("Unknown this token {0}.".format(token))

        return self.token_word[token]

    def set_excluded_stop_words(self, stop_words_dict: list) -> None:
        self.excluded_stop_words = stop_words_dict
    
    def has_in_excluded_stop_words(self, word: str):
        return word in self.excluded_stop_words

    def add_word(self, word: str) -> None:
        if not self.has_word(word):
            self.word_token[word] = self.next_token
            self.token_word[self.next_token] = word
            self.next_token += 1

    def has_intent_context_number(self, class_number: int) -> bool:
        return class_number in self.intention_contextual

    def has_intention(self, sentence: str) -> bool:
        return sentence in self.intention_classes_reverse

    def add_intention_contextual(self, sentence: str) -> None:
        if not self.has_intent_context_number(sentence):
            self.intention_contextual[self.amount_intention_contextual] = sentence
            self.intention_contextual_reverse[sentence] = self.amount_intention_contextual
            self.amount_intention_contextual += 1
    
    def get_intention_contextual_class_number(self, sentence: str) -> int:
        if not self.has_intent_context_number(sentence):
            raise KeyError("None of this sentence -> {0}".format(sentence))
        else:
            return self.intention_contextual_reverse[sentence]
    
    def get_state_contextual(self, sentence: str) -> int:
        if not self.has_state_context(sentence):
            raise KeyError("None of this sentence -> {0}".format(sentence))
        else:
            return self.state_contextual[sentence]
    
    def get_entired_state_contextual(self) -> list:
        return self.state_contextual
    
    def get_entired_intention_contextual_class_number(self) -> dict:
        return self.intention_contextual

    def get_entired_intention_contextual_class_name(self) -> dict:
        return self.intention_contextual_reverse

    def get_intention_contextual_length(self) -> int:
        return self.amount_intention_contextual
    
    def get_state_contextual_length(self) -> int:
        return self.amount_state_contextual
    
    def has_state_context(self, sentence: str) -> bool:
        return sentence in self.state_contextual

    def add_state_contextual(self, sentence: str) -> None:
        if not self.has_state_context(sentence):
            self.state_contextual.append(sentence)
            self.amount_state_contextual += 1
    
    def get_token(self, word: str) -> int:
        if not self.has_word(word):
            raise ValueError("Unknown this word {0}.".format(word))
        return self.word_token[word]

    def get_entired_response_classes(self) -> list:
        return self.response_classes

    def length(self) -> int:
        return self.next_token

    def add_intention(self, intention: str) -> None:
        if not self.has_intention(intention):
            self.intention_classes[self.amount_intention_classes] = intention
            self.intention_classes_reverse[intention] = self.amount_intention_classes
            self.amount_intention_classes += 1
    
    def add_response_class(self, response_class_name: str) -> None:
        self.response_classes.append(response_class_name)
        self.amount_response_classes += 1
    
    def get_response_class(self, response_id: int) -> str:
        return self.response_classes[response_id]

    def has_word(self, word: str) -> bool:
        return word in self.word_token

    def get_entried_words(self) -> dict:
        return self.word_token.keys()
    
    def amount_of_intention(self) -> int:
        return self.amount_intention_classes

    def get_entired_intention_class_name(self) -> dict:
        return self.intention_classes_reverse

    def get_entired_intention_class_number(self) -> dict:
        return self.intention_classes

    def get_intention_name(self, sentence_class: int) -> str:
        if not self.has_intent_context_number(sentence_class):
            raise KeyError("None of this sentence -> {0}".format(sentence_class))
        else:
            return self.intention_classes[sentence_class]
    
    def get_intention_class_number(self, sentence: str) -> int:
        if not self.has_intention(sentence):
            raise KeyError("None of this sentence -> {0}".format(sentence))
        else:
            return self.intention_classes_reverse[sentence]

    def get_response_sentence(self, intention: str) -> dict:
        if not self.has_response_sentence(intention):
            raise ValueError(\
                    "Unknown this class {0}"
                    .format(intention)
                )
        return self.response_sentence[intention]
    
    def has_response_sentence(self, class_name: str) -> bool:
        return class_name in self.response_sentence

    def set_response_sentence(self, class_name: str, response_sentence: dict) -> None:
        if self.has_response_sentence(class_name):
            raise ValueError("Already has this name -> {0}".format(class_name))
        self.response_sentence[class_name] = response_sentence
        self.amount_response_sentence += 1