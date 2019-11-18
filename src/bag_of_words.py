import json

class Bag:
    word_token = {}

    def __init__(self):
        self.word_token = {}
        self.token_word = {}
        self.token_word_keys = []
        self.next_token = 0
        self.intention_classes = {}
        self.intention_classes_reverse = {}
        self.intention_classes_keys = []
        self.amount_intention_classes = 0
        self.response_sentence = {}
        self.amount_response_sentence = 0
        self.response_classes = {}
        self.response_classes_reverse = {}
        self.response_classes_keys = []
        self.amount_response_classes = 0
        self.intention_contextual = {}
        self.intention_contextual_reverse = {}
        self.intention_contextual_keys = []
        self.amount_intention_contextual = 0
        self.state_contextual = {}
        self.state_contextual_reverse = {}
        self.state_contextual_keys = []
        self.amount_state_contextual = 0
        self.excluded_stop_words = None


    def get_word(self, token: int) -> str:
        if token not in self.token_word:
            raise KeyError("Unknown this token {0}.".format(token))

        return self.token_word[token]

    def set_excluded_stop_words(self, stop_words_dict: list) -> None:
        self.excluded_stop_words = stop_words_dict
    
    def has_in_excluded_stop_words(self, word: str) -> bool:
        return word in self.excluded_stop_words

    def add_word(self, word: str) -> None:
        if not self.has_word(word):
            self.word_token[word] = self.next_token
            self.token_word[self.next_token] = word
            self.token_word_keys.append(self.next_token)
            self.token_word_keys.sort()
            self.next_token += 1
    
    def has_intent_context_name(self, class_number: str) -> bool:
        return class_number in self.intention_contextual_reverse

    def has_intention_name(self, intention_name: str) -> bool:
        return intention_name in self.intention_classes_reverse

    def add_intention_contextual(self, sentence: str) -> None:
        if not self.has_intent_context_name(sentence):
            self.intention_contextual[self.amount_intention_contextual] = sentence
            self.intention_contextual_reverse[sentence] = self.amount_intention_contextual
            self.intention_contextual_keys.append(self.amount_intention_contextual)
            self.intention_contextual_keys.sort()
            self.amount_intention_contextual += 1
    
    def get_intention_contextual_class_number(self, intent_context_name: str) -> int:
        if not self.has_intent_context_name(intent_context_name):
            raise KeyError("Not found in this id -> {0}".format(intent_context_name))
        else:
            return self.intention_contextual_reverse[intent_context_name]

    def get_intention_contextual_class_name(self, intent_context_id: int) -> int:
        if not self.has_intent_context_name(intent_context_id):
            raise KeyError("Unknown in this name -> {0}".format(intent_context_id))
        else:
            return self.intention_contextual[intent_context_id]
    
    def get_state_contextual_class_number(self, state_name: str) -> int:
        if not self.has_state_context(state_name):
            raise KeyError("Unknown in this name -> {0}".format(state_name))
        else:
            return self.state_contextual_reverse[state_name]
    
    def get_state_contextual_class_name(self, state_id: int) -> str:
        return self.state_contextual[state_id]
    
    def get_entired_state_contextual_class_number(self) -> list:
        return self.state_contextual_keys
    
    def get_entired_intention_contextual_class_number(self) -> list:
        return self.intention_contextual_keys

    def get_entired_intention_contextual_class_name(self) -> dict:
        return self.intention_contextual_reverse.keys()

    def get_intention_contextual_length(self) -> int:
        return self.amount_intention_contextual
    
    def get_state_contextual_length(self) -> int:
        return self.amount_state_contextual
    
    def has_state_context(self, state_name: str) -> bool:
        return state_name in self.state_contextual_reverse

    def add_state_contextual(self, sentence: str) -> None:
        if not self.has_state_context(sentence):
            self.state_contextual_reverse[sentence] = self.amount_state_contextual
            self.state_contextual[self.amount_state_contextual] = sentence
            self.state_contextual_keys.append(self.amount_state_contextual)
            self.state_contextual_keys.sort()
            self.amount_state_contextual += 1
    
    def get_token(self, word: str) -> int:
        if not self.has_word(word):
            raise KeyError("Unknown this word {0}.".format(word))
        return self.word_token[word]

    def get_entired_response_classes_class_name(self) -> dict:
        return self.response_classes_reverse.keys()
    
    def get_entired_response_classes_class_number(self) -> list:
        return self.response_classes_keys

    def length(self) -> int:
        return self.next_token

    def add_intention(self, intention_name: str) -> None:
        if not self.has_intention_name(intention_name):
            self.intention_classes[self.amount_intention_classes] = intention_name
            self.intention_classes_reverse[intention_name] = self.amount_intention_classes
            self.intention_classes_keys.append(self.amount_intention_classes)
            self.intention_classes_keys.sort()
            self.amount_intention_classes += 1
    
    def add_response_class(self, response_class_name: str) -> None:
        if not self.has_response_class(response_class_name):
            self.response_classes[self.amount_response_classes] = response_class_name
            self.response_classes_reverse[response_class_name] = self.amount_response_classes
            self.response_classes_keys.append(self.amount_response_classes)
            self.response_classes_keys.sort()
            self.amount_response_classes += 1
    
    def get_response_class_name(self, response_id: int) -> str:
        if response_id not in self.response_classes:
            raise KeyError("Not found this id {0}.".format(response_id))
        return self.response_classes[response_id]
    
    def get_response_class_number(self, response_name: str) -> int:
        if not self.has_response_class(response_name):
           raise KeyError("Unknown this name -> {0}.".format(response_name)) 
        return self.response_classes_reverse[response_name]

    def has_response_class(self, response_name: str) -> bool:
        return response_name in self.response_classes_reverse

    def has_word(self, word: str) -> bool:
        return word in self.word_token

    def get_entried_words(self) -> dict:
        return self.word_token.keys()

    def get_entried_tokens(self) -> list:
        return self.token_word_keys
    
    def amount_of_intention(self) -> int:
        return self.amount_intention_classes

    def get_entired_intention_class_name(self) -> dict:
        return self.intention_classes_reverse.keys()

    def get_entired_intention_class_number(self) -> list:
        return self.intention_classes_keys

    def get_intention_name(self, intention_id: int) -> str:
        if intention_id not in self.intention_classes:
            raise KeyError("Not found in this id -> {0}".format(intention_id))
        else:
            return self.intention_classes[intention_id]
    
    def get_intention_class_number(self, sentence_name: str) -> int:
        if not self.has_intention_name(sentence_name):
            raise KeyError("Unknown in this name -> {0}".format(sentence_name))
        else:
            return self.intention_classes_reverse[sentence_name]

    def get_response_sentence(self, intention: str) -> dict:
        if not self.has_response_sentence(intention):
            raise KeyError(\
                    "Unknown this class {0}"
                    .format(intention)
                )
        return self.response_sentence[intention]
    
    def has_response_sentence(self, class_name: str) -> bool:
        return class_name in self.response_sentence

    def set_response_sentence(self, class_name: str, response_sentence: dict) -> None:
        if self.has_response_sentence(class_name):
            raise KeyError("Already has this name -> {0}".format(class_name))
        self.response_sentence[class_name] = response_sentence
        self.amount_response_sentence += 1