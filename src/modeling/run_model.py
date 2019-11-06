import pickle
import os
import numpy
import pandas
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
import bag_of_words as bag_of_words
import keras.models as keras_module_manipulation
import parser
import json
import keras
import re

def response(sentence: str, state: str) -> list:
    work_directory = os.getcwd()
    stop_word = corpus.thai_stopwords()
    inquery_model = \
        keras_module_manipulation.load_model(\
                os.path.join(work_directory, "../model/CoeChatBot_processed_type_input.ckpt")
            )
    contextual_model = \
        keras_module_manipulation.load_model(\
                os.path.join(work_directory, "../model/CoeChatBot_processed_response_type.ckpt")
            )
    bag = None
    hot_coding_word = []
    with open(os.path.join(work_directory, "../model/bag_of_word_.pkl"), "rb") \
        as model_read:
            bag = pickle.load(model_read)
    sentence = tokenization.word_tokenize(sentence, keep_whitespace=False)
    sentence = \
        [word for word in sentence \
            if word not in stop_word or bag.has_in_excluded_stop_words(word)]

    for word in bag.get_entried_words():
        hot_coding_word.append(word in sentence)

    sentence = numpy.array(hot_coding_word)
    sentence = pandas.DataFrame([sentence], dtype=float, index=["input"])
    sentence = inquery_model.predict(sentence)[0]

    sentence = [[class_name, probability] \
        for class_name, probability in enumerate(sentence)]

    sentence.sort(key=lambda x: x[1], reverse=True)
    input_type = bag.get_intention_name(sentence[0][0])
    hot_code = []

    if state == None:
        state = "null"
    
    for states in bag.get_entired_state_contextual_class_number():
        hot_code.append(state == bag.get_state_contextual_class_name(states))
    for intentions in bag.get_entired_intention_contextual_class_number():
        hot_code.append(input_type == bag.get_intention_name(intentions))

    sentence = numpy.array(hot_code)
    sentence = pandas.DataFrame([sentence], dtype=float, index=["input"])
    sentence = contextual_model.predict(sentence)[0]

    sentence = [[class_name, probability] \
        for class_name, probability in enumerate(sentence)]
    sentence.sort(key=lambda x: x[1], reverse=True)

    response = \
        bag.get_response_sentence(\
                bag.get_response_class_name(sentence[0][0])
            )

    keras.backend.clear_session()
    

    return [response, input_type]