import pickle
import os
import numpy
import pandas
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
import src.bag_of_words as bag_of_words
import keras.models as keras_module_manipulation
import flask
import flask_restful
import flask_cors
import parser
import json
import keras
import re

from pythainlp.spell import correct as typo_checking
from flask_restful import request

def response(sentence: str, state: str) -> list:

    work_directory = os.getcwd()
    stop_word = corpus.thai_stopwords()
    inquery_model = \
        keras_module_manipulation.load_model(\
                os.path.join(work_directory, "model/CoeChatBot_processed_type_input.ckpt")
            )
    contextual_model = \
        keras_module_manipulation.load_model(\
                os.path.join(work_directory, "model/CoeChatBot_processed_response_type.ckpt")
            )
    bag = None
    hot_coding_word = []
    with open(os.path.join(work_directory, "model/bag_of_word_.pkl"), "rb") \
        as model_read:
            bag = pickle.load(model_read)
    sentence = tokenization.word_tokenize(sentence, keep_whitespace=False)
    sentence = [typo_checking(word) for word in sentence if word not in stop_word]

    for word in bag.get_entried_words():
        hot_coding_word.append(word in sentence)

    sentence = numpy.array(hot_coding_word)
    sentence = pandas.DataFrame([sentence], dtype=float, index=["input"])
    sentence = inquery_model.predict(sentence)[0]

    sentence = [[class_name, probability] \
        for class_name, probability in enumerate(sentence)]

    sentence.sort(key=lambda x: x[1], reverse=True)
    input_type = bag.get_intention(sentence[0][0])
    hot_code = []

    if state == None:
        state = "null"
    
    for state_list in bag.get_entired_state_contextual():
        hot_code.append(state == state_list)
    for intention_list in bag.get_entired_intention_contextual():
        hot_code.append(input_type == intention_list)

    sentence = numpy.array(hot_code)
    sentence = pandas.DataFrame([sentence], dtype=float, index=["input"])
    sentence = contextual_model.predict(sentence)[0]

    sentence = [[class_name, probability] \
        for class_name, probability in enumerate(sentence)]
    sentence.sort(key=lambda x: x[1], reverse=True)

    response = \
        bag.get_response_sentence(\
                bag.get_response_class(sentence[0][0])
            )
    

    return [response, input_type]


class ChatWithBot(flask_restful.Resource):
    def post(self):
        data_message = json.loads(request.data)
        response_json = {}

        try:
            if not data_message["sender"] or len(data_message) == 0:
                raise AttributeError()
            response_message = response(data_message["msg"], data_message["state"])
            state = None
            
            if response_message[0]["intention_set"] == False:
                state = data_message["state"]
            else:
                state = response_message[0]["intention_set"]

            response_json["msg"] = response_message[0]["response_sentence"]
            response_json["state"] = state
            response_json["sender"] = False
            response_json["input_type"] = response_message[1]

        except Exception:
            response_json["msg"] = None
            response_json["state"] = None
            response_json["sender"] = None
        finally:
            keras.backend.clear_session()
        
        return response_json

app = flask.Flask(__name__)
api = flask_restful.Api(app)
flask_cors.CORS(app)
api.add_resource(ChatWithBot, "/chatwithbot")

if __name__ == "__main__":
    app.run(port=1996)  