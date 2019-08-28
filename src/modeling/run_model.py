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

def response(sentence, state):
    CONFIDENT = 0.80

    work_directory = os.getcwd()
    stop_word = corpus.thai_stopwords()
    model = \
        keras_module_manipulation.load_model(os.path.join(work_directory, "model/CoeChatBot.ckpt"))
    bag = None
    hot_coding_word = []
    with open(os.path.join(work_directory, "model/bag_of_word_.pkl"), "rb") \
        as model_read:
            bag = pickle.load(model_read)
    sentence = tokenization.word_tokenize(sentence, keep_whitespace=False)
    sentence = [typo_checking(word) for word in sentence if word not in stop_word]

    for word in bag.getEntriedWords():
        hot_coding_word.append(word in sentence)

    sentence = numpy.array(hot_coding_word)
    sentence = pandas.DataFrame([sentence], dtype=float, index=["input"])
    sentence = model.predict(sentence)[0]

    sentence = [[class_name, probability] \
        for class_name, probability in enumerate(sentence) \
            if probability >= CONFIDENT]
    
    if len(sentence) > 0:
        response_data = []
        sentence.sort(key=lambda x: x[1], reverse=True)
        class_name = bag.getIntention(sentence[0][0])

        if state != None:
            class_name = class_name + " {{" + state + "}}"

        if bag.classNameHasIntentionSet(class_name):
            response_data.append(True)
            response_data.append(bag.getIntentionSet(class_name))
        elif bag.classNameHasMapAnother(class_name):
            response_data.append(False)
            response_data.append(bag.getIntentionMap(class_name))
        else:
            response_data.append(None)
            response_data.append(None)
        
        while re.match(".*\{\{.*\}\}", class_name):
            class_name = bag.getIntentionMap(class_name)

        response_data.append(bag.getResponseSentence(class_name))

        return response_data
    else:
        return None


class ChatWithBot(flask_restful.Resource):
    def post(self):
        data_message = json.loads(request.data)
        response_json = {}

        try:
            if not data_message["sender"] or len(data_message) == 0:
                raise ValueError()
            response_message = response(data_message["msg"], data_message["state"])
            
            if response_message == None:
                response_json["msg"] = "ขอโทษครับ ผมไม่เข้าใจที่พิมพ์มาครับ"
                response_json["state"] = None
            elif response_message[0] == True:
                response_json["state"] = response_message[1]
                response_json["msg"] = response_message[2]
            elif response_message[0] == False:
                response_json["state"] = None
                response_json["meg"] = response_message[2]
            response_json["sender"] = False
        except ValueError:
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