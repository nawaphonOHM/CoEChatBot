import pickle
import os
import numpy
import pandas
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
import src.bag_of_words as bag_of_words
import tflearn
import tensorflow
from pythainlp.spell import correct as typo_checking

def response(sentence):
    ERROR_THRESHOLD = 0.20

    work_directory = os.getcwd()
    stop_word = corpus.thai_stopwords()
    
    tensorflow.reset_default_graph()
    neural_network = tflearn.input_data(shape=[None, len(features[0])])
    neural_network = \
        tflearn.fully_connected(neural_network, 512, activation="relu")
    neural_network = \
        tflearn.fully_connected(neural_network, 512, activation="relu")
    neural_network = \
        tflearn.fully_connected(\
                neural_network, 
                len(destination_class[0]), 
                activation="softmax"
            )
    neural_network = tflearn.regression(neural_network, learning_rate=0.005)
    model = tflearn.DNN(neural_network)
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
    sentence = model.predict([sentence])[0]

    sentence = [[class_name, probability] \
        for class_name, probability in enumerate(sentence) \
            if probability > ERROR_THRESHOLD]

    intentions = bag.getEntiredItemsIntention()
    named_intention = intentions[sentence[0][0]]

    sentence.sort(key=lambda x: x[1], reverse=True)

    return named_intention, bag.getResponseSentence(named_intention)


intention = None
response_sentence = ""

while intention != "การจากลา":

    query_sentence = input("นักศึกษา: ")
    intention, response_sentence = response(query_sentence)
    print("เจ้าหน้าที่: {0}".format(response_sentence))