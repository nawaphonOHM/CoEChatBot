import pickle
import os
import numpy
import pandas
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
import src.bag_of_words as bag_of_words
import keras.models as keras_module_manipulation

def response(sentence):
    ERROR_THRESHOLD = 0.20

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
    sentence = [word for word in sentence if word not in stop_word]

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

    print(\
            "Type: {0}; Acuracy: {1}"\
            .format(named_intention, sentence[0][1])
        )
    
    print("เจ้าหน้าที่: {0}".format(bag.getResponseSentence(named_intention)))

while True:
    query_sentence = input("นักศึกษา: ")
    response(query_sentence)