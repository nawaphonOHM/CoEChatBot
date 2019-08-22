import tensorflow
import random
import numpy
import pickle
import os
import json
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import src.bag_of_words as bag_of_words

def train():
    bag = None
    work_directory = os.getcwd()
    trainning_data = []
    raw = None
    stop_words = corpus.thai_stopwords()

    with open(os.path.join(work_directory, "data/processed/bag_of_word_.pkl"), "rb") \
        as bag_file:
            bag = pickle.load(bag_file)
    with open(os.path.join(work_directory, "data/raw/raw_data.json"), "r") \
        as raw_data:
            raw = json.load(raw_data)
    
    for json_object in raw:
        hot_coding_pattern = []
        hot_coding_class = []
        query_data = json_object["inquery_sentence"]
        class_name_data = json_object["intension"]
        query_data = \
            tokenization.word_tokenize(query_data, keep_whitespace=False)
        
        for word in bag.getEntriedWords():
            hot_coding_pattern.append(word in query_data)
            
        for class_name in bag.getEntiredItemsIntention():
            hot_coding_class.append(class_name == class_name_data)
            
        trainning_data.append([hot_coding_pattern, hot_coding_class])
        
    random.shuffle(trainning_data)
    trainning_data = numpy.array(trainning_data)
    features = list(trainning_data[:, 0])
    destination_class = list(trainning_data[:, 1])
    
    model = Sequential()
    model.add(Dense(128, input_shape=(len(features[0]), ), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(destination_class[0]), activation='softmax'))
    
    stochastic_gradient_descent = \
        SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(\
        loss='categorical_crossentropy', 
        optimizer=stochastic_gradient_descent, 
        metrics=['accuracy']
    )
    model.fit(\
        numpy.array(features), 
        numpy.array(destination_class), 
        epochs=20, 
        batch_size=5, 
        verbose=1
    )

    with open(os.path.join(work_directory, "model/bag_of_word_.pkl"), "wb") \
        as model_write:
            pickle.dump(bag, 
                model_write, 
                protocol=pickle.HIGHEST_PROTOCOL
            )
    
    model.save(os.path.join(work_directory, "model/CoeChatBot.ckpt"))