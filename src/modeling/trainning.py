import random
import numpy
import pickle
import os
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
import src.bag_of_words as bag_of_words

def train():
    bag = None
    work_directory = os.getcwd()
    trainning_data = []
    testing_data = []
    inquery = None
    contextual = None
    inquery_processed_data = \
        open(os.path.join(work_directory, "data/processed/inquery_cleaned_data.csv"), "r")
    inquery = csv.reader(inquery_processed_data, delimiter=",")
    inquery.__next__()

    with open(os.path.join(work_directory, "data/processed/bag_of_word_.pkl"), "rb") \
        as bag_file:
            bag = pickle.load(bag_file)
    
    for table in inquery:
        hot_coding_pattern = []
        hot_coding_class = []
        class_name_data = table[0]
        query_data = table[1].split(" ")
                
        for word in bag.get_entried_words():
            hot_coding_pattern.append(word in query_data)
                
        for class_name in bag.get_entired_items_intention():
            hot_coding_class.append(class_name == class_name_data)
            
        trainning_data.append([hot_coding_pattern, hot_coding_class])

    
        
    random.shuffle(trainning_data)
    trainning_data = numpy.array(trainning_data)
    features = list(trainning_data[:,0])
    destination_class = list(trainning_data[:,1])
    
    model = Sequential()
    model.add(
            Dense(len(destination_class[0]), 
            input_shape=(len(features[0]),), 
            activation='relu')
        )
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(Dense(len(destination_class[0]), activation='softmax'))

    model.compile(\
        loss='categorical_crossentropy', 
        optimizer=Adam(), 
        metrics=['accuracy']
    )
    model.fit(\
        numpy.array(features), 
        numpy.array(destination_class), 
        epochs=1024, 
        batch_size=int(len(features) * 0.2), 
        verbose=1
    )
    
    model.save(os.path.join(work_directory, "model/CoeChatBot_processed_type_input.ckpt"))
    inquery_processed_data.close()

    contextual_processed_data = \
        open(os.path.join(work_directory, "data/processed/responsing_cleaned_data.csv"), "r")

    contextual = csv.reader(contextual_processed_data, delimiter=",")
    contextual.__next__()
    trainning_data = []

    for table in contextual:
        class_name = table[0]
        context = table[1]
        state = table[2]
        hot_coding_pattern = []
        hot_coding_class = []

        for state_list in bag.get_entired_state_contextual():
            hot_coding_pattern.append(state == state_list)

        for context_list in bag.get_entired_intention_contextual():
            hot_coding_pattern.append(context == context_list)

        for response_class in bag.get_entired_response_classes():
            hot_coding_class.append(class_name == response_class)
        
        trainning_data.append([hot_coding_pattern, hot_coding_class])

    random.shuffle(trainning_data)
    trainning_data = numpy.array(trainning_data)
    features = list(trainning_data[:,0])
    destination_class = list(trainning_data[:,1])

    model = Sequential()
    model.add(
            Dense(len(destination_class[0]), 
            input_shape=(len(features[0]),), 
            activation='relu')
        )
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(Dense(len(destination_class[0]), activation='softmax'))

    model.compile(\
        loss='categorical_crossentropy', 
        optimizer=Adam(), 
        metrics=['accuracy']
    )

    model.fit(\
        numpy.array(features), 
        numpy.array(destination_class), 
        epochs=1024, 
        batch_size=int(len(features) * 0.2), 
        verbose=1
    )
    
    model.save(os.path.join(work_directory, "model/CoeChatBot_processed_response_type.ckpt"))
    contextual_processed_data.close()

    with open(os.path.join(work_directory, "model/bag_of_word_.pkl"), "wb") \
        as model_write:
            pickle.dump(bag, 
                model_write, 
                protocol=pickle.HIGHEST_PROTOCOL
            )

train()