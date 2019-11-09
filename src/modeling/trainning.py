import random
import numpy
import pickle
import os
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
import bag_of_words as bag_of_words
import math

def train(train_split_ratio=0.8) -> None:
    if train_split_ratio > 1:
        raise ValueError(\
                "train_split_ratio should be [0, 1], but got {0}"\
                    .format(train_split_ratio)
            )
    bag = None
    work_directory = os.getcwd()
    trainning_data = []
    testing_data = []
    inquery = None
    contextual = None
    inquery_processed_data = \
        open(os.path.join(work_directory, "../data/processed/inquery_cleaned_data.csv"), "r")
    inquery = csv.reader(inquery_processed_data, delimiter=",")
    test_data = \
        open(os.path.join(work_directory, "../data/processed/inquery_cleaned_data_test.csv"), "w")
    writer = csv.writer(test_data)
    writer.writerow(["Intention", "Query_Sentence_Pattern_Cleaned", "Query_Sentence_Pattern"])
    inquery.__next__()

    with open(os.path.join(work_directory, "../model/bag_of_word_.pkl"), "rb") \
        as bag_file:
            bag = pickle.load(bag_file)
    
    for table in inquery:
        if random.random() > 1 - train_split_ratio:
            hot_coding_pattern = []
            hot_coding_class = []
            class_name_data = table[0]
            query_data = table[1].split(" ")
            
            for word in bag.get_entried_tokens():
                hot_coding_pattern.append(bag.get_word(word) in query_data)
            
            for class_id in bag.get_entired_intention_class_number():
                hot_coding_class.append(class_name_data == bag.get_intention_name(class_id))
                
            trainning_data.append([hot_coding_pattern, hot_coding_class])
        else:
            writer.writerow([table[0], table[1], table[2]])
    
    test_data.close()
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
    model.add(
        Dropout(0.1)
    )
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(
        Dropout(0.1)
    )
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
        batch_size=(2 ** (int(math.log2(len(features)))))
    )
    
    model.save(os.path.join(work_directory, "../model/CoeChatBot_processed_type_input.ckpt"))
    inquery_processed_data.close()

    contextual_processed_data = \
        open(os.path.join(work_directory, "../data/processed/responsing_cleaned_data.csv"), "r")

    test_data = \
        open(os.path.join(work_directory, "../data/processed/responsing_cleaned_data_test.csv"), "w")
    writer = csv.writer(test_data)
    writer.writerow(["response_classes", "intention", "state"])
    contextual = csv.reader(contextual_processed_data, delimiter=",")
    contextual.__next__()
    trainning_data = []

    for table in contextual:
        if random.random() > 1 - train_split_ratio:
            class_name = table[0]
            context = table[1]
            state = table[2]
            hot_coding_pattern = []
            hot_coding_class = []
            
            for state_id in bag.get_entired_state_contextual_class_number():
                hot_coding_pattern.append(state == bag.get_state_contextual_class_name(state_id))
                
            for context_id in bag.get_entired_intention_contextual_class_number():
                hot_coding_pattern.append(context == bag.get_intention_name(context_id))
                
            for response_class in bag.get_entired_response_classes_class_number():
                hot_coding_class.append(class_name == bag.get_response_class_name(response_class))
            
            trainning_data.append([hot_coding_pattern, hot_coding_class])
        else:
            writer.writerow([table[0], table[1], table[2]])

    test_data.close()
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
    model.add(
        Dropout(0.04)
    )
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(
        Dropout(0.04)
    )
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(
        Dropout(0.04)
    )
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(
        Dropout(0.04)
    )
    model.add(Dense(len(destination_class[0]), activation='relu'))
    model.add(
        Dropout(0.04)
    )
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
        batch_size=(2 ** (int(math.log2(len(features)))))
    )
    
    model.save(os.path.join(work_directory, "../model/CoeChatBot_processed_response_type.ckpt"))
    contextual_processed_data.close()

    with open(os.path.join(work_directory, "../model/bag_of_word_.pkl"), "wb") \
        as model_write:
            pickle.dump(bag, 
                model_write, 
                protocol=pickle.HIGHEST_PROTOCOL
            )