import sklearn
import os
import csv
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import bag_of_words as bag_of_words
import pickle
import keras.models as keras_module_manipulation
import numpy
import pandas

def contextual_model_testing() -> None:

    model_name = "../model/CoeChatBot_processed_response_type.ckpt"
    test_data_file_name = "../data/processed/responsing_cleaned_data_test.csv"
    test_data_object = None
    work_directory = os.getcwd()
    cadidated_intention_test_data_sets = []
    cadidated_state_test_data_sets = []
    real_classes_number = []
    predicted_classes_number = []
    bag = None
    model = None

    print("being test contextual model ...")

    with open(os.path.join(work_directory, "../model/bag_of_word_.pkl"), "rb") \
        as bag_file:
            bag = pickle.load(bag_file)
    file_obj = open(\
        os.path.join(work_directory, test_data_file_name), 'r')
    file_obj.__next__()

    model = keras_module_manipulation.load_model(\
                os.path.join(work_directory, "../model/CoeChatBot_processed_response_type.ckpt")
            )
    test_data_object = csv.reader(file_obj)

    for line in test_data_object:
        real_classes_number.append(bag.get_response_class_number(line[0]))
        cadidated_intention_test_data_sets.append(\
                bag.get_intention_contextual_class_number(line[1])
            )
        cadidated_state_test_data_sets.append(\
                bag.get_state_contextual_class_number(line[2])
            )

    for i in range(len(cadidated_intention_test_data_sets)):
        state_test = cadidated_state_test_data_sets[i]
        context_test = cadidated_intention_test_data_sets[i]
        hotcoding_test_data = []

        for state_id in bag.get_entired_state_contextual_class_number():
            hotcoding_test_data.append(state_test == state_id)
        for context_id in bag.get_entired_intention_contextual_class_number():
            hotcoding_test_data.append(context_test == context_id)

        hotcoding_test_data = numpy.array(hotcoding_test_data)
        hotcoding_test_data = \
            pandas.DataFrame(\
                    [hotcoding_test_data], 
                    dtype=float, 
                    index=["input"]
                )
        predicted_classes_number.append(\
                numpy.argmax(model.predict(hotcoding_test_data)[0])
            )
    print("done")

    print("accuracy = {0}".format(\
                accuracy_score(\
                        real_classes_number, predicted_classes_number
                    )
            )
        )
    print("recall = {0}".format(\
                recall_score(\
                        real_classes_number, predicted_classes_number, average="micro"
                    )
            )
        )
    print("precision = {0}".format(\
                precision_score(\
                        real_classes_number, predicted_classes_number, average="micro"
                    )
            )
        )
    print("f1 = {0}".format(\
                f1_score(\
                        real_classes_number, predicted_classes_number, average="micro"
                    )
            )
        )

    file_obj.close()
    

def intension_model_testing() -> None:
    model_name = "../model/CoeChatBot_processed_type_input.ckpt"
    test_data_file_name = "../data/processed/inquery_cleaned_data_test.csv"
    test_data_object = None
    work_directory = os.getcwd()
    cadidated_test_data_sets = []
    real_classes_number = []
    predicted_classes_number = []
    bag = None
    model = None

    print("being test intention model ...")

    with open(os.path.join(work_directory, "../model/bag_of_word_.pkl"), "rb") \
        as bag_file:
            bag = pickle.load(bag_file)

    model = keras_module_manipulation.load_model(\
                os.path.join(work_directory, "../model/CoeChatBot_processed_type_input.ckpt")
            )

    file_obj = open(\
        os.path.join(work_directory, test_data_file_name), 'r')
    file_obj.__next__()
    
    test_data_object = csv.reader(file_obj)

    for line in test_data_object:
        real_classes_number.append(bag.get_intention_class_number(line[0]))
        cadidated_test_data_sets.append(line[1].split(" "))

    for test_data in cadidated_test_data_sets:
        hot_coding_pattern = []
        resule = None

        for word in bag.get_entried_words():
            hot_coding_pattern.append(word in test_data)
        hot_coding_pattern = numpy.array(hot_coding_pattern)
        hot_coding_pattern = \
            pandas.DataFrame(\
                    [hot_coding_pattern], 
                    dtype=float, 
                    index=["input"]
                )
        predicted_classes_number.append(\
                numpy.argmax(model.predict(hot_coding_pattern)[0])
            )

    print("done")

    print("accuracy = {0}".format(\
                accuracy_score(\
                        real_classes_number, predicted_classes_number
                    )
            )
        )
    print("recall = {0}".format(\
                recall_score(\
                        real_classes_number, predicted_classes_number, average="micro"
                    )
            )
        )
    print("precision = {0}".format(\
                precision_score(\
                        real_classes_number, predicted_classes_number, average="micro"
                    )
            )
        )
    print("f1 = {0}".format(\
                f1_score(\
                        real_classes_number, predicted_classes_number, average="micro"
                    )
            )
        )


    file_obj.close()