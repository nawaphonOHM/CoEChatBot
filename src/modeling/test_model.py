import sklearn
import os
import csv
import random
import scipy
import bag_of_words as bag_of_words
import pickle

def contextual_model_testing() -> None:
    test_data_file_name = "data/processed/inquery_cleaned_data.csv"
    test_data_object = None
    work_directory = os.getcwd()
    cadidated_test_data_sets = []
    real_classes = []
    bag = None

    print("being test contextual model ...")

    with open(os.path.join(work_directory, "data/processed/bag_of_word_.pkl"), "rb") \
        as bag_file:
            bag = pickle.load(bag_file)

    file_obj = open(\
        os.path.join(work_directory, test_data_file_name), 'r')
    file_obj.__next__()
    
    test_data_object = csv.reader(file_obj)

    
    for line in test_data_object:
        if random.random() >= 0.5:
            print(line)
            real_classes.append(bag.get_intention_class_name(line[0]))
            cadidated_test_data_sets.append(line)

    file_obj.close()

#contextual_model_testing()