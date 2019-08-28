import random
import numpy
import pickle
import os
import csv
import tensorflow
import tflearn
import src.bag_of_words as bag_of_words

def train():
    bag = None
    work_directory = os.getcwd()
    trainning_data = []
    testing_data = []
    raw = None
    processed_data = \
        open(os.path.join(work_directory, "data/processed/cleaned_data.csv"), "r")
    raw = csv.reader(processed_data, delimiter=",")
    isRead = False

    with open(os.path.join(work_directory, "data/processed/bag_of_word_.pkl"), "rb") \
        as bag_file:
            bag = pickle.load(bag_file)
    
    for table in raw:
        if not isRead:
            isRead |= True
        else:
            hot_coding_pattern = []
            hot_coding_class = []
            class_name_data = table[0]
            query_data = table[1].split(" ")
                
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

    model.add(Dense(512, input_shape=(len(features[0]), ), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(len(destination_class[0]), activation='softmax'))
    
    stochastic_gradient_descent = \
        SGD(lr=0.005, decay=0.0, momentum=0.8, nesterov=True)
    model.compile(\
        loss='categorical_crossentropy', 
        optimizer=stochastic_gradient_descent, 
        metrics=['accuracy']
    )
    model.fit(\
        numpy.array(features), 
        numpy.array(destination_class), 
        epochs=1024, 
        batch_size=int(len(features[0]) * 0.2), 
        verbose=1
    )

    # results = model.evaluate(\
    #         numpy.array(features), 
    #         numpy.array(destination_class), 
    #         batch_size=int(len(features[0]) * 0.8)
    #     )
    # print("Test accuracy: {:.2f}%".format(results[1] * 100))

    with open(os.path.join(work_directory, "model/bag_of_word_.pkl"), "wb") \
        as model_write:
            pickle.dump(bag, 
                model_write, 
                protocol=pickle.HIGHEST_PROTOCOL
            )
    processed_data.close()

train()