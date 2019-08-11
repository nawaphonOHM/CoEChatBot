from src.BagsOfWords import BagsOfWords
import itertools

def encoded_string(cons, reference):
    if type(cons) is not list:
        raise TypeError(
                "Expected cons as list but got as {0}"
                .format(type(cons))
            )
    if type(reference) is not BagsOfWords:
        raise TypeError(
                "Expected reference as BagsOfWords but got as {0}"
                .format(type(reference))
            )
    
    grouped_of_vector = []

    for con in cons:
        vector = []
        for word in con:
            vector.append(reference.getToken(word))
            grouped_of_vector.append(vector)
    
            grouped_of_vector = list(
                itertools.zip_longest(*grouped_of_vector, 
                fillvalue=reference.getToken("PAD")
            )
        )

    return grouped_of_vector


def decoded_string(arrays_of_token, reference):
    if type(arrays_of_token) is not list:
        raise TypeError(
                "Expected arrays_of_token as list but got as {0}"
                .format(type(arrays_of_token))
            )
    if type(reference) is not BagsOfWords:
        raise TypeError(\
                "Expected reference as BagsOfWords but got as {0}"
                .format(type(reference))
            )
    
    new_returned_array = []

    for token in arrays_of_token:
        new_returned_array.append(reference.getWord(token))

    return new_returned_array

def binary_matrix(grouped_of_vector, reference):
    if type(grouped_of_vector) is not list:
        raise TypeError(
                    "Expected grouped_of_vector as list but got as {0}"
                    .format(type(grouped_of_vector)
                )
            )
    if type(reference) is not BagsOfWords:
        raise TypeError(
                "Expected reference as BagsOfWords but got as {0}"
                .format(type(reference))
            )

    LENGTH = len(grouped_of_vector[0])
    grouped_of_binary_array = []
    USELESS_TOKEN = reference.getToken("PAD")

    for vector in grouped_of_vector:
        if(len(vector) != LENGTH):
            raise ValueError("grouped_of_vector has an unequal vector's length")

        binary_array = []
        for dimention in vector:
            if dimention is USELESS_TOKEN:
                binary_array.append(0)
            else:
                binary_array.append(1)

        grouped_of_binary_array.append(binary_array)

    return grouped_of_binary_array

def length_maxtrix(grouped_of_vector):
    if type(grouped_of_vector) is not list:
        raise TypeError(
                    "Expected grouped_of_vector as list but got as {0}"
                    .format(type(grouped_of_vector))
                )
                
    length_maxtrix = []
    
    for vector in grouped_of_vector:
        counter = 0
        for dimension in vector:
            if dimension is not 0:
                counter = counter + 1
        length_maxtrix.append(counter)



    