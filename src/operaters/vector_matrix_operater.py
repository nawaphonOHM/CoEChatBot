from classes import BagsOfWords
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
    
    vector = []

    for con in cons:
        vector.append(reference.getToken(con))

    return vector


def decoded_string(vector, reference):
    if type(vector) is not list:
        raise TypeError(
                "Expected vector as list but got as {0}"
                .format(type(vector))
            )
    if type(reference) is not BagsOfWords:
        raise TypeError(\
                "Expected reference as BagsOfWords but got as {0}"
                .format(type(reference))
            )
    
    cons = []

    for dimension in vector:
        cons.append(reference.getWord(dimension))

    return cons

def binary_matrix(vector, reference, excluded_token):
    if type(vector) is not list:
        raise TypeError(\
                    "Expected vector as list but got as {0}"
                    .format(type(vector)
                )
            )
    if type(reference) is not BagsOfWords:
        raise TypeError(\
                "Expected reference as BagsOfWords but got as {0}"
                .format(type(reference))
            )
    if type(excluded_token) is not str:
        raise TypeError(\
                "Expected excluded_token as str but got as {0}"
                .format(type(excluded_token))
            )

    binary_vector = []

    for dimension in vector:
        if dimension == excluded_token:
            binary_vector.append(0)
        else:
            binary_vector.append(1)

    return binary_vector

def counting(vector, exclude_token_id):
    if type(vector) is not list:
        raise TypeError(
                "Expected vector as list but got as {0}"
                .format(type(vector))
            )
    if type(exclude_token_id) is not int:
        raise TypeError(\
                "Expected exclude_token_id as int but got as {0}"\
                .format(type(exclude_token_id))
            )
    count = 0
    
    for dimension in vector:
        if dimension is not exclude_token_id:
            count = count + 1
    
    return count



    