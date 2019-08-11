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

    