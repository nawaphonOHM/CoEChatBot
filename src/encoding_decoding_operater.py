from src.BagsOfWords import BagsOfWords

def encoded_string(arrays_of_char, reference):
    if type(arrays_of_char) is not list:
        raise TypeError(
                "Expected arrays_of_char as list but got as {0}"
                .format(type(arrays_of_char))
            )
    if type(reference) is not BagsOfWords:
        raise TypeError(
                "Expected reference as BagsOfWords but got as {0}"
                .format(type(reference))
            )
    
    new_returned_array = []

    for char in arrays_of_char:
        new_returned_array.append(reference.getToken(char))

    return new_returned_array


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

    