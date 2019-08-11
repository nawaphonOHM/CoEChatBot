from src.encoding_decoding_operater import *
from src.BagsOfWords import BagsOfWords

bag = BagsOfWords()

b = encoded_string(["EOS"], bag)


print(b)
# print(decoded_string(b, bag))
# print(decoded_string(b, "bag"))

