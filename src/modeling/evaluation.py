from src.processing.vector_matrix import *
import torch
from src.modeling.seq2seq.Seq2SeqEncoder import Seq2SeqEncoder as Seq2SeqEncoder
from src.modeling.seq2seq.Seq2SeqDecoder import Seq2SeqDecoder as Seq2SeqDecoder
from src.modeling.classes.GreedySearch import GreedySearch as GreedySearch
import src.classes.BagsOfWords as BagsOfWords

def evaluate(\
        searcher, 
        bag_of_words, 
        sentence, 
        max_length, 
        used_device
    ):
    if type(searcher) is not GreedySearch:
        raise TypeError(\
                "Expected searcher as GreedySearch but got as {0}"
                .format(type(searcher))
            )
    if type(bag_of_words) is not BagsOfWords.BagsOfWords:
        raise TypeError(\
                "Expected bag_of_words as BagsOfWords but got as {0}"
                .format(type(bag_of_words))
            )
    if type(sentence) is not list:
        raise TypeError(\
                "Expected sentence as list but got as {0}"
                .format(type(sentence))
            )
    if type(max_length) is not int:
        raise TypeError(\
                "Expected max_length as int but got as {0}"
                .format(type(max_length))
            )
    if max_length < 1:
        raise ValueError(\
                "a max_length must be more than 0 but got {0}"
                .format(max_length)
            )  
    if type(used_device) is not str:
        raise TypeError(\
                "Expected used device as str but got as {0}"
                .format(type(used_device))
            )
    used_devices = ["cpu", "cuda"]
    if used_device not in used_devices:
        error_message = "Accept only "
        for used_device in used_devices:
            error_message = error_message + used_device + " "
        error_message = error_message + "but got as {0}".format(used_device)
        raise ValueError(error_message)
    
    indexes_batch = [encoded_string(sentence, bag_of_words)]
    lengths_tensor = torch.Tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    input_batch = input_batch.to(used_device)
    lengths_tensor = lengths_tensor.to(used_device)

    tokens, scores = searcher(input_batch, lengths_tensor, max_length)
    
    return [bag_of_words.getWord(token.item()) for token in tokens]
