import torch
import torch.nn as neural_network_tools
import json
from src.classes.BagsOfWords import BagsOfWords
from src.modeling.seq2seq import Seq2SeqEncoder as Seq2SeqEncoder
from src.modeling.seq2seq import Seq2SeqDecoder as Seq2SeqDecoder
from src.modeling.classes import GreedySearch as GreedySearch
from src.modeling.evaluation import *
import pythainlp


model = torch.load("model/CoEChatBot.tar")
setting = None
used_device = "cuda" if torch.cuda.is_available() else "cpu"

with open("./setting.json", "r") as read_file:
    setting = json.load(read_file)

bag_of_word = model["bag_of_word"]

embedding = \
    neural_network_tools.Embedding(bag_of_word.length(), setting["hidden_size"])
embedding.load_state_dict(model["embedding"])

encoder_part = Seq2SeqEncoder(\
        setting["hidden_size"], 
        embedding, 
        setting["encoder_n_layer"]
    )
encoder_part.load_state_dict(model["en"])
encoder_part.eval()
encoder_part.to(used_device)

decoder_part = Seq2SeqDecoder(\
        setting["attention_model"], 
        embedding,
        setting["hidden_size"], 
        bag_of_word.length(), 
        setting["decoder_n_layer"], 
        setting["dropout"]
    )
decoder_part.load_state_dict(model["de"])
decoder_part.eval()
decoder_part.to(used_device)

searcher = GreedySearch(\
        encoder_part, 
        decoder_part, 
        used_device, 
        bag_of_word.getToken("SOS")
    )


input_sentence = ""

while True:
    input_sentence = input("นักศึกษา: ")
    input_sentence = \
        pythainlp.tokenize.word_tokenize(\
            input_sentence, \
            engine="newmm", 
            keep_whitespace=False
        )
    input_sentence = input_sentence + ["{เริ่มต้น}"]

    output_words = \
        evaluate(\
                searcher, 
                bag_of_word, 
                input_sentence, 
                setting["max_sentence"], 
                used_device
            )
    output_words[:] = [x for x in output_words]




