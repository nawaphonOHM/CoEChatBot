from src.preparation.get_trainning_data import get_trainning_data as trainning_data
from src.classes.BagsOfWords import BagsOfWords
import random
from src.processing.vector_matrix import *
from itertools import zip_longest
import torch
import json
from src.modeling.train import train
from src.modeling.seq2seq.Seq2SeqEncoder import Seq2SeqEncoder
from src.modeling.seq2seq.Seq2SeqDecoder import Seq2SeqDecoder
import torch.nn as neural_network_tools
from torch import optim as optimization
import os

entired_trained_data = trainning_data()
bag = BagsOfWords()
batches_input = []
batches_output = []
setting = None
encoder_saved = None
decoder_saved = None
embedding_saved = None
encoder_optimizer_saved = None
decoder_optimizer_saved = None

with open("./setting.json", "r") as read_file:
    setting = json.load(read_file)
used_device = "cuda" if torch.cuda.is_available() else "cpu"

# Building bag of words.
for trained_data in entired_trained_data:
    for word in trained_data[0]:
        if not bag.has(word):
            bag.addWord(word)
    for word in trained_data[1]:
        if not bag.has(word):
            bag.addWord(word)
# -----------------------------------------------

# Building Training batches
for _ in range(setting["iteration"]):
    batch_input = []
    batch_output = []
    mask_vectors = []
    length_vector = []
    batches = \
            [random.choice(entired_trained_data) \
                for _ in range(setting["batch_size"])]
    batches.sort(key=lambda x: len(x[0]), reverse=True)

    for batch in batches:
        input_vector, output_vector = batch

        input_vector = encoded_string(input_vector, bag)
        output_vector = encoded_string(output_vector, bag)

        batch_input.append(input_vector)
        batch_output.append(output_vector)
    
    length_vector = [\
        counting(input_batch, bag.getToken("PAD")) for input_batch in batch_input]
    length_vector = torch.Tensor(length_vector)
    padded_input = \
            list(zip_longest(*batch_input, fillvalue=bag.getToken("PAD")))
    padded_input = torch.LongTensor(padded_input)
    batches_input.append([padded_input, length_vector])

    padded_output = \
            list(zip_longest(*batch_output, fillvalue=bag.getToken("PAD")))
    max_output_length = [\
        counting(output_batch, bag.getToken("PAD")) for output_batch in batch_output]
    max_output_length = max(max_output_length)

    for padded_vector in padded_output:
        mask_vector = binary_matrix(padded_vector, bag, bag.getToken("PAD"))
        mask_vectors.append(mask_vector)
    mask_vectors = torch.BoolTensor(mask_vectors)

    padded_output = torch.LongTensor(padded_output)
    batches_output.append([padded_output, mask_vectors, max_output_length])
# -----------------------------------------------------------------------------------

with open("./test/preprocess_encoder_part.txt", "w") as write_file:
    write_file.write("encoder_input_vector: " + str(batches_input[0][0]))
    write_file.write("length_input_vector: " + str(batches_input[0][1]))


# Building Embedding, Seq2SeqEncoder, Seq2SeqDecoer
embedding_tensor = neural_network_tools.Embedding(\
        bag.length(), 
        setting["hidden_size"]
    )

encoder_part = Seq2SeqEncoder(\
        setting["hidden_size"], 
        embedding_tensor, 
        setting["encoder_n_layer"], 
        setting["dropout"]
    )
encoder_part.to(used_device)
encoder_part.train()

decoder_part = Seq2SeqDecoder(\
        setting["attention_model"], 
        embedding_tensor, 
        setting["hidden_size"], 
        bag.length(), 
        setting["decoder_n_layer"], 
        setting["dropout"]
    )
decoder_part.to(used_device)
decoder_part.train()
# --------------------------------------------------------------------


# Building Encoder Optimizer, Decoder Optimizer
encoder_optimizer = optimization.Adam(\
        encoder_part.parameters(), 
        lr=setting["learning_rate"]
    )

decoder_optimizer = optimization.Adam(\
        decoder_part.parameters(), 
        lr=setting["learning_rate"] * setting["decoder_learning_ratio"]
    )
# ---------------------------------------------------------------------------

# Setting up for using cuda as a processor
for state in encoder_optimizer.state.values():
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            state[key] = value.cuda()

for state in decoder_optimizer.state.values():
    for key, value in state.items():
        if isinstance(value, torch.Tensor()):
            state[key] = value.cuda()
# ------------------------------------------------------------------------------

# Training Processes
for rounded_iteration in range(setting["iteration"]):
    input_tensor, length_tensor = batches_input[rounded_iteration]
    target_tensor, mask_tensor, max_output_length = batches_output[rounded_iteration]

    loss = train(\
            input_tensor, 
            length_tensor, 
            target_tensor, 
            mask_tensor, 
            max_output_length, 
            encoder_part, 
            decoder_part, 
            embedding_tensor,  
            encoder_optimizer, 
            decoder_optimizer, 
            setting["batch_size"], 
            setting["clip"], 
            bag.getToken("SOS"), 
            setting["teacher_forcing_ratio"], 
            used_device
        )

    print(\
            "\rIteration: {}; Percent complete: {:.2f}%; loss: {}"
            .format(\
                    rounded_iteration, 
                    (rounded_iteration / setting["iteration"]) * 100, 
                    loss
                ), end=""
        )
    
    encoder_saved = encoder_part.state_dict()
    decoder_saved = decoder_part.state_dict()
    embedding_saved = embedding_tensor.state_dict()
    encoder_optimizer_saved = encoder_optimizer.state_dict()
    decoder_optimizer_saved = decoder_optimizer.state_dict()

torch.save(\
        {
            "iteration": setting["iteration"], 
            "en": encoder_saved, 
            "de": decoder_saved, 
            "en_opt": encoder_optimizer_saved, 
            "de_opt": decoder_optimizer_saved, 
            "embedding": embedding_saved, 
            "bag_of_word": bag
        }, os.path.join("model/CoEChatBot.tar")
    )

