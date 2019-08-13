from seq2seq.Seq2SeqEncoder import Seq2SeqEncoder
import torch.nn as neural_network_tools



hidden_size = 3
num_words = 3
encoder_n_layer = 3
dropout = 0.4
embedding = neural_network_tools.Embedding(num_words, hidden_size)

# Test an initial Seq2SeqEncoder instance.
seq = Seq2SeqEncoder(\
        hidden_size, 
        embedding, 
        encoder_n_layer, 
        dropout
    )
seq.to("cpu")

print("This is a Seq2SeqEncoder ->", end=" ")
print(seq)

print("This is the seq2seqEncoder.train() -> ", end="")

seq.train()

print(seq)
