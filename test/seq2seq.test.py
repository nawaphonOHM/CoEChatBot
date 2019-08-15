from src.processing.seq2seq.Seq2SeqDecoder import Seq2SeqDecoder
import torch.nn as neural_network_tools
from src.processing.seq2seq.Seq2SeqEncoder import Seq2SeqEncoder



hidden_size = 3
num_words = 3
encoder_n_layer = 3
decoder_n_layers = 3
dropout = 0.4
embedding = neural_network_tools.Embedding(num_words, hidden_size)
attentions_model = ["dot", "general", "concat"]

# Test an initial Seq2SeqEncoder instance.
seqEn = Seq2SeqEncoder(\
        hidden_size, 
        embedding, 
        encoder_n_layer, 
        dropout
    )

print("This is a Seq2SeqEncoder ->", end=" ")
print(seqEn)

# Test an initial Seq2SeqDecoder instance.
for mode in attentions_model:
    seqDe = Seq2SeqDecoder(\
        mode, 
        embedding, 
        hidden_size, 
        decoder_n_layers, 
        num_words, 
        dropout
    )
    print("This is a Seq2SeqDecoder " + mode + " mode " + "->", end=" ")
    print(seqDe)

