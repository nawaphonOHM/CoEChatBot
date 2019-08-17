import torch.nn as neural_network_tools
import torch
from torch.nn import Embedding

class Seq2SeqEncoder(neural_network_tools.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        if type(hidden_size) is not int:
            raise TypeError(\
                    "Expected hidden_size type as int but got as {0}"\
                    .format(type(hidden_size))
                )
        if hidden_size < 1:
            raise ValueError(\
                    "hidden_size must be more than 0 but got {0}"\
                    .format(hidden_size)
                )
        if type(embedding) is not Embedding:
            raise TypeError(\
                    "Expected embedding type as Embedding but got as {0}"\
                    .format(type(embedding))
                )
        if type(n_layers) is not int:
            raise TypeError(\
                    "Expected n_layers type as int but got as {0}"
                    .format(type(n_layers))
                )
        if n_layers < 1:
            raise ValueError(\
                    "n_layers must more than 0 but got {0}"
                    .format(n_layers)
                )
        if type(dropout) is not float:
            raise TypeError(\
                    "Expected dropout type as int but got as {0}"\
                    .format(type(dropout))
                )
        if dropout < 0 or dropout > 1:
            raise ValueError(\
                    "dropout must be range [0.0, 1.0] but got {0}"\
                    .format(dropout)
                )
        
        super(Seq2SeqEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        
        self.gru = neural_network_tools.GRU(\
                hidden_size, 
                hidden_size, 
                n_layers, 
                dropout=(0 if n_layers == 1 else dropout), 
                bidirectional=True
            )

    # Overwrite Method.
    def forward(\
            self, 
            input_seq, 
            input_lengths, 
            hidden=None
        ):
            if type(input_seq) is not torch.Tensor:
                raise TypeError(\
                        "Expected input_seq type as Tensor but got as {0}"\
                        .format(type(input_seq))
                    )
            if type(input_lengths) is not torch.Tensor:
                raise TypeError(\
                        "Expected input_length_vector as Tensor but got as {0}"\
                        .format(type(input_lengths))
                    )
            if type(hidden) is not type(None) and type(hidden) is not Embedding:
                    raise TypeError(\
                            "Expected hidden_tensor as Embedding but got as {0}"
                            .format(hidden)
                        )
            
            embedded_tensor = self.embedding(input_seq)
            packed_embedded_tensor = \
                neural_network_tools.utils.rnn.pack_padded_sequence(\
                    embedded_tensor, 
                    input_lengths
                )
            output_tensor, hiddden_tensor = self.gru(\
                    packed_embedded_tensor, 
                    hidden
                )

            output_tensor, _ = \
                neural_network_tools.utils.rnn.pad_packed_sequence(output_tensor)

            output_tensor = \
                output_tensor[:, :, :self.hidden_size] + \
                output_tensor[:, :, self.hidden_size:]

            return output_tensor, hiddden_tensor


            

