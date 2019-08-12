import torch.nn as neural_network_tools
import torch.tensor as Tensor

class Seq2SeqEncoder(neural_network_tools.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        if type(hidden_size) is not Tensor:
            raise TypeError(\
                    "Expected hidden_size type as Tensor's torch but got as {0}"\
                    .format(type(hidden_size))
                )
        if type(embedding) is not Tensor:
            raise TypeError(\
                    "Expected embedding type as Tensor's torch but got as {0}"\
                    .format(type(embedding))
                )
        if type(n_layers) is not int:
            raise TypeError(\
                    "Expected n_layers type as int but got as {0}"
                    .format(type(n_layers))
                )
        if n_layers <= 0:
            raise ValueError(\
                    "n_layers must more than 0 but got {0}"
                    .format(n_layers)
                )
        if type(dropout) is not int:
            raise TypeError(\
                    "Expected dropout type as int but got as {0}"\
                    .format(type(dropout))
                )
        if dropout < 0:
            raise ValueError(\
                    "dropout must be positive number but got {0}"\
                    .format(dropout)
                )
        
        super(Seq2SeqEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        real_dropout = 0
        if n_layers == 1:
            real_dropout = 0
        else:
            real_dropout = dropout
        
        self.gru = neural_network_tools.GRU(\
                hidden_size, 
                hidden_size, 
                n_layers, 
                dropout=real_dropout, 
                bidirectional=True
            )

    # Override Method.
    def forward(\
            self, 
            input_tensor, 
            input_length_vector, 
            hidden_tensor=None
        ):
            if type(input_tensor) is not Tensor:
                raise TypeError(\
                        "Expected input_tensor type as Tensor but got as {0}"\
                        .format(type(input_tensor))
                    )
            if type(input_length_vector) is not int:
                raise TypeError(\
                        "Expected input_length_vector as int but got as {0}"\
                        .format(type(input_length_vector))
                    )
            typed_hidden_tensor = type(hidden_tensor)
            if typed_hidden_tensor is not None and \
                typed_hidden_tensor is not Tensor:
                    raise TypeError(\
                            "Expected hidden_tensor as Tensor or None but got as {0}"
                            .format(typed_hidden_tensor)
                        )
            
            embedded_tensor = self.embedding(input_tensor)
            packed_embedded_tensor = \
                neural_network_tools.utils.rnn.pack_padded_sequence(\
                    embedded_tensor, 
                    input_length_vector
                )
            output_tensor, hiddden_tensor = self.gru(\
                    packed_embedded_tensor, 
                    hidden_tensor
                )
            
            length_tensor = None
            output_tensor, length_tensor = \
                neural_network_tools.utils.rnn.pad_packed_sequence(output_tensor)

            output_tensor = output_tensor[:, :, :self.hidden_size] + output_tensor[:, :, self.hidden_size:]

            return output_tensor, hiddden_tensor


            

