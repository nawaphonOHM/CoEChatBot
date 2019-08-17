import torch
import torch.nn as neural_network_tools
import torch
from torch.nn import Embedding
from src.modeling.seq2seq.Attentions import Attentions
import torch.nn.functional as convolution_functions

class Seq2SeqDecoder(neural_network_tools.Module):
    def __init__(\
            self, 
            attentions_model, 
            embedding, 
            hidden_size, 
            output_size, 
            n_layers=1, 
            dropout=0.1
        ):
            if type(attentions_model) is not str:
                raise TypeError(\
                        "Expected attentions_model as str but got as {0}"
                        .format(type(attentions_model))
                    )
            if type(embedding) is not Embedding:
                raise TypeError(\
                        "Expected embedding_tensor as Embedding but got as {0}"
                        .format(type(embedding))
                    )
            if type(hidden_size) is not int:
                raise TypeError(\
                        "Expected hidden_size as int but got as {0}"
                        .format(type(hidden_size))
                    )
            if hidden_size < 1:
                raise ValueError(\
                        "a hidden_size must be more than 0 but got {0}"
                        .format(hidden_size)
                    )
            if type(output_size) is not int:
                raise TypeError(\
                        "Expected output_size_tensor as int but got as {0}"
                        .format(type(output_size))
                    )
            if output_size < 1:
                raise ValueError(\
                        "an output_size must be more than 0 but got {0}"
                        .format(output_size)
                    )
            if type(n_layers) is not int:
                raise TypeError(\
                        "Expected n_layers as int but got as {0}"
                        .format(type(n_layers))
                    )
            if n_layers < 1:
                raise ValueError(\
                        "a n_layers must be more than 0 but got {0}"
                        .format(n_layers)
                    )
            if type(dropout) is not float:
                raise TypeError(\
                        "Expected dropout as float but got as {0}"
                        .format(type(dropout))
                    )
            if dropout < 0 or dropout > 1:
                raise ValueError(\
                        "dropout value must be [0.0, 1.0] but got {0}"
                        .format(dropout)
                    )
            
            super(Seq2SeqDecoder, self).__init__()
            self.attn_model = attentions_model
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = embedding
            self.embedding_dropout = neural_network_tools.Dropout(dropout)
            self.gru = \
                neural_network_tools.GRU(\
                        hidden_size, 
                        hidden_size, 
                        n_layers, 
                        dropout=(0 if n_layers == 1 else dropout)
                    )
            self.concat = neural_network_tools.Linear(hidden_size * 2, hidden_size)
            self.out = neural_network_tools.Linear(hidden_size, output_size)
            self.attn = Attentions(attentions_model, hidden_size)

    # overwrite method.
    def forward(self, input_step, last_hidden_tensor, encoder_outputs_tensor):
        if type(input_step) is not torch.Tensor:
            raise TypeError(\
                    "Expected input_step as Tensor but got as {0}"
                    .format(type(input_step))
                )
        if type(last_hidden_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected last_hidden_tensor as Tensor but got as {0}"
                    .format(type(last_hidden_tensor))
                )
        if type(encoder_outputs_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected encoder_outputs_tensor as Tensor but got as {0}"
                    .format(type(encoder_outputs_tensor))
                )
        
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        recurrent_neural_network_output_tensor, hidden_tensor = \
            self.gru(embedded, last_hidden_tensor)

        attentions_weights = \
            self.attn(\
                    recurrent_neural_network_output_tensor, 
                    encoder_outputs_tensor
                )
        
        context_vector = \
            attentions_weights.bmm(encoder_outputs_tensor.transpose(0, 1))
        
        recurrent_neural_network_output_tensor = \
            recurrent_neural_network_output_tensor.squeeze(0)
        context_vector = context_vector.squeeze(1)
        concat_input_vector = torch.cat(\
                (recurrent_neural_network_output_tensor, context_vector), 1
            )
        concat_output_vector = torch.tanh(self.concat(concat_input_vector))

        output_vector = self.out(concat_output_vector)
        output_vector = convolution_functions.softmax(output_vector, dim=1)

        return output_vector, hidden_tensor