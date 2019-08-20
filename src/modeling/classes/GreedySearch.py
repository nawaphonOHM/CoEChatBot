import torch.nn as neural_network_tools
import torch
from src.modeling.seq2seq import Seq2SeqEncoder
from src.modeling.seq2seq import Seq2SeqDecoder

class GreedySearch(neural_network_tools.Module):
    used_device = None
    SOS_TOKEN = None

    def __init__(self, encoder_part, decoder_part, used_device, SOS_TOKEN):
        if type(encoder_part) is not Seq2SeqEncoder.Seq2SeqEncoder:
            raise TypeError(\
                    "Expected encoder_part as Seq2SeqEncoder but got as {0}"
                    .format(type(encoder_part))
                )
        if type(decoder_part) is not Seq2SeqDecoder.Seq2SeqDecoder:
            raise TypeError(\
                    "Expected decoder_part as Seq2SeqDecoder but got as {0}"
                    .format(type(decoder_part))
                )
        if type(used_device) is not str:
            raise TypeError(\
                    "Expected used_device as str but got as {0}"
                    .format(type(used_device))
                )
        used_devices = ["cuda", "cpu"]      
        if used_device not in used_devices:
            error_message = "a used_device only accepted "
            for used_device in used_devices:
                error_message = error_message + used_device + " "
            error_message = error_message + "but got {0}".format(used_device)
            raise ValueError(error_message)

        if type(SOS_TOKEN) is not int:
            raise TypeError(\
                    "Expected SOS_TOKEN as int but got as {0}"
                    .format(type(SOS_TOKEN))
                )

        super(GreedySearch, self).__init__()
        self.encoder = encoder_part
        self.decoder = decoder_part
        self.used_device = used_device
        self.SOS_TOKEN = SOS_TOKEN

    # Overwrite Method

    def forward(self, input_vector, input_length, max_length):
        if type(input_vector) is not torch.Tensor:
            raise TypeError(\
                    "Expected input_vector as Tensor but got as {0}"
                    .format(type(input_vector))
                )
        if type(input_length) is not torch.Tensor:
            raise TypeError(\
                    "Expected input_length as Tensor but got as {0}"
                    .format(type(input_length))
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

        encoder_outputs_tensor, encoder_hidden_tensor = \
            self.encoder(input_vector, input_length)
        
        decoder_hidden_tensor = encoder_hidden_tensor[:self.decoder.n_layers]
        decoder_input_tensor = \
            torch.ones(1, 1, device=self.used_device, dtype=torch.long) * \
                self.SOS_TOKEN
        all_tokens = torch.zeros([0], device=self.used_device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.used_device)

        for _ in range(max_length):
            decoder_output_tensor, decoder_hidden_tensor = \
                self.decoder(\
                        decoder_input_tensor, 
                        decoder_hidden_tensor, 
                        encoder_outputs_tensor
                    )
            decoder_scores, decoder_input_tensor = \
                torch.max(\
                        decoder_output_tensor, 
                        dim=1
                    )
            all_tokens = torch.cat(\
                    (all_tokens, decoder_input_tensor), 
                    dim=0
                )
            all_scores = torch.cat(\
                    (all_scores, decoder_scores), 
                    dim=0
                )
            decoder_input_tensor = torch.unsqueeze(decoder_input_tensor, 0)
        
        return all_tokens, all_scores