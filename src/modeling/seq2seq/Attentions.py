import torch
import torch.nn as neural_network_tools
import torch.nn.functional as convolution_functions
import torch

class Attentions(neural_network_tools.Module):
    def __init__(self, method, hidden_size):
        if type(method) is not str:
            raise TypeError(\
                    "Expected method as str but got as {0}"\
                        .format(type(method))
                )
        
        allowness_methods = ["dot", "general", "concat"]
        if method not in allowness_methods:
            error_message = "Expected allowed method only "
            for allowness_method in allowness_methods:
                error_message = error_message + allowness_method + " "
            
            error_message + "but got method {0}".format(method)

            raise ValueError(error_message)
        if type(hidden_size) is not int:
            raise TypeError(\
                    "Expected hidden_size as int but got as {0}"
                    .format(type(hidden_size))
                )
        if hidden_size < 1:
            raise ValueError(\
                    "hidden_size must more than 0 but got {0}"
                    .format(hidden_size)
                )
        
        super(Attentions, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = \
                neural_network_tools.Linear(\
                    self.hidden_size, 
                    hidden_size
                )
        elif self.method == "concat":
            self.attn = \
                neural_network_tools.Linear(\
                        self.hidden_size * 2, 
                        hidden_size
                    )
            self.v = neural_network_tools.Parameter(\
                    torch.FloatTensor(hidden_size)
                )
    
    def dot_score(self, hidden_tensor, encoder_output_tensor):
        if type(hidden_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected hidden_tensor as Tensor but got as {0}"
                    .format(type(hidden_tensor))
                )
        if type(encoder_output_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected encoder_output_tensor as Tensor but got as {0}"
                    .format(type(encoder_output_tensor))
                )
        
        return torch.sum(hidden_tensor * encoder_output_tensor, dim=2)

    def general_score(self, hidden_tensor, encoder_output_tensor):
        if type(hidden_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected hidden_tensor as Tensor but got as {0}"
                    .format(type(hidden_tensor))
                )
        if type(encoder_output_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Ecpected encoder_output_tensor as Tensor but got as {0}"
                    .format(type(encoder_output_tensor))
                )
        
        energy = self.attn(encoder_output_tensor)
        return torch.sum(hidden_tensor * energy, dim=2)

    def concat_score(self, hidden_tensor, encoder_output_tensor):
        if type(hidden_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected hidden_tensor as Tensor but got as {0}"
                    .format(type(hidden_tensor))
                )
        if type(encoder_output_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected encoder_output_tensor as Tensor but got as {0}"
                    .format(type(encoder_output_tensor))
                )
        energy = self.attn(\
            torch.cat(\
                (\
                    hidden_tensor.expand(encoder_output_tensor.size(0), -1, -1), 
                    encoder_output_tensor
                ), 2)
            ).tanh()
        return torch.sum(self.v * energy, dim=2)

    #Overwrite method
    def forward(self, hidden, encoder_outputs):
        if type(hidden) is not torch.Tensor:
            raise TypeError(\
                    "Expected hidden as Tensor but got as {0}"
                    .format(type(hidden))
                )
        if type(encoder_outputs) is not torch.Tensor:
            raise TypeError(\
                    "Expected encoder_outputs as Tensor but got as {0}"
                    .format(type(encoder_outputs))
                )

        if self.method == "general":
            attention_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attention_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attention_energies = self.dot_score(hidden, encoder_outputs)
        
        attention_energies = attention_energies.t()
        return convolution_functions.softmax(\
            attention_energies, dim=1).unsqueeze(1)
