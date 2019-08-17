import torch
import random
import torch.nn as neural_network_tools
from src.modeling.seq2seq.Seq2SeqEncoder import Seq2SeqEncoder
from src.modeling.seq2seq.Seq2SeqDecoder import Seq2SeqDecoder
from src.modeling.loss_function import mask_negative_log_likelihood_loss
from torch import optim as optimization
from torch.nn import Embedding as Embedding
from torch.optim import Adam as Adam

def train(\
        input_variable_tensor, 
        lengths_vector, 
        target_variable_tensor, 
        mask_tensor, 
        max_target_len, 
        encoder_part_module, 
        decoder_part_module, 
        embedding_tensor, 
        encoder_optimizer, 
        decoder_optimizer, 
        batch_size, 
        clip, 
        START_SENTENCE_TOKEN, 
        teacher_forcing_ratio, 
        used_device
    ):
        if type(input_variable_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected input_variable_tensor as Tensor but got as {0}"
                    .format(type(input_variable_tensor))
                )
        if type(lengths_vector) is not torch.Tensor:
            raise TypeError(\
                    "Expected lengths_vector as Tensor but got as {0}"
                    .format(type(lengths_vector))
                )
        if type(target_variable_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected target_variable_tensor as Tensor but got as {0}"
                    .format(type(target_variable_tensor))
                )
        if type(mask_tensor) is not torch.Tensor:
            raise TypeError(\
                    "Expected mask_tensor as Tensor but got as {0}"
                    .format(type(mask_tensor))
                )
        if type(max_target_len) is not int:
            raise TypeError(\
                    "Expected max_target_len as int but got as {0}"
                    .format(type(max_target_len))
                )
        if max_target_len < 1:
            raise ValueError(\
                    "a max_target_len must be more than 0 but got {0}"
                    .format(max_target_len)
                )
        if type(encoder_part_module) is not Seq2SeqEncoder:
            raise TypeError(\
                    "Expected encoder_part_module as Seq2SeqEncoder but got as {0}"
                    .format(type(encoder_part_module))
                )
        if type(decoder_part_module) is not Seq2SeqDecoder:
            raise TypeError(\
                    "Expected decoder_part_module as Seq2SeqEncoder but got as {0}"
                    .format(type(decoder_part_module))
                )
        if type(embedding_tensor) is not Embedding:
            raise TypeError(\
                    "Expected embedding_tensor as Embedding but got as {0}"
                    .format(type(embedding_tensor))
                )
        if type(encoder_optimizer) is not Adam:
            raise TypeError(\
                    "Expected encoder_optimizer as Adam but got as {0}"
                    .format(type(encoder_optimizer))
                )
        if type(decoder_optimizer) is not Adam:
            raise TypeError(\
                    "Expected decoder_optimizer as Adam but got as {0}"
                    .format(type(decoder_optimizer))
                )
        if type(batch_size) is not int:
            raise TypeError(\
                    "Expected batch_size as int but got as {0}"
                    .format(type(batch_size))
                )
        if batch_size < 1:
            raise ValueError(\
                    "batch_size must be more than 0 but got as {0}"
                    .format(batch_size)
                )
        if type(clip) is not float:
            raise TypeError(\
                    "Expected clip as float but got as {0}"
                    .format(type(clip))
                )
        if type(START_SENTENCE_TOKEN) is not int:
            raise TypeError(\
                    "Expected START_SENTENCE_TOKEN as int but got as {0}"
                    .format(type(START_SENTENCE_TOKEN))
                )
        if type(teacher_forcing_ratio) is not float:
            raise TypeError(\
                    "Expected teacher_forcing_ratio as float but got as {0}"
                    .format(type(teacher_forcing_ratio))
                )
        if teacher_forcing_ratio < 0 or teacher_forcing_ratio > 1:
            raise ValueError(\
                    "a teacher_forcing_ratio must be [0.0, 1.0] but got {0}"
                    .format(teacher_forcing_ratio)
                )
        if type(used_device) is not str:
            raise TypeError(\
                    "Expected used_device as str but got as {0}"
                    .format(type(used_device))
                )

        loss = 0
        print_losses = []
        n_totals = 0

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_variable_tensor = input_variable_tensor.to(used_device)
        lengths_vetor = lengths_vector.to(used_device)
        target_variable_tensor = target_variable_tensor.to(used_device)
        mask_tensor = mask_tensor.to(used_device)

        encodert_outputs_tensor, encoder_hidden_tensor = \
            encoder_part_module(input_variable_tensor, lengths_vector)

        decoder_input_tensor = torch.LongTensor(\
                [[START_SENTENCE_TOKEN for _ in range(batch_size)]]
            ).to(used_device)

        decoder_hidden_tensor = \
            encoder_hidden_tensor[:decoder_part_module.n_layers]

        use_teacher_forcing = \
            True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output_tensor, decoder_hidden_tensor = \
                    decoder_part_module(\
                            decoder_input_tensor, 
                            decoder_hidden_tensor, 
                            encodert_outputs_tensor
                        )
                decoder_input_tensor = target_variable_tensor[t].view(1, -1)

                mask_loss, n_total = \
                    mask_negative_log_likelihood_loss(\
                            decoder_output_tensor, 
                            target_variable_tensor[t], 
                            mask_tensor[t], 
                            used_device
                        )
                loss = loss + mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals = n_totals + n_total
        else:
            for t in range(max_target_len):
                decoder_output_tensor, decoder_hidden_tensor = \
                    decoder_part_module(\
                            decoder_input_tensor, 
                            decoder_hidden_tensor, 
                            encodert_outputs_tensor
                        )
                _, largest_tensor = decoder_output_tensor.topk(1)
                decoder_input_tensor = torch.LongTensor(\
                        [
                            largest_tensor[i][0] for i in range(batch_size)
                        ]
                    )
                decoder_input_tensor = decoder_input_tensor.to(used_device)
                mask_loss, n_total = mask_negative_log_likelihood_loss(\
                        decoder_output_tensor, 
                        target_variable_tensor[t], 
                        mask_tensor[t], 
                        used_device
                    )
                loss = loss + mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals = n_totals + n_total
        
        loss.backward()

        _ = neural_network_tools.utils.clip_grad_norm_(\
                encoder_part_module.parameters(), 
                clip
            )
        _ = neural_network_tools.utils.clip_grad_norm_(\
                decoder_part_module.parameters(), clip
            )
        
        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals
                
