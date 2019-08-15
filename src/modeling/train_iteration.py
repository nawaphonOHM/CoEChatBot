

def train_iteration(\
        bag_of_word, 
        pairs, 
        encoder_part, 
        decoder_part, 
        encoder_optimizer, 
        decoder_optimizer, 
        embedding_tensor, 
        encoder_n_layers, 
        decoder_n_layers, 
        save_directory, 
        n_iteration, 
        batch_size, 
        clip, 
        corpus_name
    ):
        training_batches = []