import os
import json
import csv
import pickle
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
from pythainlp.spell import correct as typo_checking
import src.bag_of_words as bag_of_words

def pre_processing():
    raw_input = None
    fomatted_data = []
    stop_word = corpus.thai_stopwords()
    bag = bag_of_words.Bag()
    work_directory = os.getcwd()
    cleaned_data = \
        open(os.path.join(work_directory, "data/processed/cleaned_data.csv"), "w")
    writer = csv.writer(cleaned_data)
    writer.writerow(["Pair_Type", "Query_Sentence_Pattern", "Response_Sentence"])

    with open(os.path.join(work_directory, "data/raw/raw_data.json"), "r") \
        as raw:
            raw_input = json.load(raw)

    for pair in raw_input:
        type_pairing = pair["intension"]
        response_sentence = pair["response_sentence"]
        query_sentence = ""
        bag.addIntentionType(type_pairing)

        cleaned_word_inquery = pair["inquery_sentence"]
        cleaned_word_inquery = \
            tokenization.word_tokenize(cleaned_word_inquery, keep_whitespace=False)
        cleaned_word_inquery = \
            [typo_checking(word) for word in cleaned_word_inquery if word not in stop_word]
        for word in cleaned_word_inquery:
            bag.addWord(word)
            query_sentence = query_sentence + word + " "
        bag.setResponseSentence(type_pairing, response_sentence)
    
        writer.writerow(\
                [type_pairing, query_sentence.strip(), pair["response_sentence"]]
            )

    bag.sort_em()

    with open(os.path.join(work_directory, "data/processed/bag_of_word_.pkl"), "wb")\
        as bag_saved_file:
            pickle.dump(bag, bag_saved_file, protocol=pickle.HIGHEST_PROTOCOL)

    cleaned_data.close()