import os
import json
import csv
import pickle
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
from pythainlp.spell import correct as typo_checking
import src.bag_of_words as bag_of_words

def pre_processing():
    print("Pre-Processing is starting")
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

    len_data = len(raw_input)
    counter = 1
    write_to_csv = False

    for pair in raw_input:
        print("\rCleanning data {0} from {1}".format(counter, len_data), end="")
        write_to_csv = write_to_csv and False
        query_sentence = None
        response_sentence = None
        type_pairing = pair["intension"]
        bag.addIntentionType(type_pairing)
        if "intention_map" in pair.keys():
            bag.setIntentionMap(type_pairing, pair["intention_map"])
        if "intension_set" in pair.keys():
            bag.setIntentionSet(type_pairing, pair["intension_set"])
        if "inquery_sentence" in pair.keys():
            write_to_csv = write_to_csv or True
            query_sentence = ""
            cleaned_word_inquery = pair["inquery_sentence"]
            cleaned_word_inquery = \
                tokenization.word_tokenize(cleaned_word_inquery, keep_whitespace=False)
            cleaned_word_inquery = \
                [typo_checking(word) for word in cleaned_word_inquery if word not in stop_word]
            for word in cleaned_word_inquery:
                bag.addWord(word)
                query_sentence = query_sentence + word + " "
            query_sentence.strip()

        if "response_sentence" in pair.keys():
            response_sentence = pair["response_sentence"]
            bag.setResponseSentence(type_pairing, response_sentence)
        
        if write_to_csv:
            print(" Writing [{0}, {1}]".format(
                    query_sentence, response_sentence
                ), end=""
            )
            writer.writerow(\
                [type_pairing, query_sentence.strip(), response_sentence]
            )
            print(" Done", end="")
        else:
            print("\nIgnore writing pair: {0}".format(pair))
        counter += 1

    bag.sort_em()

    print("\nsaving cleaned data...", end="")
    with open(os.path.join(work_directory, "data/processed/bag_of_word_.pkl"), "wb")\
        as bag_saved_file:
            pickle.dump(bag, bag_saved_file, protocol=pickle.HIGHEST_PROTOCOL)

    cleaned_data.close()
    print("\rsaving cleaned data...done")

pre_processing()