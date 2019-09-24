import os
import json
import csv
import pickle
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
import src.bag_of_words as bag_of_words

def pre_processing():
    print("Pre-Processing is starting")
    raw_input = None
    fomatted_data = []
    stop_word = corpus.thai_stopwords()
    bag = bag_of_words.Bag()
    work_directory = os.getcwd()
    cleaned_data = \
        open(os.path.join(work_directory, "data/processed/inquery_cleaned_data.csv"), "w")
    writer = csv.writer(cleaned_data)
    writer.writerow(["Intention", "Query_Sentence_Pattern_Cleaned", "Query_Sentence_Pattern"])

    with open(os.path.join(work_directory, "data/raw/inquery_raw_data.json"), "r") \
        as raw:
            raw_input = json.load(raw)
    with open(os.path.join(work_directory, "data/raw/excluded_stop_words.json"), "r") \
        as excluded_stop_words_json:
            bag.set_excluded_stop_words(json.load(excluded_stop_words_json))


    len_data = len(raw_input)
    counter = 1
    print("Being on inquery raw data")

    for data_set in raw_input:
        print(\
            "\rCleaned data {0:.2f}% ({1} from {2})"\
                .format((counter / len_data) * 100, counter, len_data), end="")
        intention = data_set["intention"]
        bag.add_intention(intention)

        for sentence in data_set["inquery_sentence"]:
            sentence_cleaned = ""
            word_in_sentence = tokenization.word_tokenize(sentence, keep_whitespace=False)
            cleaned_word_in_sentence = \
                [word for word in word_in_sentence \
                        if word not in stop_word or bag.has_in_excluded_stop_words(word)]
            for word in cleaned_word_in_sentence:
                bag.add_word(word)
                sentence_cleaned = sentence_cleaned + word + " "
            sentence_cleaned = sentence_cleaned.strip()
            writer.writerow(\
                    [intention, sentence_cleaned, sentence]
                )
        counter += 1
    
    cleaned_data.close()

    cleaned_data = \
        open(os.path.join(work_directory, "data/processed/responsing_cleaned_data.csv"), 'w')
    writer = csv.writer(cleaned_data)
    writer.writerow(["response_classes", "intention", "state"])
    with open(os.path.join(work_directory, "data/raw/contextual_raw_data.json"), "r") \
        as raw:
            raw_input = json.load(raw)

    len_data = len(raw_input)
    counter = 1
    print("\nBeing on contextual raw data")

    for data_set in raw_input:
        print(\
            "\rCleaned data {0:.2f}% ({1} from {2})"\
                .format((counter / len_data) * 100, counter, len_data), end="")
        response_class = data_set["response_class"]
        for contextual in data_set["intention_state"]:
            intention, state = contextual.split(" ")
            bag.add_intention_contextual(intention)
            bag.add_state_contextual(state)
            writer.writerow([response_class, intention, state])
        counter += 1

    cleaned_data.close()

    cleaned_data = \
        open(os.path.join(work_directory, \
            "data/processed/responsing_details_cleaned_data.csv"), 'w')
    writer = csv.writer(cleaned_data)
    writer.writerow(["response_classes", "response_messages", "state_setting"])
    with open(os.path.join(work_directory, "data/raw/response_raw_data.json")) \
        as raw:
            raw_input = json.load(raw)
    
    len_data = len(raw_input)
    counter = 1
    print("\nBeing on responsing raw data")

    for data_set in raw_input:
        print(\
            "\rCleaned data {0:.2f}% ({1} from {2})"\
                .format((counter / len_data) * 100, counter, len_data), end="")
        response_class = data_set.pop("response_class")
        bag.add_response_class(response_class)
        bag.set_response_sentence(response_class, data_set)
        intention_set = data_set["intention_set"] \
            if data_set["intention_set"] != None else "null"
        writer.writerow(\
                [response_class, data_set["response_sentence"], intention_set]
            )
        counter += 1

    cleaned_data.close()
    bag.sort_items()

    print("\nsaving data...", end="")
    with open(os.path.join(work_directory, "data/processed/bag_of_word_.pkl"), "wb")\
        as bag_saved_file:
            pickle.dump(bag, bag_saved_file, protocol=pickle.HIGHEST_PROTOCOL)

    print("\rsaving cleaned data...done")

pre_processing()