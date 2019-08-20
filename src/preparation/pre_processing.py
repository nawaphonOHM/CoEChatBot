import os
import json
import csv
import pythainlp.tokenize as tokenization
import pythainlp.corpus.common as corpus
import src.bag_of_words as bag_of_words

raw_input = None
fomatted_data = []
stop_word = corpus.thai_stopwords()
bag = bag_of_words.BagsOfWords()
cleaned_data = open("./data/processed/cleaned_data.csv", "w")
writer = csv.writer(cleaned_data)
writer.writerow(["Pair_Type", "Query_Sentence_Pattern", "Response_Sentence"])
work_directory = os.getcwd()

with open(os.path.join(work_directory, "data/raw/raw_data.json"), "r") \
    as raw:
        raw_input = json.load(raw)

for pair in raw_input:
    type_pairing = pair["intension"]
    response_sentence = pair["response_sentence"]
    query_sentence = ""

    cleaned_word = pair["inquery_sentence"]
    cleaned_word = tokenization.word_tokenize(cleaned_word)
    cleaned_word = [word for word in cleaned_word if word not in stop_word]
    for word in cleaned_word:
        bag.addWord(word)
        query_sentence = query_sentence + word + " "

    writer.writerow([type_pairing, query_sentence, response_sentence])

cleaned_data.close()
    
