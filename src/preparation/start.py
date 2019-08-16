import src.preparation.reader_json_file as json_file
import src.classes.PreProcessingDataModel as PreData
import os
import csv
import pythainlp

raw_datas = json_file.read()

work_directory = os.getcwd()
write_file = open(os.path.join(work_directory, "data/raw/data_with_tag.csv"), "w")
writer = csv.writer(write_file)
writer.writerow(["inquery_sentence", "response_sentence"])
raws = []

for raw_data in raw_datas:
    seperater = "|"
    inqury_sentence_tokenized = \
        pythainlp.tokenize.word_tokenize(\
                raw_data.getInquerySentence(), \
                engine="newmm", 
                keep_whitespace=False
            )
    response_sentence_tokenized = \
        pythainlp.tokenize.word_tokenize(\
                raw_data.getResponseSentence(), 
                engine="newmm", 
                keep_whitespace=False
            )
    raw = [seperater.join(inqury_sentence_tokenized) + " {" + \
            raw_data.getIntension() + "}"]
    raw.append(seperater.join(response_sentence_tokenized) + " {" + \
            raw_data.getChangedIntension() + "}")
    raws.append(raw)

writer.writerows(raws)
write_file.close()

