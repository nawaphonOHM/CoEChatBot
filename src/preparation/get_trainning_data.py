from src.preparation.reader_json_file import read as json_file_read
import src.classes.PreProcessingDataModel as PreData
import os
import csv
import pythainlp


def get_trainning_data():
     raw_datas = json_file_read()
     raws = []
     for raw_data in raw_datas:
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
          inqury_sentence_tokenized.append("{" + raw_data.getIntension() + "}")
          response_sentence_tokenized.append("{" + raw_data.getChangedIntension() + "}")
          raws.append([inqury_sentence_tokenized, response_sentence_tokenized])
     return raws