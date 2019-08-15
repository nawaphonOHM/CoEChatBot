import os
from src.classes.PreProcessingDataModel import PreProcessingDataModel as PreProcessing
import json

def read():
    raw_input = None
    returned_data = []

    work_directory = os.getcwd()
    with open(os.path.join(work_directory, "data/raw/data.json"), "r") \
        as raw:
            raw_input = json.load(raw)

    for input_obj in raw_input:
        new_obj = PreProcessing()
        new_obj.setIntension(input_obj["intension"])
        new_obj.setInquerySentence(input_obj["inquery_sentence"])
        new_obj.setResponseSentence(input_obj["response_sentence"])
        new_obj.setChangedIntension(input_obj["changed_intension"])

        returned_data.append(new_obj)

    return returned_data
