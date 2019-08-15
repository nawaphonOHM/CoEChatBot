import os
import src.classes.PreProcessingDataModel as PreProcessing
import json

def read():
    raw_input = None

    work_directory = os.getcwd()
    with open(os.path.join(work_directory, "data/raw/data.json"), "r") \
        as raw:
            raw_input = json.load(raw)

    return raw_input
