import os
import json

setting = None
os.chdir(os.getenv("PYTHONPATH"))

with open("./setting.json", "r") as setting_json:
    setting = json.load(setting_json)


from preparation.pre_processing import pre_processing
from modeling.trainning import train
from modeling.test_model import *

pre_processing()
train(train_split_ratio=setting["train_split_ratio"])

if setting["validation"]:
    intension_model_testing()
    contextual_model_testing()