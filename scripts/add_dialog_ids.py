import argparse
import glob
import h5py
import json
import os
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm


parser = argparse.ArgumentParser()

# Input files
parser.add_argument('-input_json', help='Input json file')
parser.add_argument('-input_json_train', default='visdial_1.0_train.json', help='Input `train` json file')
parser.add_argument('-input_json_val', default='visdial_1.0_val.json', help='Input `val` json file')
parser.add_argument('-input_json_test', default='visdial_1.0_test.json', help='Input `test` json file')


def add_dialog_ids(data):
    for i, entry in enumerate(data['data']['dialogs']):
        entry["conversation_id"] = i


if __name__ == "__main__":
    args = parser.parse_args()

    if args.input_json:
        print('Processing single json from \"{}\"'.format(args.input_json))
        data = json.load(open(args.input_json, 'r'))
        add_dialog_ids(data)
        json.dump(data, open(args.input_json, 'w'))
    else:
        print('Processing jsons...')
        print('Processing training data json from \"{}\"'.format(args.input_json_train))
        data_train = json.load(open(args.input_json_train, 'r'))
        add_dialog_ids(data_train)
        json.dump(data_train, open(args.input_json_train, 'w'))
        del data_train

        print('Processing validation data json from \"{}\"'.format(args.input_json_val))
        data_val = json.load(open(args.input_json_val, 'r'))
        add_dialog_ids(data_val)
        json.dump(data_val, open(args.input_json_val, 'w'))
        del data_val

        print('Processing testing data json from \"{}\"'.format(args.input_json_test))
        data_test = json.load(open(args.input_json_test, 'r'))
        add_dialog_ids(data_test)
        json.dump(data_test, open(args.input_json_test, 'w'))
        del data_test


    
