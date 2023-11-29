import os
import errno
import string
import argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from collections import Counter
import random


'''
Steps:
1. Remove passwords with length greater than 32.
2. Convert all of the passwords to lowercase
3. Calculate all unique characters and total passwords.
4. Create lookup dictionaries - char_to_idx and idx_to_char
5. Calculate the frequencies of each character and store them in dictionary
6. Sampling characters as targets based on frequencies - 
    A. P(wi) = 1 - sqrt(t / f(fwi)) 
        a. t --> Threshold Parameter. 
        b. f(wi) --> Frequency of word.

'''

# Parse the arguments for the input

# arg_parser = argparse.ArgumentParser(description = "Process Arguments for Preprocessing text.")
# arg_parser.add_argument("--prob_threshold", "-pbt", metavar = "THRESHOLD", dest = "PROB_THRESHOLD", type = float,
#                         help = "Character Subsampling Probability Threshold --> t.\tP(wi) = 1 - sqrt(t / f(wi)).\tDefault Value is 1e-5.")
# arg_parser.add_argument("--target_prob_filter", "-tpf", action = 'store_true', dest = "TARGET_PROB_FILTER", 
#                         help = "Include to subsample characters using the provided threshold.")
# arg_parser.add_argument("INPUT_FILE_PATH", metavar = "PATH", action = "store", help = "Path to Input Password File. One password on each line.")
# args = arg_parser.parse_args()

# Default Values 
TARGET_PROB_FILTER = False  #  if (not args.TARGET_PROB_FILTER) else args.TARGET_PROB_FILTER
INPUT_FILE_PATH = ""  #  if (not args.INPUT_FILE_PATH) else args.INPUT_FILE_PATH
PROB_THRESHOLD = 1e-5  #  if (not args.PROB_THRESHOLD) else args.PROB_THRESHOLD
CHARS = string.ascii_lowercase
NUMS = string.digits
SPECIAL_CHARS = string.punctuation
ENABLE_SAMPLING_TABLE_USING_FREQUENCY = True  # p(wi) = { [ { f(wi) } ^ (3/4) ] / sum_i[ f(wi) ^ (3/4)) ] }
VOCAB_SIZE = len(CHARS) + len(NUMS) + len(SPECIAL_CHARS)
NUM_NEG_SAMPLES = 1.0  # Total Negative Samples = Total Positive Samples * NUM_NEG_SAMPLES
SKIP_WINDOW_SIZE = 2 # Default window size for creating skip-grams
SKIP_SHUFFLE = True
SKIP_CATEGORICAL = False
EMBEDDING_DIM = 16

'''
Redundant Manual Implementation of Char to Index Lookup Table.
'''

# class lookupTables: 
#     def __init__(self, CHARS, SPECIAL_CHARS, NUMS, INPUT_FILE_PATH):
#         # Initialize 
#         self.chars = CHARS
#         self.nums = NUMS
#         self.special_chars = SPECIAL_CHARS
        
#     def char_to_idx(self):
#         # This table maps the character to its index 
#         self.all_chars = self.chars + self.nums + self.special_chars
#         self.char_to_idx_dict = {}
#         for idx, char in enumerate(self.all_chars, start = 0):
#             self.char_to_idx_dict[char] = idx
#         return self.char_to_idx_dict
    
#     def idx_to_char(self):
#         self.idx_to_char_dict = {}
#         self.all_chars = self.char_to_idx_dict.keys()
#         self.char_index = self.char_to_idx_dict.values()
#         for (char, index) in zip(self.all_chars, self.char_index):
#             self.idx_to_char_dict[index] = char
#         return self.idx_to_char_dict
    

class validateInput():
    def __init__(self, INPUT_FILE_PATH, SPECIAL_CHARS, NUMS):
        self.INPUT_FILE_PATH = INPUT_FILE_PATH
        self.CHARS = CHARS
        self.SPECIAL_CHARS = SPECIAL_CHARS
        self.NUMS = NUMS
        self.validate_input()
    
    def validate_input(self):
        if (not os.path.isfile(self.INPUT_FILE_PATH)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.INPUT_FILE_PATH)
        elif (not self.CHARS.isalpha()):
            raise ValueError("Specified characters are incorrect. Please try again!")
        elif(not self.special_char_check(self.SPECIAL_CHARS)):
            raise ValueError("Specified special characters are incorrect. Please try again!")
        elif(not self.NUMS.isdigit()):
            raise ValueError("Specified numbers are incorrect. Please try again!")
        else:
            return True    
        
    def special_char_check(self, special_str):
        for special_char in special_str:
            if (special_char not in string.punctuation):
                return False
            else:
                continue
        return True
                
        
# Tokenizing the characters using keras Tokenizer 

class tokenizeCharacters:
    def __init__(self, VOCAB_SIZE, INPUT_FILE_PATH, PROB_THRESHOLD, TARGET_PROB_FILTER):
        self.VOCAB_SIZE = VOCAB_SIZE + 1
        self.INPUT_FILE_PATH = INPUT_FILE_PATH
        self.PROB_THRESHOLD = PROB_THRESHOLD
        self.TARGET_PROB_FILTER = TARGET_PROB_FILTER
        self.tokenizer = None

    def create_tokenizer(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(self.VOCAB_SIZE, filters = "", lower = True, char_level = True)
        print("[+]  Tokenizer has been created.")

    def get_tokenizer_obj(self):
        if(not self.tokenizer):
            raise Exception("[+]  Failed. Tokenizer has to be initialized first.")
        else:
            return self.tokenizer

    def fit_tokenizer(self):
        pass_list = []
        print("[+]  Reading the password file into memory...")
        if(self.TARGET_PROB_FILTER):
            self.char_counter = Counter()
            with open(self.INPUT_FILE_PATH, "r") as pass_file:
                while True:
                    single_pass = pass_file.readline().rstrip("\n")
                    if(single_pass == ""):
                        break
                    else:
                        self.char_counter.update(single_pass)
            self.prob_drop_char = {}
            for key in self.char_counter:
                self.prob_drop_char[key] = (1 - np.sqrt(self.PROB_THRESHOLD / self.char_counter[key]))
            

        with open(self.INPUT_FILE_PATH, "r") as pass_file:
            while True:
                single_pass = pass_file.readline().rstrip("\n")
                if(single_pass == ""):
                    print("[+]  Success! Password file has been completely read.")
                    break
                else:
                    if(self.TARGET_PROB_FILTER):
                        modified_pass = []
                        for each_char in single_pass:
                            if(random.random() < (1 - self.prob_drop_char[each_char])):
                                modified_pass.append(each_char)
                            else:
                                continue
                        pass_list.append(modified_pass)
                    else:
                        pass_list.append(single_pass)
        print("[+]  Fitting the passwords and creating the dictionary...")
        self.tokenizer.fit_on_texts(pass_list)
        print("[+]  Success!")
        

    def tokenizer_config(self):
        print(self.tokenizer.get_config())

    def save_tokenizer_config(self, OUTPUT_FILE_NAME):
        self.tokenizer_config_json_str = self.tokenizer.to_json()
        with open(OUTPUT_FILE_NAME+".json", "w") as cfg_file:
            cfg_file.write(self.tokenizer_config_json_str)

    def load_tokenizer_using_config(self, TOKENIZER_FILE_PATH):
        _, file_extension = os.path.splitext(TOKENIZER_FILE_PATH)
        if(file_extension != ".json"):
            raise Exception("Incorrect File.")
        else:
            with open(TOKENIZER_FILE_PATH, "r") as tokenizer_cfg_file:
                tokenizer_config = tokenizer_cfg_file.read()
                tokenizer_cfg = json.loads(tokenizer_config)
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_cfg))
    
        

class createSkipGrams:
    def __init__(self, VOCAB_SIZE, SKIP_WINDOW_SIZE, NUM_NEG_SAMPLES, SKIP_SHUFFLE, SKIP_CATEGORICAL, ENABLE_SAMPLING_TABLE_USING_FREQUENCY):
        self.VOCAB_SIZE = VOCAB_SIZE + 1
        self.SKIP_WINDOW_SIZE = SKIP_WINDOW_SIZE
        self.NUM_NEG_SAMPLES = NUM_NEG_SAMPLES
        self.SKIP_SHUFFLE = SKIP_SHUFFLE
        self.SKIP_CATEGORICAL = SKIP_CATEGORICAL
        self.SAMPLING = ENABLE_SAMPLING_TABLE_USING_FREQUENCY
        self.x_train = []
        self.y_train = []
        
    def preprocessing_rank_word(self, tokenizer_obj, type = "zipf"):
        
        conf_str = tokenizer_obj.get_config()
        if(conf_str["document_count"] == 0):
            raise Exception("[+]  The tokenizer needs to be fit on the data first!")
        else:
            if(type == "mikolov"):
                self.char_frequency = {}
                self.char_freq_raised = {}
                self.char_prob = {}
                char_dict = json.loads(conf_str["word_counts"])
                for char_tokenized in list(char_dict.keys()):
                    self.char_frequency[char_tokenized] = char_dict[char_tokenized]
                    self.char_freq_raised[char_tokenized] = (char_dict[char_tokenized] ** 3/4)
                self.total_freq = sum(list(self.char_freq_raised.values()))
                for char_tokenized in list(self.char_freq_raised.keys()):
                    self.char_prob[char_tokenized] = (self.char_freq_raised[char_tokenized] / self.total_freq)
                self.sampling_table = [None] * (self.VOCAB_SIZE)
                self.word_index = json.loads(conf_str["word_index"])
                for char_tokenized in list(char_dict.keys()):
                    char_index = self.word_index[char_tokenized]
                    self.sampling_table[char_index - 1] = self.char_prob[char_tokenized]
                if(self.sampling_table.count(None) > 1):
                    raise Exception("[+]  Preprocessing failed. Sampling table has not been created.")
                print("[+]  Success! Sampling table has been created!")
            else:
                self.sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(self.VOCAB_SIZE)
                print("[+]  Success! Sampling table has been created!")
        


    def create_skipgrams(self, encoded_texts, tokenizer_obj):
        self.x_train = []
        self.y_train = []
        for text in encoded_texts:
            if(self.SAMPLING):
                train_n_test = tf.keras.preprocessing.sequence.skipgrams(text, self.VOCAB_SIZE, self.SKIP_WINDOW_SIZE, self.NUM_NEG_SAMPLES, self.SKIP_SHUFFLE, self.SKIP_CATEGORICAL, self.sampling_table)
            else:
                train_n_test = tf.keras.preprocessing.sequence.skipgrams(text, self.VOCAB_SIZE, self.SKIP_WINDOW_SIZE, self.NUM_NEG_SAMPLES, self.SKIP_SHUFFLE, self.SKIP_CATEGORICAL)
            self.x_train = self.x_train + [element for element in train_n_test[0]] 
            self.y_train = self.y_train + [element for element in train_n_test[1]]
        if(len(self.x_train) != len(self.y_train)):
            raise Exception("[+]  An Error Occured! Skipgrams couldn't be created.")
        print("[+]  Skipgrams have been created.")
        return(self.x_train, self.y_train) 


class trainingUtil:
    @staticmethod
    def save_data(train_set, label, CSV_EMBEDDING_FILE): 
        print("[+]  Appending to the file...")
        train_label = [[skipgram[0],skipgram[1],label] for skipgram,label in zip(train_set, label)]
        with open(CSV_EMBEDDING_FILE + ".csv", "a+") as csv_file:
            for example in train_label:
                csv_file.write(f"{example[0]},{example[1]},{example[2]}\n")
        print("[+]  Success! New set appended.")

    @classmethod
    def split_data(cls, FILE_PATH, TRAIN_PATH, TEST_PATH, TEST_SPLIT_SIZE):
        dataset_frame = pd.read_csv(FILE_PATH, header = None, names = ["Target", "Context", "Label"])
        skipgrams_train, skipgrams_test, labels_train, labels_test = train_test_split(dataset_frame[["Target", "Context"]], dataset_frame["Label"], test_size = TEST_SPLIT_SIZE, shuffle = True)
        cls.save_data(skipgrams_train.to_numpy(), labels_train.to_numpy(), TRAIN_PATH)
        cls.save_data(skipgrams_test.to_numpy(), labels_test.to_numpy(), TEST_PATH)
      

    @staticmethod
    def read_file_lines(INPUT_FILE_PATH):
        pass_list = []
        with open(INPUT_FILE_PATH, "r") as pass_file:
            while True:
                single_pass = pass_file.readline()
                if(single_pass == ""):
                    break
                else:
                    pass_list.append(single_pass)
        
        string_digits_ord = [ord(ele) for ele in string.digits]
        string_lowercase_ord = [ord(ele) for ele in string.ascii_lowercase]
        string_punctuation_ord = [ord(ele) for ele in string.punctuation]
        string_uppercase_ord = [ord(ele) for ele in string.ascii_uppercase]
        string_ord = string_digits_ord + string_lowercase_ord + string_punctuation_ord + string_uppercase_ord

        modified_list_text = []

        for text in pass_list:

            modi = []
            for char in text:
                if(ord(char) in string_ord):
                    modi.append(char.lower())
                else:
                    continue
            modified_list_text.append("".join(char for char in modi))
        return modified_list_text
        




