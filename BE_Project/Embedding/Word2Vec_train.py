import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Model, metrics, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy
from tensorboard.plugins import projector
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import *
import pandas as pd
import numpy as np
import math
import errno
import json
import os 
'''

This script will be used to train a Logistic Classifier
for training character embeddings using the Functional Keras
API. 

'''


OPTIMIZER = 'adam' #  Other Option is RMSprop, SGD
LEARNING_RATE = 0.001
LOSS = 'binary_crossentropy'
METRICS = [tf.keras.metrics.BinaryAccuracy()]
OUTPUT_MODEL_PATH = 'mymodel'
MODEL_SAVE_FORMAT = 'default'  # Other value is h5. Default is better.
WT_SAVE_FORMAT = "tf"  # Other value is h5. Default is better.
ENABLE_MODEL_CHECKPOINT = True 
BATCH_SIZE = 32  #  Default 
VALIDATION_BATCH_SIZE = 32 #  Default
EPOCHS = 20  #  Default 
VALIDATION_FREQ = 1
TRAIN_SEQ_WORKERS = 4
USE_MULTIPROCESSING = True
EMBED_REGULARIZER = True


class modelOperations:
    def __init__(self, ENABLE_MODEL_CHECKPOINT, EMBED_REGULARIZER, TOKENIZER_OBJ, VOCAB_SIZE, EMBEDDING_DIM, LOG_DIR, TENSORBOARD_LOG_DIR):
        self.char_emb_model = None
        self.tensorboard_log_dir = TENSORBOARD_LOG_DIR
        self.checkpoint = ENABLE_MODEL_CHECKPOINT
        self.log_dir = LOG_DIR
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = EMBEDDING_DIM
        self.model_checkpoint = None
        self.optimizer_inst = None
        self.embed_regularizer = EMBED_REGULARIZER
        self.tokenizer_obj = TOKENIZER_OBJ
        self.fit_flag = None
        self.metadata_file = "metadata.tsv"

    def create_embedding_model(self):
        input_target = Input(shape = (1, ))
        input_context = Input(shape = (1 , ))
        if(self.embed_regularizer):
            target_embedding = Embedding(self.vocab_size + 1, self.embedding_dim, embeddings_initializer = "glorot_uniform", input_length = 1, name = "target_embedding", activity_regularizer = l2(), mask_zero = True)(input_target)
            context_embedding = Embedding(self.vocab_size + 1, self.embedding_dim, embeddings_initializer = "glorot_uniform", input_length = 1, name = "context_embedding", activity_regularizer = l2(), mask_zero = True)(input_context)
        else:    
            target_embedding = Embedding(self.vocab_size + 1, self.embedding_dim, embeddings_initializer = "glorot_uniform", input_length = 1, name = "target_embedding", mask_zero = True)(input_target)
            context_embedding = Embedding(self.vocab_size + 1, self.embedding_dim, embeddings_initializer = "glorot_uniform", input_length = 1, name = "context_embedding", mask_zero = True)(input_context)

        target_embedding_reshaped = Reshape((self.embedding_dim, ))(target_embedding)
        context_embedding_reshaped = Reshape((self.embedding_dim, ))(context_embedding)

        emb_merged = Dot(axes = 1)([target_embedding_reshaped, context_embedding_reshaped])
        sigmoid_sim_op = Dense(1, kernel_initializer="glorot_uniform", activation = "sigmoid", name = "Output_Sigmoid_Layer")(emb_merged)

        self.char_emb_model = Model(inputs=[input_target ,input_context], outputs=[sigmoid_sim_op])
        print("Remember the models needs to be compiled later on.")


    def save_keras_model(self, OUTPUT_MODEL_PATH, MODEL_SAVE_FORMAT):
        print("[+]  Saving the model to the disk. Saves the architecture, optimizer, losses, metrics and state of the model.")
        if(self.char_emb_model):
            raise Exception("Failed. Model has not been created yet.")
        if(MODEL_SAVE_FORMAT == 'default'):
            self.char_emb_model.save(OUTPUT_MODEL_PATH)
        else:
            self.char_emb_model.save(OUTPUT_MODEL_PATH, save_format = 'h5')

    def load_keras_model(self, INPUT_MODEL_PATH):
        if (not os.path.isfile(INPUT_MODEL_PATH)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), INPUT_MODEL_PATH)
        else:
            self.char_emb_model = keras.models.load_model(INPUT_MODEL_PATH)
        

    def save_keras_model_weights(self, OUTPUT_MODEL_WT, WT_SAVE_FORMAT):
        print("[+]  Saving keras model weights")
        if(self.char_emb_model):
            raise Exception("Failed. Model has not been created yet.")
        else:
            self.char_emb_model.save_weights(OUTPUT_MODEL_WT, save_format = WT_SAVE_FORMAT)
            print("[+]  Success! Weights have been saved.")

    def load_keras_model_weights(self, INPUT_MODEL_WT):
        print("[+]  Loading Model Weights..")
        if(self.char_emb_model):
            print("Model has to be first configured using the same API. Please use create_embedding_model() to remedy this and then try loading weights.")
        else:
            self.char_emb_model.load_weights(INPUT_MODEL_WT)

    def load_keras_model_config(self, INPUT_MODEL_CONFIG_JSON):
        print("[+]  Loading model configuration/architecture in JSON Format")
        _, file_extension = os.path.splitext(INPUT_MODEL_CONFIG_JSON)
        if(file_extension != ".json"):
            raise Exception("Incorrect File.")
        else:
            with open(INPUT_MODEL_CONFIG_JSON, "r") as model_cfg_file:
                model_cfg = json.loads(model_cfg_file)
            self.char_emb_model = keras.models.model_from_json(json.dumps(model_cfg))

    def save_keras_model_config(self, OUTPUT_MODEL_CONFIG_JSON):
        print("[+]  Saving model configuration/architecture in JSON Format.")
        if(self.char_emb_model):
            raise Exception("Failed. Model has not been created yet.")
        else:
            with open(OUTPUT_MODEL_CONFIG_JSON, "w") as cfg_file:
                json_str = self.char_emb_model.to_json()
                cfg_file.write(json_str)
            print("[+]  Success! Model configuration has been saved.")

    def model_checkpoint_callback(self, CHECKPOINT_PATH):
        if(os.path.isdir(CHECKPOINT_PATH)):
            self.model_checkpoint = ModelCheckpoint(filepath =  CHECKPOINT_PATH + "_" + "{epoch:03d}", save_weights_only = False, verbose = 1, save_freq = "epoch")
        else:
            print("[+]  Specified Directory doesn't exist.")
            bool_dir = self.ask_again("Do you want to make a new one? (y/n)")
            if(bool_dir):
                os.mkdir(CHECKPOINT_PATH)
                self.model_checkpoint = ModelCheckpoint(filepath =  CHECKPOINT_PATH + "_" + "{epoch:03d}", save_weights_only = False, verbose = 1, save_freq = "epoch")
            else:
                print("Warning: Checkpoints will not be created.")
                return

    def setup_model_tensorboard(self):
        self.tensorboard_callback = TensorBoard(log_dir = self.tensorboard_log_dir, histogram_freq = 1, write_graph = True, write_images = True, update_freq = 'batch')
        print("[+]  Writing metadata to the tensorboard logging dir.")
        if (not os.path.isdir(self.tensorboard_log_dir)):
            print("[+]  Directory path doesn't exist, making a new directory for saving metadata.")
            os.mkdir(self.tensorboard_log_dir)
        self.write_metadata(self.tensorboard_log_dir)
        print("[+]  Success!")
        print("[+]  Creating a custom callback to write the embeddings at the end of each epoch..")
        self.tensorboard_config_projector = projector.ProjectorConfig()
        self.tensorboard_config_embedding = self.tensorboard_config_projector.embeddings.add()
        self.tensorboard_config_embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        self.tensorboard_config_embedding.metadata_path = os.path.join(self.tensorboard_log_dir, self.metadata_file)
        self.tensorboard_emb_callback = embedding_tensorboard_callback(self.tensorboard_config_projector, self.tensorboard_log_dir)


    def optimizer_instance(self, OPTIMIZER):
        if(OPTIMIZER == "adam"):
            self.optimizer_inst = Adam(learning_rate = LEARNING_RATE, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, name = "Adam", amsgrad = False)
        elif(OPTIMIZER == "RMSprop"):
            self.optimizer_inst = RMSprop(learning_rate = LEARNING_RATE, rho = 0.9, momentum = 0.0, epsilon = 1e-07, centered = "False", name = "RMSprop")
        elif(OPTIMIZER == "SGD"):
            self.optimizer_inst = SGD(learning_rate = LEARNING_RATE, momentum = 0.0, nesterov = False, name = "SGD")
        else:
            print("[+]  Failed. Selected optimizer is incorrect. Choose from [adam, RMSprop, SGD]")
            self.optimizer_inst = None
            return 

    def model_compile(self, METRICS):
        if(not self.optimizer_inst):
            print("[+]  Compiling Failed. Choose an optimizer first.")
        elif(not self.char_emb_model):
            print("[+]  Compiling Failed. Please use create_embedding_model() to remedy this and then try to compile.")
        else:
            self.char_emb_model.compile(optimizer = self.optimizer_inst, loss = "binary_crossentropy", metrics = METRICS)
            print("[+]  Model has been successfully compiled.")
            return 

    def ask_again(self, str_ask):
        bool_val = input(str_ask)
        if(bool_val == "y"):
            return True
        elif(bool_val == "n"):
            return False
        else:
            self.ask_again(str_ask)

    def model_fit(self, TRAIN_FILE_PATH, VALIDATION_FILE_PATH, BATCH_SIZE, VALIDATION_BATCH_SIZE, EPOCHS, VERBOSE, SHUFFLE, TRAIN_SEQ_WORKERS, USE_MULTIPROCESSING, VALIDATION_FREQ):
        self.train_file_path = TRAIN_FILE_PATH
        self.validation_file_path = VALIDATION_FILE_PATH
        self.batch_size = BATCH_SIZE
        self.validation_batch_size = VALIDATION_BATCH_SIZE
        self.epochs = EPOCHS
        self.verbose = VERBOSE
        self.shuffle = SHUFFLE
        self.validation_freq = VALIDATION_FREQ
        self.workers = TRAIN_SEQ_WORKERS
        self.multi_process = USE_MULTIPROCESSING
        self.xy_train = embeddingIOSequenceTrain(BATCH_SIZE = self.batch_size, CSV_EMBEDDING_FILE = self.train_file_path)
        self.xy_validation = embeddingIOSequenceValidation(VALIDATION_BATCH_SIZE = self.validation_batch_size, CSV_EMBEDDING_FILE = self.validation_file_path)
        print("[+]  WARNING: some of the validation data may not be used. (size < Validation Batch Size). To reduce the unused data, lower the validation batch size.")
        print("[+]  Fitting the model...")
        self.fit_flag = True
        self.model_history = self.char_emb_model.fit(x = self.xy_train, epochs = self.epochs, verbose = self.verbose, callbacks = [self.model_checkpoint, self.tensorboard_callback, self.tensorboard_emb_callback],
            validation_data = self.xy_validation.get_data(), shuffle = 'batch', steps_per_epoch = self.xy_train.__len__(),
            validation_batch_size = self.validation_batch_size, validation_freq = self.validation_freq, 
            max_queue_size = self.xy_train.__len__(), workers = self.workers, use_multiprocessing = self.multi_process)

    def write_metadata(self, META_DIR):
        self.tokenizer_conf_str = self.tokenizer_obj.get_config()
        self.word_index_map = json.loads(self.tokenizer_conf_str["word_index"])
        with open(os.path.join(META_DIR, self.metadata_file), "w") as meta_file:
                for char_tokenized in self.word_index_map.keys():
                    meta_file.write(f"{char_tokenized}\n")

    def create_metadata_file(self):
        if(not self.fit_flag):
            print("[+]  Failed! The model has to be fit on data first!")
            return
        self.tokenizer_conf_str = self.tokenizer_obj.get_config()
        self.word_index_map = json.loads(self.tokenizer_conf_str["word_index"])
        self.metadata_file = "metadata.tsv"
        if(not self.char_emb_model):
            print("[+]  Failed! Model has to be initialized first!")
        else:
            if(not os.path.isdir(self.tensorboard_log_dir)):
                os.mkdir(self.tensorboard_log_dir)
            with open(os.path.join(self.tensorboard_log_dir, self.metadata_file), "w") as meta_file:
                for char_tokenized in self.word_index_map.keys():
                    meta_file.write(f"{char_tokenized}\n")
            
    def visualize_model(self, OPTIONAL_DIR = None):
        if(not self.fit_flag):
            print("[+]  Failed! The model has to be fit on data first!")
            return
        self.tokenizer_conf_str = self.tokenizer_obj.get_config()
        self.word_index_map = json.loads(self.tokenizer_conf_str["word_index"])
        self.metadata_file = "metadata.tsv"
        if(not self.char_emb_model):
            print("[+]  Failed! Model has to be initialized first!")
        else:
            self.weight_matrix = tf.Variable(self.char_emb_model.layers[2].get_weights()[0][1:], dtype = tf.float64)
            self.train_checkpoint = tf.train.Checkpoint(embedding = self.weight_matrix)
            if(not os.path.isdir(self.log_dir)):
                os.mkdir(self.log_dir)
            self.train_checkpoint_save_path = self.train_checkpoint.save(os.path.join(self.log_dir, "embedding.ckpt"))
            with open(os.path.join(self.log_dir, self.metadata_file), "w") as meta_file:
                for char_tokenized in self.word_index_map.keys():
                    meta_file.write(f"{char_tokenized}\n")
            self.config_projector = projector.ProjectorConfig()
            self.config_embedding = self.config_projector.embeddings.add()
            self.config_embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
            self.config_embedding.metadata_path = os.path.join(self.log_dir, self.metadata_file)
            projector.visualize_embeddings(self.log_dir, self.config_projector)        
                
    def print_model_summary(self):
        print(self.char_emb_model.summary())


class embeddingIOSequenceTrain(Sequence):
    def __init__(self, BATCH_SIZE, CSV_EMBEDDING_FILE):
        self.batch_size = BATCH_SIZE
        self.csv_path =  CSV_EMBEDDING_FILE
        _, file_extension = os.path.splitext(CSV_EMBEDDING_FILE)
        if(file_extension != ".csv"):
            raise Exception("[+]  Incorrect Training Set File. Please choose correct Training Embedding CSV.")
    
    def __len__(self):
        with open(self.csv_path, "r") as csv_file:
            for count, _ in enumerate(csv_file):
                pass
            return(math.floor((count + 1) / self.batch_size))

    def __getitem__(self, current_batch_num):
        train_label = pd.read_csv(self.csv_path, header = None, names = ["Target", "Context", "Label"], dtype = np.int32, skiprows = (self.batch_size * (current_batch_num - 1)), nrows = self.batch_size)
        x_batch = [train_label["Target"].to_numpy(dtype = np.int32), train_label["Context"].to_numpy(dtype = np.int32)]
        y_batch = train_label["Label"].to_numpy(dtype = np.int32)
        return(x_batch, y_batch)

class embeddingIOSequenceValidation():
    def __init__(self, VALIDATION_BATCH_SIZE, CSV_EMBEDDING_FILE):
        self.validation_batch_size = VALIDATION_BATCH_SIZE
        self.csv_path = CSV_EMBEDDING_FILE
        _, file_extension = os.path.splitext(CSV_EMBEDDING_FILE)
        if(file_extension != ".csv"):
            raise Exception("[+]  Incorrect Validation Set File. Please choose correct Validation Embedding CSV.")

    def length(self):
        with open(self.csv_path, "r") as csv_file:
            for count, _ in enumerate(csv_file):
                pass
            return(math.floor(count + 1) / self.validation_batch_size)
        
    def get_data(self):
        valid_label = pd.read_csv(self.csv_path, header = None, names = ["Target", "Context", "Label"], dtype = np.int32, skiprows = 0, nrows = self.validation_batch_size * self.length())
        x_valid = [valid_label["Target"].to_numpy(dtype = np.int32), valid_label["Context"].to_numpy(dtype = np.int32)]
        y_valid = valid_label["Label"].to_numpy(dtype = np.int32)
        return (x_valid, y_valid)


class embedding_tensorboard_callback(tf.keras.callbacks.Callback):
    def __init__(self, config_projector, tensorboard_dir):
        super(embedding_tensorboard_callback, self).__init__()
        self.config_projector = config_projector
        self.tensorboard_dir = tensorboard_dir
    
    def on_epoch_end(self, epoch, logs):
        wt_mat = tf.Variable(self.model.get_layer("target_embedding").get_weights()[0][1:], dtype = tf.float64)
        train_checkpt = tf.train.Checkpoint(embedding = wt_mat)
        train_checkpt.save(os.path.join(self.tensorboard_dir, "embedding.ckpt"))
        projector.visualize_embeddings(self.tensorboard_dir, self.config_projector)