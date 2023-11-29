
# author : Rohit Mutalik
# date : 5-2-2021 (rev 1)
# date : 9-2-2021 (rev 1.1) (Added Length Normalization)
# date : 10-2-2021(rev 1.2) (Added Unlimited Generation and Exit Handling)

# This script is for sampling novel sequences
# with probability greater than threshold.
# This is only applicable to GRU Networks.

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import os
import json
import atexit
import math

class GuessCompletedError(Exception):
    def __init__(self, message):
        super(GuessCompletedError, self).__init__(message)
        if message:
            self.message = message
        else:
            self.message = None
    def __str__(self):
        if self.message:
            return self.message
        else:
            return "GuessCompletedError has been raised"

class Guesser():
    def __init__(
        self, INFERENCE_MODEL, OP_FILE,
        VOCAB_SIZE, GRU_UNITS, PREDICTION_BATCH_SIZE,
        RECUR_BATCH_SIZE, TOKENIZER, END_IDX, NUM_GUESS,
        START="", PROB=1, MAX_LENGTH=32,
        PROB_THRESHOLD=1e-6, PASSWORD_END_CHAR="\n", CALCULATE_PROB = False,
        USE_LENGTH_NORMALIZATION = False, NORMALIZED_SCORE_THRESHOLD = -1.72693,
        NORMALIZATION_ALPHA = 0.7, UNLIMITED_GENERATION = False
        ):
        # Inference Model (ONE_TO_ONE) Shared Layer Architecture
        self.model = INFERENCE_MODEL

        # File Handle for the file
        self.file_handle = open(OP_FILE, "w")
        self.pwd_generated = 0
        self.num_to_guess = NUM_GUESS

        # Batch Sizes for prediction and recursive calls
        self.gpu_prediction_batch_size = PREDICTION_BATCH_SIZE
        self.recur_batch_size = RECUR_BATCH_SIZE

        # Utility -
        self.tokenizer = TOKENIZER
        self.end_idx = END_IDX
        self.end_char = PASSWORD_END_CHAR
        self.vocab_size = VOCAB_SIZE
        self.gru_units = GRU_UNITS

        # Starting Configurations for guessing
        self.start = START
        self.start_prob = PROB
        self.use_calculated_prob = CALCULATE_PROB

        # Normalization Parameters
        self.use_length_normalization = USE_LENGTH_NORMALIZATION
        self.length_normalization_threshold = NORMALIZED_SCORE_THRESHOLD
        self.normalization_alpha = NORMALIZATION_ALPHA

        # Max Length for Passwords
        self.max_length = MAX_LENGTH

        # Probability Threshold for guessing
        self.prob_threshold = PROB_THRESHOLD

        # Initializing the initial states of the inference network
        # All are 0s
        self.gru_c_1 = np.zeros((1, self.gru_units))
        self.gru_c_2 = np.zeros((1, self.gru_units))
        self.gru_c_3 = np.zeros((1, self.gru_units))

        # Building the Initial Data Structure to be supplied with the prefix
        self.start_state = {
            "1": (self.gru_c_1,),
            "2": (self.gru_c_2,),
            "3": (self.gru_c_3,)
            }

        # Generation Type:
        self.generation_type_unlimited = UNLIMITED_GENERATION

        # Start Guessing:
        self.complete_guessing(self.start, self.start_prob)

    def finish_guessing(self):
        # Empty the buffer and close the file handle
        # Cleanup
        self.file_handle.flush()
        self.file_handle.close()

    def exit_handler(self):
        print(f"Total passwords generated are: {self.pwd_generated}")
        print("Cleaning up and saving the file...")
        self.finish_guessing()
        print("Done!")
        print("Exiting..")
        exit()

    def complete_guessing(self, start="", start_prob=1.0):
        print(f"[+] Enumerating guesses starting from {start}")
        # Start Guessing
        try:
            self.guess(start, start_prob)
            self.finish_guessing()
            print(f"[+] Guessed {self.pwd_generated} passwords.")
        except GuessCompletedError:
            print(f"[+] Guessed {self.num_to_guess} passwords.")
            # Cleanup
            exit()
        except KeyboardInterrupt:
            self.exit_handler()

    def guess_check(self):
        if(self.num_to_guess == self.pwd_generated):
            self.finish_guessing()
            raise GuessCompletedError("Guessing Completed!")

    def guess(self, start="", start_prob=1.0):
        # Call the function that defines the data structure
        self._recur(start, start_prob)

    def super_node_recur(self, node_list):
        # Stop Condition:
        if(len(node_list) == 0):
            # Return to parent
            return

        # None of the prefix passwords should be empty -
        prefix_passwords = self._extract_pwd_from_nodelist(node_list)

        # Find the probabilities for all the nodes in the list
        predictions = self.batch_probs(node_list)

        # Specify a batch for next recursive call
        node_batch = []

        for count, current_node in enumerate(node_list):
            pwd_str, pwd_prob, states = current_node
            for next_node in self.next_nodes(pwd_str, pwd_prob, predictions[count]):
                node_batch.append(next_node)
                if(len(node_batch) == self.recur_batch_size):
                    self.super_node_recur(node_batch)
                    node_batch = []
        if(len(node_batch) > 0):
            self.super_node_recur(node_batch)
            node_batch = []

    def _extract_pwd_from_nodelist(self, nodelist):
        passwd_list = list(map(lambda x: x[0], nodelist))
        return passwd_list

    def batch_probs(self, node_list):
        # Divides the node list into batches
        # Calls the function making computations
        # Parses the return values / answers
        answer = []
        if(len(node_list) > self.gpu_prediction_batch_size):
            for chunk in range(
                math.ceil(len(node_list) / self.gpu_prediction_batch_size)
            ):
                predictions = self.calculate_conditional_probs(
                    node_list[chunk * self.gpu_prediction_batch_size:
                    (chunk+1) * self.gpu_prediction_batch_size]
                )

                answer = answer + predictions
        else:
            answer = self.calculate_conditional_probs(node_list)
        return answer

    def next_nodes(self, pwd_str, pwd_prob, predictions):
        assert pwd_str == predictions[0]
        total_prob = pwd_prob * predictions[1]

        # This function calculates the probability as well as the
        # Length Normalized Score:
        # The formula for length normalized score is:
        # score = (1 / Ty^α) * arg max (y) Σ log( P ( y<t> | x, y<1>, y<2>, ... , y<t-1> ) )
        if ((len(pwd_str) + 1) > (self.max_length + 1)):
            prob_end = total_prob[self.end_idx]
            if(self.use_length_normalization == False):
                if(prob_end >= self.prob_threshold):
                    self.output_serializer(pwd_str, prob_end)
                    self.pwd_generated += 1
                    if(self.generation_type_unlimited == False):
                        self.guess_check()
                return []
            else:
                prob_end = prob_end + 1e-15
                length_normalized_score = np.log(prob_end) / \
                    np.power(len(pwd_str), self.normalization_alpha)
                if(length_normalized_score >= self.length_normalization_threshold):
                    self.output_serializer(pwd_str, prob_end, length_normalized_score)
                    self.pwd_generated += 1
                    if(self.generation_type_unlimited == False):
                        self.guess_check()
                return[]

        # For passwords that have length below max length

        indexes = np.arange(self.vocab_size + 1)

        # Not using length normalized beam search
        if(self.use_length_normalization == False):
            above_cutoff = total_prob >= self.prob_threshold
            above_indices = indexes[above_cutoff]
            probs_above = total_prob[above_cutoff]
            answer = []
            for i, chain_prob in enumerate(probs_above):
                if(above_indices[i] == 0): # <PAD> Token
                    continue
                if(above_indices[i] == 1): # \t Token (START)
                    continue
                char = self.tokenizer.sequences_to_texts([[above_indices[i]]])[0][-1]
                if char == self.end_char:
                    if(pwd_str != "\t"): # Don't write only \t (START CHAR)
                        self.output_serializer(pwd_str, chain_prob)
                        self.pwd_generated += 1
                        if(self.generation_type_unlimited == False):
                            self.guess_check()
                else:
                    chain_pass = pwd_str + char
                    answer.append((chain_pass, chain_prob, predictions[2]))
        else: # Using the length normalized beam search
            # Tokens that are assigned 0.00 probability
            # cause numerical instability in calculating np.log(prob)
            # To avoid this, we add a small constant 1e-15 to the
            # Probability Distribution.
            total_prob = total_prob + 1e-15
            length_normalized_scores = np.log(total_prob) / \
                np.power(len(pwd_str), self.normalization_alpha)
            above_cutoff = length_normalized_scores >= self.length_normalization_threshold
            above_indices = indexes[above_cutoff]
            scores_above = length_normalized_scores[above_cutoff]
            probs_above = total_prob[above_cutoff]
            answer = []
            for i, chain_score in enumerate(scores_above):
                if(above_indices[i] == 0): # <PAD> Token
                    continue
                if(above_indices[i] == 1): # \t Token (START)
                    continue
                char = self.tokenizer.sequences_to_texts([[above_indices[i]]])[0][-1]
                if(char == self.end_char):
                    if(pwd_str != "\t"): # Don't write only \t (START CHAR)
                        self.output_serializer(pwd_str, probs_above[i], chain_score)
                        self.pwd_generated += 1
                        if(self.generation_type_unlimited == False):
                            self.guess_check()
                else:
                    chain_pass = pwd_str + char
                    answer.append((chain_pass, probs_above[i], predictions[2]))
        return answer

    def output_serializer(self, pwd_str, pwd_prob, length_normalized_score = None):
        pwd_stripped = pwd_str.lstrip("\t")
        pwd_justified_str = pwd_stripped.ljust(32)
        if(self.use_length_normalization == False):
            self.file_handle.write(f"{pwd_justified_str}\t{pwd_prob}\n")
        else:
            self.file_handle.write(f"{pwd_justified_str}\t{pwd_prob}\t{length_normalized_score}\n")


    def calculate_conditional_probs(self, subset_node_list):
        # encoded_prev_char = np.zeros((0, 1))
        pwd_list = self._extract_pwd_from_nodelist(subset_node_list)
        tokenized_passwords = self.tokenizer.texts_to_sequences(pwd_list)
        tokenized_last_chars = list(map(lambda x: x[-1], tokenized_passwords))

        assert len(pwd_list) == len(tokenized_last_chars)

        # This is the prev timestep generated input for the batch (encoded)
        encoded_batch_prev_chars = np.array(tokenized_last_chars).reshape(-1, 1)

        # Placeholders for states - hidden and cell for all three layers:
        gru_1_c_states = np.zeros((0, self.gru_units))
        gru_2_c_states = np.zeros((0, self.gru_units))
        gru_3_c_states = np.zeros((0, self.gru_units))


        # Add the states to the placeholders :
        for node in subset_node_list:
            gru_1_c_states = np.concatenate((gru_1_c_states, node[2]["1"][0]), 0)

            gru_2_c_states = np.concatenate((gru_2_c_states, node[2]["2"][0]), 0)

            gru_3_c_states = np.concatenate((gru_3_c_states, node[2]["3"][0]), 0)

        # Add Batch Size here - gpu max size

        predictions = self.model.predict(
            [encoded_batch_prev_chars, gru_1_c_states,
            gru_2_c_states, gru_3_c_states],
            batch_size=self.gpu_prediction_batch_size,
            verbose=0)

        # predictions is a list
        # [
        # Conditional Probability Distribution Over the token space - (chunk, 1, vocab_size)
        # GRU Cell State Layer 1 - (self.gpu_prediction_batch_size, 300)
        # GRU Cell State Layer 2 - (self.gpu_prediction_batch_size, 300)
        # GRU Cell State Layer 3 - (self.gpu_prediction_batch_size, 300)
        # ]

        prob_distribution = np.zeros((0, (self.vocab_size + 1))) # Includes the <PAD> token too.
        next_states = []

        for i in range(len(pwd_list)): # Total number of passwords == total number of output prob distributions
            prob_distribution = np.concatenate((prob_distribution, predictions[0][i]), axis=0)
            next_states.append(
                {
                    "1": (predictions[1][i].reshape(1, self.gru_units),),
                    "2": (predictions[2][i].reshape(1, self.gru_units),),
                    "3": (predictions[3][i].reshape(1, self.gru_units),)
                }
            )

        assert prob_distribution.shape[0] == len(next_states)

        # Creating a list consisting of elements of structure - [..., (pwd, prob_dist, next_state) , ...]

        answer = []

        for i in range(len(pwd_list)):
            answer.append(tuple([pwd_list[i], prob_distribution[i], next_states[i]]))


        # answer is a list containing
        # tuples of -
        # pwd, prob_dist, next_states (both hidden and cell)


        return answer

    def _recur(self, start_string="", start_prob=1.0):
        if start_string == "":
            initial_string = "\t" + start_string
            initial_prob = start_prob
            # Define the first node
            initial_node = (initial_string, initial_prob, self.start_state)
        else:
            compute_string = "\t" + start_string
            computed_state, computed_prob = self.compute_states_and_prob(compute_string)
            # Define the first node
            if(self.use_calculated_prob):
                initial_node = (compute_string, computed_prob, computed_state)
            else:
                initial_node = (compute_string, 1.0, computed_state)

        # Define the node list
        node_list = [(initial_node)]

        # Call the super recursive function
        self.super_node_recur(node_list)

    def compute_states_and_prob(self, compute_string):
        state = self.start_state
        prob = 1.0
        for index, char in enumerate(compute_string):
            tokenized_char = self.tokenizer.texts_to_sequences([char])[0][0]
            timestep_input = np.array(tokenized_char).reshape(1, 1)
            prediction = self.model.predict(
                [timestep_input,
                state["1"][0], state["2"][0],
                state["3"][0]], verbose = 0
            )
            state = {
                "1" : (prediction[1][0].reshape(1, self.gru_units),),
                "2" : (prediction[2][0].reshape(1, self.gru_units),),
                "3" : (prediction[3][0].reshape(1, self.gru_units),)
            }
            prob_distribution = prediction[0][0][0] # shape => (71,)
            next_char = compute_string[index + 1]
            tokenized_next_char = self.tokenizer.texts_to_sequences([char])[0][0]
            prob = prob * prob_distribution[tokenized_next_char]
            if(index == (len(compute_string) - 2)):
                break
                # The condition to break the loop
        return(state, prob)

def loadTokenizer(TOKENIZER_FILE_PATH):
    _, file_extension = os.path.splitext(TOKENIZER_FILE_PATH)
    if(file_extension != ".json"):
        raise Exception("Incorrect File.")
        return
    else:
        with open(TOKENIZER_FILE_PATH, "r") as tokenizer_cfg_file:
            tokenizer_config = tokenizer_cfg_file.read()
            tokenizer_cfg = json.loads(tokenizer_config)
            passwordTokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_cfg))
            return passwordTokenizer

if __name__ == "__main__":
    inference_model = tf.keras.models.load_model("H5-Inference-Checkpoint-019.h5")
    tokenizer = loadTokenizer("prototypeTokenizer.json")
    pwd_guesser = Guesser(inference_model, "password_guesses.txt", 70,
                        300, 32, 32,
                        tokenizer, 2, 15, "",
                        1.0, 32, 1e-9, "\n",
                        False, False, -4.67, 0.7, False)