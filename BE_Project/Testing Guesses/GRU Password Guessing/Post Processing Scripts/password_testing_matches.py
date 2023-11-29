import pandas as pd
import numpy as np
import time
import os

'''
Make a csv file showing all the matches
between passwords obtained via sampling 
from the neural network and from the 
dataset.
The file will be stored in csv format.
This can be later used for plotting
in matplotlib.
'''


generated_file = os.path.join("..", "Password Guess Files", "gru_pwd_best64_cleaned.txt")
original_file = os.path.join("P:\\BE_Project", "Testing Guesses", "Testing Data", "rockyou_test_5_mil.txt")
csv_file_name = ".\\result_test.csv"

make_data_point_threshold = 10000
# 10000 will result in 10^5 + 1 data points for 1 Billion passwords

match_dict = {"Guesses": [], "one_to_eight":[], "nine_to_fifteen": [], "sixteen_to_thirtytwo": []}

list_pwds = []
matched_unique_set = set()

with open(original_file, "r") as og_file:
    # Loading the original file
    print("[+] Loading the original file...")
    start_time = time.time()
    for count, pwd in enumerate(og_file):
        list_pwds.append(pwd.rstrip().lstrip().lower())
    end_time = time.time()
    print(f"[+] Total passwords loaded {count + 1}.")
    print(f"[+] Total time required to load was {int(end_time - start_time)} seconds.")

print(f"[+] Total passwords in testing set: {len(list_pwds)}")
list_pwds = set(list_pwds)
print(f"[+] Total unique passwords in testing set: {len(list_pwds)}")

def create_data_point(temp_list):
    global matched_unique_set

    temp_list = set(temp_list)

    matched_set = (temp_list.intersection(list_pwds))
    # Takes a lot of time
    matched_unique = matched_set.difference(matched_unique_set)

    # Add matched passwords
    matched_unique_set = matched_unique_set.union(matched_unique)
    matched_list = list(matched_unique) + [pwd for pwd in list(matched_set) if (len(pwd) >= 16)]
    return matched_list



with open(generated_file, "r") as gen_file:
    # Loading the generated file line by line
    outer_count = 1
    count_one_to_eight = 0
    count_nine_to_fifteen = 0
    count_sixteen_to_thirtytwo = 0
    total_pwds_tried = 0
    print("[+] Checking the generated file..")
    start_time = time.time()
    temp_list = []
    for count, pwd in enumerate(gen_file):
        # Strip the newline character
        pwd = pwd.rstrip()
        temp_list.append(pwd)
        if((count + 1) % 10000 == 0):
            print(f"\r[+] Total {count + 1} passwords tried.", end = "")    
        if(len(temp_list) == make_data_point_threshold):
            total_pwds_tried += make_data_point_threshold
            matched_list = create_data_point(temp_list)
            for matched_pwd in matched_list:
                matched_pwd_len = len(matched_pwd)
                if(matched_pwd_len <= 8):
                    count_one_to_eight += 1
                elif(matched_pwd_len >= 9 and matched_pwd_len <=15):
                    count_nine_to_fifteen += 1
                else:
                    count_sixteen_to_thirtytwo += 1
            # Now create a datapoint
            match_dict["Guesses"].append(total_pwds_tried)
            match_dict["one_to_eight"].append(count_one_to_eight)
            match_dict["nine_to_fifteen"].append(count_nine_to_fifteen)
            match_dict["sixteen_to_thirtytwo"].append(count_sixteen_to_thirtytwo)
            temp_list = []
    if(len(temp_list) > 0):
        total_pwds_tried += len(temp_list)
        matched_list = create_data_point(temp_list)
        for matched_pwd in matched_list:
            matched_pwd_len = len(matched_pwd)
            if(matched_pwd_len <= 8):
                count_one_to_eight += 1
            elif(matched_pwd_len >= 9 and matched_pwd_len <=15):
                count_nine_to_fifteen += 1
            else:
                count_sixteen_to_thirtytwo += 1
        match_dict["Guesses"].append(total_pwds_tried)
        match_dict["one_to_eight"].append(count_one_to_eight)
        match_dict["nine_to_fifteen"].append(count_nine_to_fifteen)
        match_dict["sixteen_to_thirtytwo"].append(count_sixteen_to_thirtytwo)        

    end_time = time.time()
    print(f"\n[+] Total time required to match was {int(end_time - start_time)} seconds.")

# Creating a dataframe:
print(f"[+] Now creating a dataframe...")
df = pd.DataFrame(data = match_dict)

# Save the df as csv
print(f"[+] Saving the dataframe in csv format...")
df.to_csv(csv_file_name, header = True, index = False, index_label = False)

# Done.
print(f"Done!")










