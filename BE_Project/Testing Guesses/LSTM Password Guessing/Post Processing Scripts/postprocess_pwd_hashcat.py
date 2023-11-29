import time
import string
import os

file_name = ".\\lstm_post_processed_best64.txt"
ascii_file_name = ".\\lstm_pwd_best64_ascii.txt"
cleaned_file_name = ".\\lstm_pwd_best64_cleaned.txt"

# ASCII conversion parameter
piece_size = 4096

def get_length(file_name):
    print("[+] Finding the total number of passwords..")
    start_time = time.time()
    with open(file_name, "r") as f:
        for count, line in enumerate(f):
            pass
        print(f"The total number of passwords are {count + 1}")
    end_time = time.time()
    print(f"Total time taken to count {end_time - start_time} seconds.")


def remove_other_chars(file_name, cleaned_file_name):
    start_time = time.time()
    print("[+] Removing characters other than alphabets, numbers and special characters...")
    string_digits_ord = [ord(ele) for ele in string.digits]
    string_uppercase_ord = [ord(ele) for ele in string.ascii_uppercase]
    string_lowercase_ord = [ord(ele) for ele in string.ascii_lowercase]
    string_punctuation_ord = [ord(ele) for ele in string.punctuation]
    string_ord = string_digits_ord + string_lowercase_ord + string_punctuation_ord + string_uppercase_ord
    
    with open(file_name, "r") as gen_file:
        with open(cleaned_file_name, "w") as clean_file:
            modi = []
            for count, gen_pwd in enumerate(gen_file):
                gen_pwd = gen_pwd.rstrip()
                if(count % 1e5 == 0):
                    print(f"\r[+] Total passwords processed {count}", end="")
                for each_char in gen_pwd:
                    if(ord(each_char) in string_ord):
                        modi.append(each_char)
                    else:
                        continue
                temp_password = "".join(modi)
                if(temp_password == ""):
                    modi = []
                    continue # Don't write empty password to clean file
                else:
                    clean_file.write(temp_password + "\n") # Write the cleaned password 
                    modi = []
    
    end_time = time.time()
    print(f"\nTotal time required to remove characters other than alphabets, numbers and special characters is {end_time - start_time} seconds.")

def eliminate_unicode_chars(file_name, ascii_file_name):
    print("[+] Eliminating Unicode characters...")
    start_time = time.time()
    outer_count = 0
    count = 0
    with open(file_name, "rb") as gen_file:
        with open(ascii_file_name, "w") as ascii_file:
            while True:
                count += 1
                if(count % 1e5 == 0):
                    outer_count += 1
                    print(f"\r[+] {outer_count * 400} MB of data was processed.", end="")
                    count = 1
                piece_data = gen_file.read(piece_size)
                if(piece_data == b""):
                    break
                data = piece_data.decode("ascii", "ignore").replace("\r", "")
                ascii_file.write(data)
    end_time = time.time()
    print(f"\n[+] Total time required to convert to ASCII was {end_time - start_time} seconds.")


if __name__ == "__main__":
    start_time = time.time()
    eliminate_unicode_chars(file_name, ascii_file_name)
    remove_other_chars(ascii_file_name, cleaned_file_name)
    get_length(cleaned_file_name)
    end_time = time.time()
    print(f"\n\nTotal time required was {end_time - start_time} seconds.")


            


