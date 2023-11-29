import time

generated_pwd_file = ".\\password_guesses_lstm_testing_original.txt"
post_processed_pwd_file = ".\\captialization_post_processed_original.txt"
extracted_pwd_file = ".\\extracted_pwd_file_original.txt"

def get_length(generated_pwd_file):
    start_time = time.time()
    with open(generated_pwd_file, "r") as gen_file:
        for count, pwd in enumerate(gen_file):
            pass
    end_time = time.time()
    print(f"The total number of passwords are {count+1}")
    print(f"The total time required to count was {end_time - start_time} seconds.")


def generate_post_processed_pwd(generated_pwd_file, post_processed_pwd_file):
    start_time = time.time()
    with open(generated_pwd_file, "r") as gen_file:
        with open(post_processed_pwd_file, "w") as post_file:
            with open(extracted_pwd_file, "w") as extracted_file:
                for total_processed_pwds, line in enumerate(gen_file):
                    if(total_processed_pwds % 1e5 == 0):
                        print(f"\rTotal passwords processed are : {total_processed_pwds * 1e5}", end = "")
                    pwd = line.split("\t")[0].rstrip()
                    extracted_file.write(pwd + "\n")
                    for index, char in enumerate(pwd):
                        if(char.isalpha()):
                            new_password = pwd[:index] + char.upper() + pwd[(index+1):]
                            post_file.write(new_password+"\n") # Write the new password
    end_time = time.time()
    print(f"\nThe total time required to post-process was {end_time - start_time} seconds.")

if __name__ == "__main__":
    get_length(generated_pwd_file)
    generate_post_processed_pwd(generated_pwd_file, post_processed_pwd_file)