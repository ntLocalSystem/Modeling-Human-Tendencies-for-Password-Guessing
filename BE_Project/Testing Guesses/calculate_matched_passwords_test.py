import time

def create_list_passwords(INPUT_FILE_NAME):
    pwds = []
    with open(INPUT_FILE_NAME, "r") as pwd_file:
        for line in pwd_file:
            if(line != ""):
                pwds.append(line.split("\t")[0].rstrip())
    return set(pwds)

def calculate_num_matched_passwords(DATASET_FILE_NAME, GENERATED_FILE_NAME):
    start_time = time.time()
    dataset_pwds = create_list_passwords(DATASET_FILE_NAME)
    generated_pwds = create_list_passwords(GENERATED_FILE_NAME)
    num_matched_uniq = len(list(dataset_pwds & generated_pwds))
    done = time.time()
    elapsed = done - start_time
    print(f"Total matched passwords are: {num_matched_uniq}.")
    print(f"Total time elapsed is {elapsed} seconds.")

if(__name__ == "__main__"):
    calculate_num_matched_passwords(".\\password_guesses_lstm.txt", ".\\password_guesses_lstm.txt")