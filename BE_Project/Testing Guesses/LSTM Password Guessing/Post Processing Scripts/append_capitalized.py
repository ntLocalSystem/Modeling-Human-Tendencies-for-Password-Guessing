import time
import os

original_file = os.path.join("..", "Password Guess Files", "lstm_pwd_best64_cleaned.txt")
processed_file = os.path.join("..", "Password Guess Files", "captialization_post_processed_original.txt")

def append_processed_to_original(original_file, processed_file):
    start_time = time.time()
    with open(original_file, "a") as orig:
        with open(processed_file, "r") as proc:
            for pwd in proc:
                if(pwd == ""):
                    continue
                else:
                    orig.write(pwd)
    end_time = time.time()
    print(f"Total time required was {int(end_time - start_time)}")


if __name__ == "__main__":
    append_processed_to_original(original_file, processed_file)


