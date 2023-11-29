FILE_PATH = "/home/rm/BE_Project/Embedding/Data/ascii_rockyou_less_than_thirty_two_cleaned.txt"
max_length = 0

with open(FILE_PATH, "r") as f:
    for count, password in enumerate(f):
        temp = password.rstrip("\n")
        if(len(temp) > max_length):
            max_length = len(temp)


print(f"The maximum length in given file is : {max_length}")
