import string


def preprocess_remove(list_of_texts):
    string_digits_ord = [ord(ele) for ele in string.digits]
    string_lowercase_ord = [ord(ele) for ele in string.ascii_lowercase]
    string_punctuation_ord = [ord(ele) for ele in string.punctuation]
    string_ord = string_digits_ord + string_lowercase_ord + string_punctuation_ord

    modified_list_text = []

    for text in list_of_texts:

        modi = []
        for char in text:
            if(ord(char) in string_ord):
                modi.append(char)
            else:
                continue
        temp_password = "".join(char for char in modi)
        if(temp_password == ""):
            continue   
        modified_list_text.append(temp_password)
    return modified_list_text

pass_list = []

with open("/home/rm/BE_Project/Embedding/Data/ascii_rockyou_less_than_thirty_two.txt", "r") as f:
    while True:
        temp = f.readline() 
        if(temp == ""):
            break
        else:
            pass_list.append(temp)



def write_cleaned(pass_list, path):
    cleaned_list = preprocess_remove(pass_list)
    count = 0
    print(len(cleaned_list))
    with open(path, "w") as path_file:
        for password in cleaned_list:
            path_file.write(password+"\n")
    
    with open(path, "r") as f:
        while True:
            temp = f.readline() 
            if(temp == ""):
                break
            else:
                count += 1
    print(count)
    

write_cleaned(pass_list, "/home/rm/BE_Project/Embedding/Data/ascii_rockyou_less_than_thirty_two_cleaned.txt")



# def unique_chars(FILE_PATH):
#     uniq_ch = []
#     with open(FILE_PATH, "r") as f:
#         while True:
#             single_line = f.readline()
#             if(single_line == ""):
#                 break
#             else:
#                 uniq_ch.extend(list(set(single_line.lower())))
#     return list(set(uniq_ch))


# print(unique_chars("/home/rm/BE_Project/Embedding/ASCII_rockyou.txt"))
