'''

Script segregates the passwords based on length:
0-8 : ascii_rockyou_one_to_eight.txt
8-16 : ascii_rockyou_eight_to_sixteen.txt
16-32 : ascii_rockyou_sixteen_to_thirty_two.txt
0-32 : ascii_rockyou_less_than_thirty_two.txt

'''
max_length = 0

with open("ASCII_rockyou_removed.txt", "r") as orig_modi:
    with open("ascii_rockyou_one_to_eight.txt", "w") as zero_to_eight:
        with open("ascii_rockyou_eight_to_sixteen.txt", "w") as eigth_to_sixteen:
            with open("ascii_rockyou_sixteen_to_thirty_two.txt", "w") as sixteen_to_thirty_two:
                with open("ascii_rockyou_greater_than_thirty_two.txt", "w") as greater_than_thirty_two:
                    with open("ascii_rockyou_less_than_thirty_two.txt", "w") as less_than_thirty_two: 
                        for count, password in enumerate(orig_modi):
                            temp_str = password.rstrip("\n")
                            if(temp_str == ""):
                                break
                            if(len(temp_str) > max_length):
                                max_length = len(temp_str)
                            if(len(temp_str) <= 32):
                                less_than_thirty_two.write(temp_str + "\n")
                            if(len(temp_str) > max_length):
                                max_length = len(temp_str)
                            if(len(temp_str) <= 8):
                                zero_to_eight.write(temp_str + "\n")
                            elif(len(temp_str) <= 16):
                                eigth_to_sixteen.write(temp_str + "\n")
                            elif(len(temp_str) <= 32):
                                sixteen_to_thirty_two.write(temp_str + "\n")
                            else:
                                greater_than_thirty_two.write(temp_str + "\n")


print("The maximum length of password in rockyou.txt (ASCII) is : ", max_length)

less_than_thirty_two_max_length = 0

with open("ascii_rockyou_less_than_thirty_two.txt", "r") as greater_than_thirty_two_next:
    for count, password in enumerate(greater_than_thirty_two_next):
        temp_str = password.rstrip("\n")
        if(len(temp_str) > less_than_thirty_two_max_length):
            less_than_thirty_two_max_length = len(temp_str)


print(less_than_thirty_two_max_length)