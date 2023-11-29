
'''
Converts the Unicode to ASCII 
Ignores the Errors during conversion.

'''

piece_size = 4096 

'''

Reads 4096 bytes and then decodes as ASCII

'''

with open("/home/rm/BE_Project/rockyou.txt", "rb") as orig:
    with open("/home/rm/BE_Project/Embedding/ASCII_rockyou.txt", "w") as modi:
        while True:
            piece_data = orig.read(piece_size)
            if(piece_data == b""):
                break
            modi.write(piece_data.decode("ascii", "ignore"))

