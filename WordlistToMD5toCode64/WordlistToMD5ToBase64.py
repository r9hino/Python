#!/usr/bin/python3

import sys
import os
import base64
import hashlib


def main():
    wordlistpath = sys.argv[1]
    outputpath = sys.argv[2]

    if not os.path.isfile(wordlistpath):
        print("File path {} does not exist. Exiting...".format(wordlistpath))
        sys.exit()

    # Using readlines()
    wordlistfile = open(wordlistpath, 'r')
    lines = wordlistfile.readlines()

    # Writing to a file 
    outputfile = open(outputpath, 'w')


    # Strips the newline character
    for line in lines:
        lineMD5 = hashlib.md5(line.strip().encode("UTF-8")).hexdigest()
        userPassword = ('admin:' + lineMD5).encode("UTF-8")
        userPasswordCode64 = str(base64.b64encode(userPassword), 'utf-8')
        userPasswordCode64 = userPasswordCode64.replace('=', '%3D')
        print('admin:' + line.strip() + '  ->  ' + str(userPassword,'utf-8') + '  ->  ' + userPasswordCode64)
        
        outputfile.writelines(userPasswordCode64 + '\n')

    wordlistfile.close()
    outputfile.close()

if __name__ == '__main__':
    main()