import os
import string

rootdir = "./data/Anne/"
outdir = "./data/Anne_converted/"

punctuation = string.punctuation
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if '.cha' in file:
            print("file: ", file)
            textfile = subdir + '/' + file
            new_file = file.split(".cha")[0] + ".capp"
            file = open(textfile, 'r', encoding='utf-8')
            lines = file.readlines()
            file.close()

            new_lines = []
            for line in lines:
                if line.startswith("*"):
                    new_lines.append("*"+line.translate(str.maketrans('', '', punctuation)).replace("	"," "))

            if new_lines:
                with open(outdir + new_file,"w") as fp:
                    fp.writelines(new_lines)
