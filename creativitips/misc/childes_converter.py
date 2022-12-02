import os
import string
import re
import const

rootdir = const.CHILDES_REPO
outdir = "../data/CHILDES_converted/"

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
                    # get rid of special chars!!
                    # if lpt != "." or lpt != "?":
                    #     print("sdsdsfds: ", lpt)
                    line = line.replace("	"," ").strip()
                    if line[-1] == "?":
                        last = "?"
                    elif line[-1] == "!":
                        last = "!"
                    else:
                        last = "."
                    ss = " ".join(re.sub(r'\[[^()]*\]', '', line).split())
                    ss = " ".join(re.sub(r'\<[^()]*\>', '', ss).split())
                    ss = " ".join(re.sub(r'\[^()]*\', '', ss).split())

                    ss = ss.replace("„","").replace("+...","").replace("‡","").replace("_"," ") \
                        .replace("xxx","").replace("+//","").replace("+/","")
                    first = ss.split()[0]
                    ss = ss.translate(str.maketrans('', '', string.punctuation+"”“")).strip().split()
                    nl = " ".join(ss[1:]).replace("  ", " ")
                    if len(nl.strip()) > 0:
                        new_lines.append(" ".join([first, nl, last]) + "\n")

            if new_lines:
                vals = "/".join(subdir.split("/")[-1].split("\\"))
                fout = outdir + vals
                os.makedirs(fout, exist_ok=True)
                with open(fout + "/" + new_file,"w",  encoding='utf-8') as fp:
                    fp.writelines(new_lines)
