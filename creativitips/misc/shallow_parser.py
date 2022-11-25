import os
from ccg_nlpy import local_pipeline

pipeline = local_pipeline.LocalPipeline()
root_dir = '../data/CHILDES_converted/'

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if '.capp' in file:
            filenm = file
            textfile = subdir + '/' + file
            outdir = subdir.replace("data", "data/out")
            os.makedirs(outdir, exist_ok=True)

            print("processing file: ", filenm)
            with open(textfile, "r",encoding='cp1252') as fpi:
                with open(outdir + '/' + filenm.split('.capp')[0] + '.shpar', "w") as fpo:
                    for line in fpi.readlines():
                        utter = line.split()
                        # if list has less than 3 elements, it is empty b/c
                        # auto-cleaning removed a non-speech sound, etc.
                        if len(utter) < 3:
                            continue
                        doc = pipeline.doc(" ".join(utter[1:]), pretokenized=False)
                        tks = " || ".join(x["tokens"] for x in doc.get_shallow_parse.cons_list)
                        fpo.write(tks + "\n")

# doc = pipeline.doc("The dog chase the cat", pretokenized=False)
# sp = list(x["tokens"] for x in doc.get_shallow_parse.cons_list)
# print(sp)

# from ccg_nlpy import remote_pipeline
#
# pipeline = remote_pipeline.RemotePipeline()
# doc = pipeline.doc("I am gonna stop the train with my whist")
# sp = " || ".join(x["tokens"] for x in doc.get_shallow_parse.cons_list)
# print(sp)
