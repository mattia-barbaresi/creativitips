import os
from ccg_nlpy import local_pipeline

pipeline = local_pipeline.LocalPipeline()
rootdir = '../data/CHILDES_converted/'

for subdir, dirs, files in os.walk(rootdir):
    if "Edinburgh" in subdir:
        continue
    for file in files:
        if ('.capp' in file):
            filenm = file
            textfile = subdir + '/' + file
            outdir = subdir.replace("data", "data/out")
            os.makedirs(outdir, exist_ok=True)

            with open(textfile, "r") as fpi:
                with open(outdir + '/' + filenm.split('.capp')[0] + '.shpar', "w") as fpo:
                    for line in fpi.readlines():
                        doc = pipeline.doc(line, pretokenized=False)
                        # tks = list(x["tokens"] for x in doc.get_shallow_parse.cons_list)
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

