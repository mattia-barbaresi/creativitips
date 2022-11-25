from ccg_nlpy import local_pipeline
pipeline = local_pipeline.LocalPipeline()

file_in = "../data/br_text.txt"
file_out = "../data/out/br_text.txt"

# with open(file_in, "r") as fpi:
#     with open(file_out, "w") as fpo:
#         for line in fpi.readlines():
#             doc = pipeline.doc(line, pretokenized=False)
#             # tks = list(x["tokens"] for x in doc.get_shallow_parse.cons_list)
#             tks = " || ".join(x["tokens"] for x in doc.get_shallow_parse.cons_list)
#             fpo.write(tks + "\n")


# doc = pipeline.doc("The dog chase the cat", pretokenized=False)
# sp = list(x["tokens"] for x in doc.get_shallow_parse.cons_list)
# print(sp)

from ccg_nlpy import remote_pipeline

pipeline = remote_pipeline.RemotePipeline()
doc = pipeline.doc("I am gonna stop the train with my whist")
sp = " || ".join(x["tokens"] for x in doc.get_shallow_parse.cons_list)
print(sp)

