from ccg_nlpy import remote_pipeline

pipeline = remote_pipeline.RemotePipeline()
doc = pipeline.doc("the naughty dog chased the cat alone")
sp = " || ".join(x["tokens"] for x in doc.get_shallow_parse.cons_list)
print(sp)

# from ccg_nlpy import local_pipeline
# pipeline = local_pipeline.LocalPipeline()
#
# document = [ ["Hi", "!"], ["How", "are", "you", "?"] ]
# doc = pipeline.doc(document, pretokenized=True)
