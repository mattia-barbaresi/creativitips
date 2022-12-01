#
# ILLINOIS SHALLOW PARSER
#
import os
from ccg_nlpy import local_pipeline
from ccg_nlpy import remote_pipeline

def run():
    pipeline = local_pipeline.LocalPipeline()
    root_dir = '../data/CHILDES_converted/'

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if '.capp' in file:
                filenm = file
                textfile = subdir + '/' + file
                outdir = subdir.replace("CHILDES_converted", "CHILDES_ISP")
                os.makedirs(outdir, exist_ok=True)

                print("processing file: ", textfile)
                with open(textfile, "r", encoding='utf-8') as fpi:
                    with open(outdir + '/' + filenm.split('.capp')[0] + '.shpar', "w") as fpo:
                        for line in fpi.readlines():
                            if '*AGEIS:' in line:
                                continue
                            utter = line.strip().split()
                            # if list has less than 3 elements, it is empty b/c
                            # auto-cleaning removed a non-speech sound, etc.
                            if len(utter) < 3:
                                continue
                            doc = pipeline.doc(" ".join(utter[1:]), pretokenized=False)
                            tks = " || ".join(x["tokens"] for x in doc.get_shallow_parse.cons_list)
                            fpo.write(tks + "\n")


def test():
    pipeline_remote = remote_pipeline.RemotePipeline()
    pipeline_local = local_pipeline.LocalPipeline()
    input_str = "youve not played with this ."
    doc_r = pipeline_remote.doc(input_str, pretokenized=False)
    doc_l = pipeline_local.doc(input_str, pretokenized=False)
    spr = " || ".join(x["tokens"] for x in doc_r.get_shallow_parse.cons_list)
    spl = " || ".join(x["tokens"] for x in doc_l.get_shallow_parse.cons_list)
    print("spl:", spl)
    print("spr: ", spr)


if __name__ == "__main__":
    run()
