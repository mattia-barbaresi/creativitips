# ISL and Creativity - A production-oriented model
## Segmentation, Chunking and creative generations

____________________________

### Main modules (in creativitips.CTPs):
- **pparser.py** (implementation of PARSER)
- **tps.py** (modul for tracking Transitional Probabilities)
- **computation.py** (for computing sequences iteratively: using previous modules)


### Entry points:
- **parser_main.py**: for experiments using PARSER only
- **tps_main.py**: for experimentsd combining TPs and PARSER mehcanisms

____________________________

### Issues: 
- segmentation: TPS (forward, backward, MI)??
- segmentation: with or without chunking?
- chunking: with or without TPS as cues? what about using MDL instead?
- abstraction/generalization: (algebraic patterns?)
- creative generation, using **Simonton's formula**: how to make it converge faster?


### What's next?
- Replay of experiences
- Retrieval
- state uncertainty
- Inference and error-driven learning
- Multi-modal integration


--------------------------------

#### Local dist steps:
build the package:

`pip install .`

`python setup.py sdist`

Then (inside new project's venv):

`pip install [FULL_PATH_TO_DIST]/creativitips-[VERS].tar.gz`


