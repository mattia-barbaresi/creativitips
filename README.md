# ISL and Creativity - A production-oriented model
## Segmentation, Chunking, Attention and Creative generations

### Main modules (in creativitips.CTPs):
- **pparser.py** (implementation of PARSER)
- **tps.py** (modul for tracking Transitional Probabilities)
- **computation.py** (for computing sequences iteratively: using previous modules)


### Entry points:
- **parser_main.py**: for experiments using PARSER only
- **tps_main.py**: for experiments combining TPs and PARSER mechanisms

____________________________

### Issues: 
- segmentation 
  - TPS (forward, backward, MI). How to select the right one??
  - With or without chunking?
- chunking
  - with or without TPS as cues? what about using MDL instead?
- abstraction and generalization:
  - algebraic patterns for abstraction?


### What's next?
- Replay of experiences
- Retrieval
- state uncertainty
- Inference and error-driven learning
- Multi-modal integration


[//]: # (--------------------------------)

[//]: # (#### Local dist steps:)

[//]: # (build the package:)

[//]: # ()
[//]: # (`pip install .`)

[//]: # ()
[//]: # (`python setup.py sdist`)

[//]: # ()
[//]: # (Then &#40;inside new project's venv&#41;:)

[//]: # ()
[//]: # (`pip install [FULL_PATH_TO_DIST]/creativitips-[VERS].tar.gz`)


