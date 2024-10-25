***
# Recovering from Poisoning through Active Learning (RPAL) Framework

This repository is the official release of the code used for the 'The Impact of Active Learning on Availability Data Poisoning for Android Malware Classifiers' Paper published in the Workshop on Recent Advances in Resilient and Trustworthy Machine Learning (ARTMAN) 2024, co-located with ACSAC.

If you plan to use this repository in your projects, please cite the following paper:

```bibtex
@inproceedings{mcfadden2024recovery,
  title = {The Impact of Active Learning on Availability Data Poisoning for Android Malware Classifiers},
  author = {McFadden, Shae and Zeliang, Kan and Cavallaro, Lorenzo and Pierazzi, Fabio},
  booktitle = {Proc. of the 39th Annual Computer Security Applications Conference ({ACSAC})},
  year = {2024},
}
```
***

### Disclaimer 

Please note that the code in this repository is only a research prototype. This code is released under a "Modified (Non-Commercial) BSD License": see the terms [here](./LICENSE).

***

## Installation
Please note that this project requires tesseract-ml, which can be found [here](https://github.com/s2labres/tesseract-ml-release) and installed as follows.
```bash
pushd ${PATH_TO}/tesseract-ml
python setup.py install (install tesseract)
popd
```
Once tesseract-ml has been installed, RPAL can be setup as follows.
``` bash
pip install NumpyEncoder
cd RPAL;
pip install -r requirements.txt
pip install .
```

***
