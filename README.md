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

## Repository Contents

**RPAL**
- `RPAL/classification.py`: This code handles training & testing of the classifier and returns the results.
- `RPAL/constraints.py`: This code enables easy checking of spatial and temporal bias in the data.
- `RPAL/data.py`: This code handles the various data manipulations required.
- `RPAL/grapher.py`: This code generates the experiment and results plots.
- `RPAL/loader.py`: This code handles loading the dataset.
- `RPAL/poison.py`: This code performs all the data poisoning.
- `RPAL/recovery.py`: This code generates all the recovery data.

**Results**
- `Results/Data/`: Contains the data presented in the paper.
- `Results/Scripts/`: Contains the scripts used to generate the plots and table data in the paper.

**Experiments**:
- `Drebin-Label-Flip-Deep-Tesseract.py`: Runs all DNN experiments shown in the paper.
- `Drebin-Label-Flip-RF-Tesseract.py`: Runs all RF experiments shown in the paper.
- `Drebin-Label-Flip-SVM-Tesseract.py`: Runs all SVM experiments shown in the paper.

**Other**:
- `deepdrebin.py`: Implements a SKLean compatible implementation of the architecture used in 'Adversarial Examples for Malware Detection' by Grosse et al.
- `Clean_Label_Poisoning_Mapping.py`: Generates the feature-flip mappings used to mimic the label-flip attack.

***
