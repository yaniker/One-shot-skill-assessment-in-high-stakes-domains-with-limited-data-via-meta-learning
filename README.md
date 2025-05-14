
**!!! If you have any questions or need features please add them to `issues`. !!!**

## License

This repository contains two distinct codebases under separate licenses:

### 🔹 Model Code (VBA_net and Related)

The code in the `/model/` directory (including `vbanet.py` and related modules) was developed by **Erim Yanik** and is licensed under the **GNU General Public License v3.0 (GPLv3)**.

This means:
- You **must attribute** Erim Yanik as the original author.
- Any modifications or derivative works **must also be released** under GPLv3.
- **Commercial or closed-source use is not permitted** under this license.

Full license: [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html)

### 🔹 Third-Party Code (ProtoMAML)

The code in the `/protomaml/` directory is **adapted from external sources** and is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license.

This means:
- You must **attribute the original creators** of ProtoMAML.
```bash
Website: https://pytorch-lightning.readthedocs.io/en/1.5.10/notebooks/course_UvA-DL/12-meta-learning.html
Author: Phillip Lippe
```
- Any use or derivative of that code must be licensed under **CC BY-SA 4.0**.

We do **not claim authorship** of the ProtoMAML code. It is included for reference and compatibility only.

Full CC BY-SA license: [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)

`!!! Unauthorized use of this code in violation of its license will be documented and made publicly visible. Attempts to remove attribution or bypass licensing terms will be treated as a violation of open-source ethics and may be disclosed to relevant communities, funders, and publishers. !!!`

This code is released as a part of the research article:
```bash
BibTeX:@article{yanik2024one,
	  title={One-shot skill assessment in high-stakes domains with limited data via meta learning},
	  author={Yanik, Erim and Schwaitzberg, Steven and Yang, Gene and Intes, Xavier and Norfleet, Jack and Hackett, Matthew and De, Suvranu},
	  journal={Computers in Biology and Medicine},
	  pages={108470},
	  year={2024},
	  publisher={Elsevier}
	}
```
```bash
APA: Yanik, E., Schwaitzberg, S., Yang, G., Intes, X., Norfleet, J., Hackett, M., & De, S. (2024). One-shot skill assessment in high-stakes domains with limited data via meta learning. Computers in Biology and Medicine, 108470.
```

## HOW TO USE?

### Nomenclature:
     PC     : pattern cutting
     STB    : laparoscopic suturing
     ST/JST : robotic suturing
     KT/JKT : knot tying
     NP/JNP : needle passing
     Cholec : laparoscopic chlecystectomy

### Contained files:

* main.py: main script that contains the entire workflow of meta learning training and adaptation.
* test_on_cholec.py: used to test the trained meta learners on laparoscopic cholecystectomy.
* config.py" the configurations, i.e., parameters and hyperparameters for meta learning development.
* import_datasets.py: imports the datasets.
* model/vbanet.py: the backbone deep neural network - VBA-Net.
* protomaml/ProtoMAML.py: meta learning script responsible from training to adaptation.
* utils/dataset_utils.py: utility functions for importing data.
* utils/logging_utils.py: utility functions for logging the process and the results.
* utils/model_utils.py: utility functions needed for meta learner to run properly.

* The provided inputs in the "input folder" are for the self-supervised feature sets of 8 in size (Temporal_length, 8). The remaining sizes were not provided at this time (2,4,16,32,64,128). This is for reproducibility.
* The provided results in the "output" folder are after one run with a random seed to validate the code.

### Run
 To run the code please use the following on the root directory:
```bash
python main.py 2 1 8 STB
 ```

Other examples:  

"python main.py 2 1 8 PC",
"python main.py 2 1 8 JST",
"python main.py 2 1 8 JKT",
"python main.py 2 1 8 JNP"
	
Here the user inputs are as follows: number of training classes per batch, 
				     number of training samples per batch,
				     self-supervised feature size (enter 8),
				     validation dataset (the rest go to training)
### Test

To test the trained meta learners on laparoscopic cholecystoctomy dataset please use the following on the root directory:
```bash
python adapt_cholec.py 2 1 8 STB
 ```

Other examples:

"python adapt_cholec.py 2 1 8 PC",
"python adapt_cholec.py 2 1 8 JST",
"python adapt_cholec.py 2 1 8 JKT",
"python adapt_cholec.py 2 1 8 JNP"

Here the user inputs are as follows: number of training classes per batch, 
				     number of training samples per batch,
				     self-supervised feature size (enter 8),
				     validation dataset (used as a validation during training with the other datasets.)


## JIGSAWS

The JIGSAWS dataset was taken from:
```basH 
Website: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
Reference: Yixin Gao, S. Swaroop Vedula, Carol E. Reiley, Narges Ahmidi, Balakrishnan Varadarajan, Henry C. Lin, Lingling Tao, Luca Zappella, Benjam ́ın B ́ejar, David D. Yuh, Chi Chiung Grace Chen, Ren ́e Vidal, Sanjeev Khudanpur and Gregory D. Hager, The JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset for Human Motion Modeling, In Modeling and Monitoring of Computer Assisted Interventions (M2CAI) – MICCAI Workshop, 2014.
Cited and acknowledged in the paper: YES.
YOU HAVE TO GET THEIR PERMISSION IN ORDER TO USE THE JIGSAWS DATASET.
```


