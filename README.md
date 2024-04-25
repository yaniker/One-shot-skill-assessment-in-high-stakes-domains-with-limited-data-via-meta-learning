 	Licence: CC BY-SA-4.0
	Author: Dr. Erim Yanik
	Affiliation: FAMU-FSU College of Engineering
	Date: 04.05.2024
	For inquiries: erimyanik@gmail.com

#############################################

This code is released as a part of the research article:

 	"One-shot skill assessment in high-stakes domains with limited data via meta learning
  
	"Erim Yanik1*, Steven Schwaitzberg2, Gene Yang2, Xavier Intes3, Jack Norfleet4, Matthew Hackett4, and Suvranu De1
 
	1 College of Engineering, Florida A&M University and The Florida State University, USA
	2 School of Medicine and Biomedical Sciences, University at Buffalo, USA
	3 Biomedical Engineering Department, Rensselaer Polytechnic Institute, USA
	4 U.S. Army Combat Capabilities Development Command Soldier Center STTC, USA

#############################################

Please use the following citation:

	BibTeX:@article{yanik2024one,
	  title={One-shot skill assessment in high-stakes domains with limited data via meta learning},
	  author={Yanik, Erim and Schwaitzberg, Steven and Yang, Gene and Intes, Xavier and Norfleet, Jack and Hackett, Matthew and De, Suvranu},
	  journal={Computers in Biology and Medicine},
	  pages={108470},
	  year={2024},
	  publisher={Elsevier}
	}

	APA: Yanik, E., Schwaitzberg, S., Yang, G., Intes, X., Norfleet, J., Hackett, M., & De, S. (2024). One-shot skill assessment in high-stakes domains with limited data via meta learning. Computers in Biology and Medicine, 108470.

#############################################

Nomenclature:
     PC     : pattern cutting
     STB    : laparoscopic suturing
     ST/JST : robotic suturing
     KT/JKT : knot tying
     NP/JNP : needle passing
     Cholec : laparoscopic chlecystectomy

#############################################

Contained files:

* main.py: main script that contains the entire workflow of meta learning training and adaptation.
* test_on_cholec.py: used to test the trained meta learners on laparoscopic cholecystectomy.
* config.py" the configurations, i.e., parameters and hyperparameters for meta learning development.
* import_datasets.py: imports the datasets.
* generator_1D.py: yields the batches during the training.
* trustworthiness_conditional.py: calculates the trustworthiness of the model.
* VBA_Net.py: the backbone deep neural network - VBA-Net.
* ProtoMAML.py: meta learning script responsible from training to adaptation.
* utils_inputs.py: utility functions for importing data.
* utils_logging.py: utility functions for logging the process and the results.
* utils_model.py: utility functions needed for meta learner to run properly.
* The provided inputs in the "input folder" are for the self-supervised feature sets of 8 in size (Temporal_length, 8). The remaining sizes were not provided at this time (2,4,16,32,64,128).
* The provided results in the "output" folder are after one run with a random seed to validate the code.

#############################################

 To run the code please use the following on the root directory:

	python main.py 2 1 8 STB
 
"python main.py 2 1 8 PC",
"python main.py 2 1 8 JST",
"python main.py 2 1 8 JKT",
"python main.py 2 1 8 JNP"
	
Here the user inputs are as follows: number of training classes per batch, 
				     number of training samples per batch,
				     self-supervised feature size,
				     validation dataset (the rest go to training)

#############################################

To test the trained meta learners on laparoscopic cholecystoctomy dataset please use the following on the root directory:
	
	python adapt_cholec.py 2 1 8 STB
 
"python adapt_cholec.py 2 1 8 PC",
"python adapt_cholec.py 2 1 8 JST",
"python adapt_cholec.py 2 1 8 JKT",
"python adapt_cholec.py 2 1 8 JNP"

Here the user inputs are as follows: number of training classes per batch, 
				     number of training samples per batch,
				     self-supervised feature size,
				     validation dataset (used as a validation during training with the other datasets.)

Currently there is one model released for each validation dataset, so the adaptation to laparoscopic cholecystectomy happens through these released models.

Currently the developer option is turned on (dev = True); hence the results won't be saved. To save your results please set dev to False.

#############################################

Acknowledgement:

	The US Army Futures Command, Combat Capabilities Development Command Soldier Center STTC cooperative research agreement #W912CG-21-2-0001


#############################################

Parts of the code was taken from:

	Website: https://pytorch-lightning.readthedocs.io/en/1.5.10/notebooks/course_UvA-DL/12-meta-learning.html
 	Author: Phillip Lippe
	License: CC BY-SA
	Generated: 2021-10-10T18:35:50.818431
	Acknowledged in the paper: YES.

#############################################

 The JIGSAWS dataset was taken from:
 
	Website: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
  	Reference: Yixin Gao, S. Swaroop Vedula, Carol E. Reiley, Narges Ahmidi, Balakrishnan Varadarajan, Henry C. Lin, Lingling Tao, Luca Zappella, Benjam ́ın B ́ejar, David D. Yuh, Chi Chiung Grace Chen, Ren ́e Vidal, Sanjeev Khudanpur and Gregory D. Hager, The JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset for Human Motion Modeling, In Modeling and Monitoring of Computer Assisted Interventions (M2CAI) – MICCAI Workshop, 2014.
   	Cited and acknowledged in the paper: YES.


