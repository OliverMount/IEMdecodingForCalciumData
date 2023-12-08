# IEM decoding For Calcium Data

The Inverted Encoding Model (IEM) is a neuroimaging analysis approach applied traditionally to functional Magnetic Resonance Imaging (fMRI), Electroencephalography (EEG) data. It involves decoding sensory information from brain activity patterns, working in the opposite direction of traditional encoding models. Instead of predicting brain activity given external stimuli (as in encoding models), IEM aims to reconstruct and decode features or representations of stimuli from observed neural patterns. This technique is particularly useful for investigating the neural representation of sensory information and understanding how different brain regions contribute to encoding sensory features.

IEM can be described mathematically by four linear equations. For more details about the IEM, please refer to Myers et al. [^1].

[^1]: Nicholas E MyersGustavo RohenkohlValentin WyartMark W WoolrichAnna C NobreMark G Stokes (2015) Testing sensory evidence against mnemonic templates eLife 4:e09000.

1. Training data model: 
![Equation](https://latex.codecogs.com/svg.image?%20X=WC%20)

2. Weight estimate based on trianing data
![Equation](https://latex.codecogs.com/svg.image?%5Cwidehat%7BW%7D=XC%5ET(CC%5ET)%5E%7B-1%7D)

3. Test data model:
![Equation](https://latex.codecogs.com/svg.image?Y=%5Cwidehat%7BW%7DC_%7B%5Ctext%7Btest%7D%7D)

4. Estimated tuning curve of the test data:
![Equation](https://latex.codecogs.com/svg.image?%20C_%7B%5Ctext%7Btest%7D%7D=(%5Cwidehat%7BW%7D%5ET%5Cwidehat%7BW%7D)%5E%7B-1%7D%5Cwidehat%7BW%7D%5E%7BT%7DY)


This repository contains the Python class `InvertedEncoding` (in the scripts/utils.py file) that implements the IEM. We employed the IEM for extracting motion direction from the preprocessed calcium imaging of mouse V1 and PPC data through parallel processing of the time points. A brief demo of how to create an instance of the model is provided in `develop/IEMdemo.ipynb`.

### Instruction for running the code

1. git clone the repository to a local folder in your computer

2. Navigate to the 'scripts' folder and set the following paths in the 'decoding.py' file:

	a. `data_path`: This is the path where the preprocessed calcium data is stored.

	b. `pval_path`: P-values of the tuned and untuned neurons are stored here.

	The directory structure at two levels of the data folder will look like this:

	```

	├── passive
	│   ├── PPC_135
	│   ├── PPC_45
	│   ├── PPC_90
	│   ├── V1_135
	│   ├── V1_45
	│   └── V1_90
	├── pvals
	│   ├── task_PPC_135.mat
	│   ├── task_PPC_45.mat
	│   ├── task_PPC_90.mat
	│   ├── task_V1_135.mat
	│   ├── task_V1_45.mat
	│   └── task_V1_90.mat
	└── task
	    ├── PPC_135
	    ├── PPC_45
	    ├── PPC_90
	    ├── V1_135
	    ├── V1_45
	    └── V1_90

	```
3. Run the `scripts.py` file; it will create a `decoding` folder (results folder) at the same level as the `scripts` folder. The structure of the 'decoding' folder will be as follows:


```
.
├── plots
│   ├── montages
│   ├── task_V1_45_10.png
│   ├── task_V1_45_20.png
│   ├──        . 
│   ├──        .  
│   └── task_PPC_135_100.png
├── slopes
│   └── task
│   └── passive
└── tuning_curves
    ├── passive
    └── task
```

Note: Individual figures for each condition are stored inside the 'plots' directory. The 'Montage' folder (containing sewed figure files) will be recreated only if the platform is Linux (if 'montage' is installed). 
