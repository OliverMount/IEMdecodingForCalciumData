# IEM decoding For Calcium Data

The Inverted Encoding Model (IEM) is a neuroimaging analysis approach applied traditionally to functional Magnetic Resonance Imaging (fMRI), Electroencephalography (EEG) data. It involves decoding sensory information from brain activity patterns, working in the opposite direction of traditional encoding models. Instead of predicting brain activity given external stimuli (as in encoding models), IEM aims to reconstruct and decode features or representations of stimuli from observed neural patterns. This technique is particularly useful for investigating the neural representation of sensory information and understanding how different brain regions contribute to encoding sensory features.

IEM can be described mathematically via four linear equations. For more details about the IEM please refer to  Myers et al. [^1].

![Equation](https://latex.codecogs.com/png.latex?\int_0^\infty%20e^{-x^2}%20dx%20=%20\frac{\sqrt{\pi}}{2})


[^1]: Nicholas E MyersGustavo RohenkohlValentin WyartMark W WoolrichAnna C NobreMark G Stokes (2015) Testing sensory evidence against mnemonic templates eLife 4:e09000.

1. ![Equation](https://latex.codecogs.com/svg.image?%20X=WC%20)
2. ![Equation](https://latex.codecogs.com/svg.image?%5Cwidehat%7BW%7D=XC%5ET(CC%5ET)%5E%7B-1%7D)
3. ![Equation](https://latex.codecogs.com/svg.image?%20X=WC%20)
4. ![Equation](https://latex.codecogs.com/svg.image?%20X=WC%20)






This repository contains the python class `InvertedEncoding` (in the scripts/utils.py) that implements the IEM. We employed IEM via for extracting motion direction from the preprocessed calcium imaging of mouse V1 and PPC data via parallel processing the time points.

### Instruction for running the code

1. git clone the repository to a local folder in your computer

2. cd in to the scripts folder and set the following paths in the  decoding.py
	a. data_path: This is the path where the preprocessed calcium data are stored
	b. pval_path: pvalues of the tuned and untuned neurons are stored here

	The two levels of the data directory will look like this

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
3. Run the scripts.py it would create a decoding folder (this is the results folder) in the level as of the scripts folder. The structure  of the decoding folder would be the following

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

Note: Individual figures for each condition is stored inside the plots directory. Montage folder (that cotains sewed figure files) will be recrated only if the platform is Lixux (if montage is installed). 
