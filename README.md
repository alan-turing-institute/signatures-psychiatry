# A signature-based machine learning model for bipolar disorder and borderline personality disorder

This repository contains the code from the paper *A signature-based machine learning model for bipolar disorder and borderline personality disorder*:

> Perez Arribas, I., Goodwin, G.M., Geddes, J.R., Lyons, T. and Saunders, K.E., 2018. A signature-based machine learning model for distinguishing bipolar disorder and borderline personality disorder. _Translational Psychiatry_, _8_(1), p.274.

## Data

Given that the data of the original study is confidential, we are not permitted to share it.

## Setting up signatures-psychiatry

_The instructions below assume you are comfortable cloning a git repository and running Python scripts via the command line.
If not, you may find the tutorials available from [GitHub](https://help.github.com/en/articles/cloning-a-repository) and [Software Carpentry](http://swcarpentry.github.io/python-novice-inflammation/10-cmdline/index.html) helpful._ 

Begin by obtaining a copy of this repository using
```
git clone git@github.com:alan-turing-institute/signatures-psychiatry.git
```
and move into the directory
```
cd <your-path-name>/signatures-psychiatry
```
This project uses `Python 2.7` and the packages listed in `requirements.txt`.
Let's start by setting up a virtual environment and installing the dependencies inside it.

If you use CPython, use `virtualenv` to set up a virtual environment named `env`.
Activate the environment and install the packages required by this project using `pip`.
```
virtualenv env                                      # Use if your default Python version is 2.7
virtualenv --python=<path-to-your-python-2.7> env   # If not, specify the path manually
source env/bin/activate
pip install -r requirements.txt
```

If you use Anaconda, create the virtual environment with `conda` and install the `pip` package directly into the environment.
Once the environment is activated, the dependencies can be installed using `pip` (we use `pip` because the `esig` package is not currently available through `conda`).
```
conda create -n sig-psy python=2.7 pip
conda activate sig-psy
pip install -r requirements.txt
```

With the virtual environment set up and all the dependencies installed, we can use the scripts in this project by following the instructions below.

## Generating figures and tables from the paper

#### Table 1: Accuracy and area under the ROC curve

If you have access to the full dataset, run
```
python pairwise_group_classification.py
```
To run the same analysis on the synthetic signatures, use
```
python pairwise_group_classification.py --synth
```
By default, the synthetic cohort `772192` will be used, or the cohort ID can be specified with `--synth=<cohort-id>`.

The random seed for this script is set by default to `83042`, or it can be changed using `--seed=<random-seed>`.

The pairwise values will be displayed in the terminal and also saved to a log file, `log/mood_prediction.log`.

#### Table 2: Demographic characteristics of study participants

The content of this table was gathered manually.

#### Figure 1: Normalised anxiety scores of a sample participant

This repository contains a set of "fake" data from a single patient.
To convert this dataset into the normalised format shown in Figure 1, run 
```
python plot_path.py
```

#### Figure 2: Pairwise normalised mood scores of a sample participant

The same command used above also produces the pairwise mood score plots for the fake patient.
```
python plot_path.py
```

#### Figure 3 (top row): 

The heat maps in figure 3 can be produced from either the original data or the synthetic signatures.
To use the original data, run
```
python heat_map.py
```
and to use the synthetic signatures, use
```
python heat_map.py --synth
```
Alternatives to the default cohort of `772192` and random seed of `1` can be set with, for example, the options `--synth=239673 --seed=100`.

The script will save three figures in `.png` format. The names of the figures and the settings can be found in the `heat_map.log` file in the `log` folder.
 
#### Figure 3 (bottom row): Accuracy and MAE of predictions of future mood score

The lower row of plots in Figure 3 require mood score data, so we cannot use the synthetic signature data here.

#### Table 3: Summary of accuracy and MAE of predictions of each aspect of next-day mood score

To generate the accuracy and MAE scores, run
```
python mood_prediction.py
```
As the next-day mood score is required, this script cannot be applied to the synthetic data which is already in the signature format.
 