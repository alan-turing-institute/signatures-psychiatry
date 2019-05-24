# A signature-based machine learning model for bipolar disorder and borderline personality disorder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alan-turing-institute/signatures-psychiatry/lab-add-synth-data?urlpath=lab)

This repository contains the code from the paper [*A signature-based machine learning model for bipolar disorder and borderline personality disorder*](https://doi.org/10.1038/s41398-018-0334-0):

> Perez Arribas, I., Goodwin, G.M., Geddes, J.R., Lyons, T. and Saunders, K.E., 2018. A signature-based machine learning model for distinguishing bipolar disorder and borderline personality disorder. _Translational Psychiatry_, _8_(1), p.274. DOI: [10.1038/s41398-018-0334-0](https://doi.org/10.1038/s41398-018-0334-0).

**Contents**
- [Reproducible Research Champions](#reproducible-research-champions)
- [Data](#data)
- [Setting up signatures-psychiatry](#setting-up-signatures-psychiatry)
- [Generating figures and tables from the paper](#generating-figures-and-tables-from-the-paper)

## Reproducible Research Champions

In May 2018, Terry Lyons was selected as one of the Alan Turing Institute's Reproducible Research Champions - academics who encourage and promote reproducible research through their own work, and who want to take their latest project to the "next level" of reproducibility.

The Reproducible Research programme at the Turing is led by Kirstie Whitaker and Martin O'Reilly, with the Champions project also involving Louise Bowler from the Research Engineering Group.

Each of the Champions' projects will receive several weeks of support from the Research Engineering Group throughout 2018-2019. Over this time, we will work on the project with Terry and Imanol and will track our efforts in this repository. So far, we've added installation instructions, set the project up to work with [Binder](https://mybinder.readthedocs.io/en/latest/), made it possible to use synthetic data in the same workflow and added some tests. Given our focus on reproducibility, we obviously don't intend to change any of the code's core functionality - but we hope that our work, both past and future, will make it easier for you to install, use and test out your own ideas with the methods used in the `signatures-psychiatry` project.

You can find out more about the Turing's Reproducible Research Champions project [here](https://github.com/alan-turing-institute/ReproducibleResearchResources).

## Data

The dataset used in the study is confidential, so we are not able to publicly release it. Access to the dataset is restricted to staff and students of the University of Oxford who have received the appropriate permission.

However, we feel that it is important that some similar data are provided for the purposes of demonstrating how the methods outlined in the paper work. To this end, we include three sets of synthetic data, and one example of a fake entry in the dataset.

**Mood score data** The original dataset contains the mood scores of participants who were either healthy or were diagnosed with borderline personality disorder or bipolar disorder. Participants recorded their mood score on a seven-point scale across six different categories (anxiety, elation, sadness. anger, irritability and energy) at approximately daily intervals. Further details of this dataset, which was collected as part of the _Automated Monitoring of Symptoms Severity_ (AMoSS) study, can be found in the [paper](https://doi.org/10.1038/s41398-018-0334-0). Access to this dataset is limited and so it is not included in this repository.

**Synthetic signature data** Synthetic data is data that has been generated to exhibit the same statistical properties as the original data, without containing the original entries. The synthetic data in this repository was derived from the signatures of the original mood score data, and is therefore in signature form itself. Each dataset contains mood score signatures and their associated diagnostic classification. The synthetic signatures were derived from all mood score signatures from each of the three diagnostic classifications, so the concept of "participant" does not apply when using this dataset. Three synthetic datasets were generated and have been included in the `synthetic-data` folder.

**Fake mood score data** This repository contains an example of a fake mood score dataset from one participant in `data/fake_patient.csv`. This data is not statistically related to the original mood score data but is presented in the same format in order to illustrate the data normalisation process and how Figures 1 and 2 in the paper were generated.

## Setting up signatures-psychiatry

_The instructions below assume you are comfortable cloning a git repository and running Python scripts via the command line.
If not, you may find the tutorials available from [GitHub](https://help.github.com/en/articles/cloning-a-repository) and [Software Carpentry](http://swcarpentry.github.io/python-novice-inflammation/10-cmdline/index.html) helpful. You can also open an interactive version of this project on_ [`mybinder.org`](https://mybinder.org) _by clicking the badge below._

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alan-turing-institute/signatures-psychiatry/lab-add-synth-data?urlpath=lab)

_Once the Binder project loads, open a Python 2 console and skip down this page to ["Generating figures and tables from the paper"](#generating-figures-and-tables-from-the-paper)._

Begin by obtaining a copy of this repository using
```
git clone https://github.com/alan-turing-institute/signatures-psychiatry.git
```
and move into the directory
```
cd signatures-psychiatry
```
This project uses `Python 2.7` and the packages listed in `requirements.txt`.
Let's start by setting up a virtual environment and installing the dependencies inside it.
We give examples here for using `virtualenv` and `conda`.

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

_If you are running this project via Binder (or any other Jupyter Lab installation), open a Python 2 console. The commands need a minor change in this environment - swap_ `python` _for_ `%run`_, and use the_ `shift+enter` _keys to run the cell._

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

The heat maps in Figure 3 can be produced from either the original data or the synthetic signatures.
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
