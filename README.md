# Starting kit for time-series generation hackathon

In this repository, we provide a standard pipeline to help you with the kick off of our hackathon. In this pipeline,
we include: 
1) data importation and explanatory analysis,
2) the model build-up for both the generator and discriminator using LSTM modules,
3) training algorithm design,
4) offline evaluation module.

The data used for training and testing all come from the public data from the main hackathon website,

## Environment Setup
The code has been tested successfully using Python 3.8 and pytorch 1.11.0. A typical process for installing the package dependencies involves creating a new Python virtual environment.

To install the required packages, run the following:
```console
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
pip install cupy-cuda102
pip install -r requirements.txt
```

For code illustration, please take a closer look on the Jupyter-Notebook we created, namely, example_pipeline.ipynb.

Finally, we wish you good luck during the competition and most importantly, have fun!!!
