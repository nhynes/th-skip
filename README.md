# Skip-instructions

Torch implementation of skip-thoughts model from the paper:

```
Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. "Skip-Thought Vectors." arXiv preprint arXiv:1506.06726 (2015).
```

Original theano implementation can be found [here](https://github.com/ryankiros/skip-thoughts).


## Installation

Install [Torch](http://torch.ch/docs/getting-started.html):
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```

Install the following packages:

```
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install rnn
luarocks install moses
luarocks install moonscript
```

Install CUDA and cudnn. Then run:

``` 
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

A custom fork of torch-hdf5 with string support is needed:

```
cd ~/torch/extra
git clone https://github.com/nhynes/torch-hdf5.git
cd torch-hdf5
git checkout chars2
luarocks build hdf5-0-0.rockspec
```

We also use python2.7 to assemble the dataset with numpy and h5py packages:

```pip install -r requirements.txt```

## Usage

Here we describe how to train the model and extract encoded sentence features using recipe instructions from the Recipe1M dataset:

```
cite here
```

you can download the dataset [here]().


- Create directories where data will be stored:
```
mkdir data
mkdir snaps
```

- Prepare the dataset running from ```scripts```directory:

```python mk_dataset.py --dataset /path/to/recipe1M/ --vocab /path/to/w2v/vocab.txt --toks /path/to/tokenized.txt```

where ```tokenized.txt``` contains text instructions for the entire dataset. Different instructions for the same recipe are separated by '\t' and different recipe instructions are delimited with '\n'. ```vocab.txt``` is a file containing the entries of the previously trained [word2vec](https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip) model. 


- Train the model with:

```
moon main.moon 
-dataset data/dataset.h5 
-dim 1024 
-nEncRNNs 2 
-snapfile snaps/encoder_im2r 
-savefreq 500 
-batchSize 128 
-w2v /path/to/w2v/vocab.bin
```

- Get encoder from the trained model:

```
cd scripts;
moon extract_encoder.moon
../snaps/snapfile.t7
encoder.t7
true
```
- Extract features with:

```
cd scripts;
moon encode.moon 
-data ../data/dataset.h5
-model encoder.t7
-partition test
-out encs_test_1024.t7
```

Run for ```-partition = {train,val,test}``` and ```-out={encs_train_1024,encs_val_1024,encs_test_1024}``` to extract features for the dataset.
