# Parts of Speech Tagging (using Hidden Markov Models (HMM) -Viterbi Algorithm)

> In this project, We will see implementation of viterbi algorithm, (enhancement on HMM which reduce exponential to polynomial time complexity) to perform parts of speech tagging of given sentences.

```
Given:
No. of Parts of speech tags = N
No. of Tokens per sentence = L
```

### [Bruteforce Approach] O(N<sup>L</sup>) `==>` O(L * N<sup>2</sup>) [Viterbi Algoritm]


## Data

* Train file: It consists of tagged training data in word/TAG format, with words(tokens) seperated by spaces and each sentence on new line.
* Test file: It consist of untagged data, which to be tested on trained model, with words(tokens) seperated by spaces and each sentence on new line.
* Test Results file: It consist of true tagged data (which to be used for test score evalute purpose) with in word/TAG format, with words(tokens) seperated by spaces and each sentence on new line.

**NOTE:**
If training file not provided then by default [NLTK Brown Corpus](http://korpus.uib.no/icame/brown/bcm.html) will be taken as training data and [NLTK Universal Tagset](https://www.nltk.org/_modules/nltk/tag/mapping.html) will be used.


## Installation

### From GitHub
```
$ git clone https://github.com/parth-np/Parts-of-Speech-Tagging.git
$ cd  POS_Tagging
$ python3 -m venv pos
$ source venv/bin/activate  
$ pip install -r requirements.txt 
```

## Usage

### Train the Model 

```
$ python hmm_learn.py -h
usage: hmm_learn.py [-h] [-i INPUT_FILE] [-m MODEL_OUTPUT_FILE] [-v]

Train model for Parts of Speech Tagging.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE,        --input_file INPUT_FILE
                        Input Train file. (Default: NLTK Brown Corpus)
  -m MODEL_OUTPUT_FILE, --model_output_file MODEL_OUTPUT_FILE
                        Model file Path to save model
  -v, --verbose         increase output verbosity (Set Log Level: INFO)


```
#### With Training data 
```
python hmm_learn.py -i corpus/train_tagged.txt -m ./train_model/custom_pos_model.h5 -v

```
#### Without Training data (Using Default NLTK Brown corpus as training data)
```
$ python hmm_learn.py -m train_model/brown_pos_model.h5 -v 
INFO:__main__:Loading Train Data....
INFO:__main__:Downloading NLTK 'brown' Corpus and 'universal-tagset' for POS Tagging
[nltk_data] Downloading package brown to /home/parth/nltk_data...
[nltk_data]   Package brown is already up-to-date!
[nltk_data] Downloading package universal_tagset to
[nltk_data]     /home/parth/nltk_data...
[nltk_data]   Package universal_tagset is already up-to-date!
INFO:__main__:Calculating Transition Probabilities...
100%|███████████████████████████████████████████| 14/14 [00:05<00:00,  2.75it/s]
INFO:__main__:Calculating Emission Probabilities...
100%|███████████████████████████████████████████| 14/14 [00:17<00:00,  1.23s/it]
INFO:__main__:Saving model to train_model/pos_model.h5
```
### Test the Model
```
$ python hmm.py -h
usage: hmm.py [-h] [-o OUTPUT] [-f] [-m MODEL] [-v] sentences

positional arguments:
  sentences             Sentence OR Test file path contaning list of input
                        sentences (Each at newline)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Test Output file
  -f, --from_file       Read Input Sentences to train from a file
  -m MODEL, --model MODEL
                        Model file path
  -v, --verbose         increase output verbosity


```

#### Input: Test file 

```
$ python hmm.py data/test_sents.txt -f -o test_op.txt  -m train_model/custom_pos_model.h5 -v
```
**NOTE:**
For POS-Tags Refer: [Penn Treebank Tagset](https://www.cs.upc.edu/~nlp/SVMTool/PennTreebank.html)

#### Example Capture Context (word:'race')
```
$ python hmm.py "People continue to inquire the reason for the race for the outer space ."  -m ./train_model/custom_pos_model.h5  

People/NNPS continue/VBP to/TO inquire/VB the/DT reason/NN for/IN the/DT race/NN for/IN the/DT outer/JJ space/NN ./.


$ python hmm.py "James is expected to race tomorrow ."  -m ./train_model/custom_pos_model.h5 

James/NNP is/VBZ expected/VBN to/TO race/VB tomorrow/NN ./.


```

### Evaluate Model

```
$ python evaluate.py -h
usage: evaluate.py [-h] -t TEST_OUTPUT -c CORRECT_OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -t TEST_OUTPUT, --test_output TEST_OUTPUT
                        Test Output file contains tagged sentences generated
                        by Model
  -c CORRECT_OUTPUT, --correct_output CORRECT_OUTPUT
                        Test Output file contains correct tagged sentence

```



# References 

* [NPTEL - Natural Language Processing Course L-18, L-19](https://nptel.ac.in/courses/106/101/106101007/)
* [(Alternative of above) NPTEL - Natural Language Processing Course L-16, L-17](https://nptel.ac.in/courses/106/105/106105158/)
* [Chapter from Speech and Language Processing by  Daniel Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/A.pdf)
* [Natural Language Toolkit (NLTK) Library](https://www.nltk.org/)
* [Penn Treebank Tagset](https://www.cs.upc.edu/~nlp/SVMTool/PennTreebank.html)

