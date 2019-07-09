# Named-Entity Recognition for Indonesian Language using Bidirectional LSTM-CNNs

In this paper, we describe the implementation of Named-Entity Recognition (NER) for Indonesian Language by using various deep learning approaches, yet mainly focused on hybrid bidirectional LSTM (BLSTM) and convolutional neural network (CNN) architecture. There are already several developed NERs dedicated to specific languages such as English, Vietnamese, German, Hindi and many others. However, our research focuses on Indonesian language. Our Indonesian NER is managed to extract the information from articles into 4 different classes; they are Person, Organization, Location, and Event. We provide comprehensive comparison among all experiments by using deep learning approaches. Some discussions related to the results are presented at the end of this paper. Through several conducted experiments, Indonesian NER has successfully achieved a good performance.

## Architecture

For the base model architecture, please refer to our paper here: [https://www.sciencedirect.com/science/article/pii/S1877050918314832](https://www.sciencedirect.com/science/article/pii/S1877050918314832)

## Corpus

For this research, we use two corpus.

One corpus for training a word embedding model, and another one for the NER itself.

We could also choose to use a pretrained model instead.

### Word Embedding

As mentioned in the paper, we used word2vec. The corpus comes from the articles we crawled, and preprocessed.

### Tagged Data for NER

The training file is a CSV file with two columns: word, entity type (with header: word, tag).

Each row contains a pair of word-entity_type.

Each sentence is separated by a blank line.

You can download the training file example [here](dataset/example.csv).

## Training Script

Training script is provided as reference [here](train.py).

```bash
TRAIN_DIR=dataset/ WORD_EMBEDDING_DIR=dim-100-skip-window-1/ OUTPUT_DIR=models python3 train.py
```





---

Kurio-DSE@2018