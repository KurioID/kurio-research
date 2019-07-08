# Copyright 2019 PT. Kurio

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import csv
import os
import time

from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution1D
from keras.layers import Bidirectional, Embedding, LSTM, concatenate
from keras.layers.core import Dense, SpatialDropout1D
from keras.models import Input, Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import tensorflow as tf

train_directory = os.getenv("TRAIN_DIR", "dataset")
word_embedding_directory = os.getenv("WORD_EMBEDDING_DIR", "word-embedding")
output_directory = os.getenv("OUTPUT_DIR", "models")

LABEL2ID = {
    "U-PERSON": 1,
    "B-PERSON": 2,
    "I-PERSON": 3,
    "L-PERSON": 4,
    "U-ORG": 5,
    "B-ORG": 6,
    "I-ORG": 7,
    "L-ORG": 8,
    "U-LOC": 9,
    "B-LOC": 10,
    "I-LOC": 11,
    "L-LOC": 12,
    "U-EVENT": 13,
    "B-EVENT": 14,
    "I-EVENT": 15,
    "L-EVENT": 16,
    "O": 17
}

# +1 = label for zero padding
nb_classes = max(LABEL2ID.values()) + 1

sequence_length = 120

meta_filename = "model.ckpt.meta"
vocab_filename = "metadata.tsv"
model_filename = "model.h5"


def load_datasets(directory):
    if not os.path.exists(directory):
        print("file not found!")
        return None, None

    t0 = time.time()
    sentences = []
    labels = []
    csv_files = [fp for fp in os.listdir(directory) if fp.endswith(".csv")]
    for csv_file in csv_files:
        filename = os.path.join(directory, csv_file)
        with codecs.open(filename, "r",
                         encoding="utf-8",
                         errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            words = []
            tags = []
            for i, x in enumerate(reader):
                if x[0] != "" and x[1] != "":
                    tags.append(x[1])
                    words.append(x[0].lower())
                elif x[0] == "" and x[1] == "":
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
                else:
                    print("Error on {} in line {}".format(
                        filename.split("/")[-1], i+2
                    ))
                    print("Error message, word [{}] with tag [{}]".format(
                        x[0], x[1]
                    ))

    print("Load {} data finish in {}s".format(
        len(sentences), time.time() - t0
    ))

    return sentences, labels


def load_word2vec(directory, meta_filename, vocab_filename):
    vocabulary = []
    vocab_filename = os.path.join(directory, vocab_filename)
    with open(vocab_filename) as f:
        for line in f:
            vocabulary.append(line.strip())

    meta_filename = os.path.join(directory, meta_filename)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_filename)
        saver.restore(sess, os.path.join(directory, "model.ckpt"))
        tvars = tf.trainable_variables(scope="embeddings")
        vectors = sess.run(tvars[0])

    return vocabulary, vectors


def create_dictionary(vocabulary, label2id):
    id2word = {}
    index = 0
    for word in vocabulary:
        id2word.update({index: word})
        index += 1

    word2id = dict((v, k) for k, v in id2word.items())
    id2label = dict((v, k) for k, v in label2id.items())
    return word2id, id2word, label2id, id2label


def encode(x, n):
    temp = np.zeros(n)
    temp[x] = 1
    return temp


def create_training_data(sentences, labels, sequence_length,
                         nb_label, word2id, label2id):
    X_encoder = []
    for x in sentences:
        sentence_encoder = []
        for c in x:
            if c in word2id:
                sentence_encoder.append(word2id[c])
            else:
                sentence_encoder.append(word2id["UNK"])
        X_encoder.append(sentence_encoder)

    y_encoder = [[0] * (sequence_length - len(x))
                 + [label2id[c] for c in x] for x in labels]
    y_encoder = [[encode(c, nb_label) for c in x] for x in y_encoder]

    X = pad_sequences(X_encoder, maxlen=sequence_length)
    y = pad_sequences(y_encoder, maxlen=sequence_length)
    return X, y


def create_embedding_layer(vectors, id2word, sequence_length):
    embeddings_index = {}
    for i in range(vectors.shape[0]):
        word = id2word[i]
        vector = vectors[i]
        embeddings_index[word] = vector

    embedding_matrix = np.zeros((len(id2word) + 1, vectors.shape[1]))
    for i, word in id2word.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embeddings_index["UNK"]

    embedding_layer = Embedding(len(id2word) + 1,
                                vectors.shape[1],
                                weights=[embedding_matrix],
                                input_length=sequence_length,
                                trainable=False)

    return embedding_layer


def get_model(embeddings_layer, sequence_length, nb_classes):
    word_input = Input(shape=(sequence_length,), name="X_input")
    embedded = embeddings_layer(word_input)
    embedded_d = SpatialDropout1D(0.4)(embedded)

    conv1 = Convolution1D(filters=64, kernel_size=2, padding="same",
                          activation="relu")(embedded_d)
    conv2 = Convolution1D(filters=64, kernel_size=4, padding="same",
                          activation="relu")(embedded_d)
    conv3 = Convolution1D(filters=64, kernel_size=6, padding="same",
                          activation="relu")(embedded_d)
    conv4 = Convolution1D(filters=64, kernel_size=8, padding="same",
                          activation="relu")(embedded_d)
    conv5 = Convolution1D(filters=64, kernel_size=10, padding="same",
                          activation="relu")(embedded_d)

    merge_layer = concatenate([conv1, conv2, conv3, conv4, conv5],
                              name="convolutional_concat")

    blstm = Bidirectional(LSTM(units=100, return_sequences=True,
                               recurrent_dropout=0.5))(merge_layer)
    blstm = Bidirectional(LSTM(units=100, return_sequences=True,
                               recurrent_dropout=0.5))(blstm)
    predict = Dense(nb_classes, activation="softmax")(blstm)
    model = Model(word_input, predict)

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def predict_report(y_val, y_pred, label2id):
    y_temp = [list(x) for x in y_pred]
    y_val_temp = [list(x) for x in np.argmax(y_val, axis=-1)]
    y_pred = []
    y_target = []
    for x in range(len(y_temp)):
        y_pred += y_temp[x]
        y_target += y_val_temp[x]

    target_names = ["support"] + [k for k, v in label2id.items()]
    return metrics.classification_report(
        y_target,
        y_pred,
        target_names=target_names
    )


def train(model, X_train, X_test, y_train, y_test, nb_epoch):
    nb_batch_size = 128
    early_stopping = False

    if early_stopping:
        callbacks = [EarlyStopping(patience=8)]
    else:
        callbacks = None

    print(model.summary())

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=nb_batch_size,
        epochs=nb_epoch,
        validation_data=[X_test, y_test],
        callbacks=callbacks,
        shuffle=True
    )

    return model


def save_model(model, directory, filename):
    filename = os.path.join(directory, filename)

    # save a model if model name is not exits or model is available
    if not os.path.exists(directory):
        os.makedirs(directory)

    model.save(filename)
    print('model has been save')


t0 = time.time()
label2id = LABEL2ID
sentences, labels = load_datasets(directory=train_directory)

vocab, vectors = load_word2vec(
    word_embedding_directory,
    meta_filename,
    vocab_filename
)

word2id, id2word, label2id, id2label = create_dictionary(
    vocabulary=vocab,
    label2id=label2id
)

X, y = create_training_data(
    sentences=sentences,
    labels=labels,
    sequence_length=sequence_length,
    nb_label=nb_classes,
    word2id=word2id,
    label2id=label2id
)

embeddings_layer = create_embedding_layer(
    vectors=vectors,
    id2word=id2word,
    sequence_length=sequence_length
)

model = get_model(
    embeddings_layer=embeddings_layer,
    sequence_length=sequence_length,
    nb_classes=nb_classes
)

test_size = 0.3
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

clf = train(model=model, X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test, nb_epoch=5)

y_pred = model.predict(np.array(X_test))
y_pred = np.argmax(y_pred, axis=-1)
clf_report = predict_report(y_test, y_pred, label2id)
print("Classification Report:")
print("\n{}".format(clf_report))

save_model(clf, output_directory, model_filename)
print("Process done in {:.3f}s".format(time.time() - t0))
