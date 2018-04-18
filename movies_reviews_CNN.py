import datetime
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import pickle

import data_helpers
from text_cnn import TextCNN

train_path = "aclImdb/train/"  # source data

# Parameters
# ==================================================


# Data loading params

'''
params = {
    "num_epochs": 200,
    "batch_size": 64,
    "num_filters": 128,
    "filter_sizes": "3,4,5",
    "embedding_dim": 128,
    "l2_reg_lambda": 0.0,
    "evaluate_every": 100,
    "dropout_keep_prob": 0.5,
    "checkpoint_every": 100}
'''

params = {
    "num_epochs": 100,
    "batch_size": 64,
    "num_filters": 128,
    "filter_sizes": "3,4,5",
    "embedding_dim": 64,
    "l2_reg_lambda": 0.0,
    "evaluate_every": 100,
    "dropout_keep_prob": 0.5,
    "checkpoint_every": 10}


def retrive_data(path, name=""):
    indices = []
    text = []
    rating = []
    i = 0
    for filename in os.listdir(path + "pos"):
        try:
            data = open(path + "pos/" + filename, 'r').read()
            data = data_helpers.clean_str(data)
            indices.append(i)
            text.append(data)
            rating.append([0, 1])
            i += 1
        except:
            pass

    for filename in os.listdir(path + "neg"):
        try:
            data = open(path + "neg/" + filename, 'r').read()
            indices.append(i)
            text.append(data)
            rating.append([1, 0])
            i += 1
        except:
            pass
    '''
    Dataset = list(zip(text,rating))
    df = pd.DataFrame(data=Dataset, columns=['text', "rating"])
    df.to_csv(name, index=False, header=True)
    '''
    return text, rating

if __name__ == "__main__":

    if not os.path.isfile("data.p") and not os.path.isfile("label.p"):
        xxx, yyy = retrive_data(path=train_path)
        #xxx_t, yyy_t = retrive_data(path=test_path, name="imdb_test.csv")

        Xtrain_text = xxx
        Ytrain = yyy

        output = open("data.p", "wb")
        pickle.dump(xxx, output)

        output = open("label.p", "wb")
        pickle.dump(yyy, output)
    else:
        print("load data")
        Xtrain_text = pickle.load(open("data.p", "rb"))
        Ytrain = pickle.load(open("label.p", "rb"))
        print("data loaded")

    # get the training data.
    # data = pd.read_csv("imdb_train.csv", header=0)


    # Build vocabulary
    max_document_length_train = max([len(x.split(" ")) for x in Xtrain_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length_train,min_frequency=25000)
    x = np.array(list(vocab_processor.fit_transform(Xtrain_text)))
    y = np.array(list(Ytrain))

    print(vocab_processor.vocabulary_._mapping)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(Ytrain)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.2)

    del x, y, Xtrain_text, Ytrain, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print(len(y_train))

    # Training
    # =================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'],
                l2_reg_lambda=params['l2_reg_lambda'])

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params["dropout_keep_prob"]
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch ,writer = None):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy],feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)


            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), params["batch_size"], params["num_epochs"])
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % params["evaluate_every"] == 0:
                    print("\nEvaluation:")
                    dev_batches = data_helpers.dev_batch_iter(list(zip(x_dev, y_dev)), 500)
                    x_dev, y_dev =  zip(*dev_batches)
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % params["checkpoint_every"] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
