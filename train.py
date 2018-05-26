import datetime
import operator
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from data_helpers import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from text_cnn import TextCNN

train_path = "./"  # source data

# Parameters
# ==================================================
params = {
    "num_epochs": 150,
    "batch_size": 128,
    "num_filters": 128,
    "filter_sizes": "3,4,5",
    "embedding_dim": 128,
    "evaluate_every": 2,
    "checkpoint_every": 100}



def retrive(source=""):
    df = pd.read_csv(source, header=None)
    pos_text = []
    neg_text = []
    pos_y = []
    neg_y = []

    for i, label in enumerate(df[df.columns[2]]):
        if label == 0:
            pos_text.append(df[df.columns[1]][i])
            pos_y.append([0, 1])

        elif label == 1:
            neg_text.append(df[df.columns[1]][i])
            neg_y.append([1, 0])

        else:
            print("something wrong")
    return pos_text, pos_y, neg_text, neg_y


def tensor_graph_train(x_train, y_train, x_val, y_val, drop = .5 , l2=3.0, hyperparameters = params, eval=False , x_test=None, y_test=None):
    '''

    :param x_train: train data
    :param y_train: train label
    :param x_val: validation data : used for validation and stoping condition
    :param y_val: validation label
    :param drop: keep dorp out value defult = .5
    :param l2: l2 regularization
    :param params: dic
    :param x_test: testset data  optional
    :param y_test: testset lable optional
    :param eval: defult false if no testset, change to True to evaluate the perfourmance over the testset
    :return: accuracy , precision , recall for the validation set, if the eval True  accuracy , precision , recall for the testset
    '''
    dev_step_n = 0
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
                l2_reg_lambda=l2)

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
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),
                                                         tf.nn.zero_fraction(g))
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
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

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
                    cnn.dropout_keep_prob: drop
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 20 == 0:
                    print("l2: {} drop: {} # {}: step {}, loss {:g}, acc {:g}".format(l2, drop,time_str, step, loss, accuracy))
                if step % 2 == 0:
                    train_summary_writer.add_summary(summaries, step)
                return accuracy

            def dev_step(x_batch, y_batch, name="", writer=None):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, ], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 50 == 0:
                    print("\nEvaluation:")
                    print("l2: {} drop: {} # {}: step {}, loss {:g}, acc for {}  {:g}".format(l2, drop, time_str, step, loss, name, accuracy))
                    print("")
                    if writer:
                        writer.add_summary(summaries, step)
                return loss, accuracy

            batches = batch_iter(list(zip(x_train, y_train)), params["batch_size"], params["num_epochs"])
            val_loss_windows = []

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                trainacc = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % params["evaluate_every"] == 0:
                    val_loss, valacc = dev_step(x_val, y_val, name="validation", writer=dev_summary_writer)
                    val_loss_windows.append(val_loss)
                    dev_step_n += 1

                    if dev_step_n % 2 == 0 and dev_step_n >= 600:
                        if dev_step_n % 10 == 0:
                            print("check if still learning ..... ")
                        med_first = np.median(
                            val_loss_windows[len(val_loss_windows) - 600:len(val_loss_windows) - 300])
                        med_second = np.median(
                            val_loss_windows[len(val_loss_windows) - 300:len(val_loss_windows)])

                        if (med_first - med_second) < .0005:
                            print("the stopping condition achieved  ..... ")
                            print("Median: ", med_first, med_second, med_first - med_second)
                            dev_step_n = 0
                            val_loss_windows = []
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                            break
                break
                if current_step % params["checkpoint_every"] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))



            #feed_dict_train = {cnn.input_x: x_train, cnn.input_y: y_train, cnn.dropout_keep_prob: 1.0}
            #trainacc, predictions_train = sess.run([cnn.accuracy, cnn.predictions], feed_dict_train)
            #precision_t, recall_t = precision_recall(predictions_train, y_train)

            if eval:
                print("compute performance over Test Set ....")
                feed_dict_test = {cnn.input_x: x_test, cnn.input_y: y_test, cnn.dropout_keep_prob: 1.0}
                testacc, predictions_test = sess.run([cnn.accuracy, cnn.predictions], feed_dict_test)
                precision_test, recall_test = precision_recall(predictions_test, y_test)
                print("re-initialize .... ")
                train_summary_writer.close()
                dev_summary_writer.close()
                sess.run(tf.global_variables_initializer())
                return testacc, precision_test, recall_test

            else:
                print("compute performance over validation Set ....")
                feed_dict_val = {cnn.input_x: x_val, cnn.input_y: y_val, cnn.dropout_keep_prob: 1.0}
                valacc, predictions_val = sess.run([cnn.accuracy, cnn.predictions], feed_dict_val)
                precision_val, recall_val = precision_recall(predictions_val, y_val)
                print("re-initialize .... ")
                train_summary_writer.close()
                dev_summary_writer.close()
                sess.run(tf.global_variables_initializer())
                return valacc, precision_val, recall_val


if __name__ == "__main__":

    # retrive the data
    pos_text, pos_y, neg_text, neg_y = retrive(source="./data.csv")

    x_pos = np.array(list(pos_text))
    y_pos = np.array(list(pos_y))
    x_neg = np.array(list(neg_text))
    y_neg = np.array(list(neg_y))

    dropout = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    grid_accuracy = {}
    grid_precision = {}
    grid_recall = {}
    val_accuracy_avarage = {}
    overall_performance = []
    overall_precision = []
    overall_recall = []
    ######################### Estimate Performance for the Model ############################

    # 10 folds outer_loop
    kf_pos_outer = KFold(n_splits=10, shuffle=True)
    kf_neg_outer = KFold(n_splits=10, shuffle=True)
    kf_pos_outer.get_n_splits(x_pos)
    kf_neg_outer.get_n_splits(x_neg)
    kflod_outer_neg_indices = list(kf_neg_outer.split(x_neg))
    for outer_fold_n, [non_holdout_outer_folds_p, holdout_outer_fold_p] in enumerate(kf_pos_outer.split(x_pos)):

        non_holdout_data_pos, non_holdout_label_pos, holdout_data_pos, holdout_label_pos = x_pos[
                                                                                               non_holdout_outer_folds_p], \
                                                                                           y_pos[
                                                                                               non_holdout_outer_folds_p], \
                                                                                           x_pos[holdout_outer_fold_p], \
                                                                                           y_pos[holdout_outer_fold_p]

        non_holdout_outer_folds_n, holdout_outer_fold_n = kflod_outer_neg_indices[outer_fold_n]
        non_holdout_data_neg, non_holdout_label_neg, holdout_data_neg, holdout_label_neg = x_neg[
                                                                                               non_holdout_outer_folds_n], \
                                                                                           y_neg[
                                                                                               non_holdout_outer_folds_n], \
                                                                                           x_neg[holdout_outer_fold_n], \
                                                                                           y_neg[holdout_outer_fold_n]
        # 10 folds inner_loop
        kf_pos_inner = KFold(n_splits=10, shuffle=True)
        kf_neg_inner = KFold(n_splits=10, shuffle=True)
        kf_pos_inner.get_n_splits(non_holdout_data_pos)
        kf_neg_inner.get_n_splits(non_holdout_data_neg)
        kflod_inner_neg_indices = list(kf_neg_inner.split(non_holdout_data_neg))

        for drop in dropout:
            hp_pair = "{}".format(str(drop))
            grid_accuracy[hp_pair] = []
            grid_precision[hp_pair] = []
            grid_recall[hp_pair] = []

        for inner_fold_n, [inner_training_folds_p, inner_val_fold_p] in enumerate(
                kf_pos_inner.split(non_holdout_data_pos)):

            inner_train_data_pos, inner_train_label_pos, inner_val_data_pos, inner_val_label_pos = \
                non_holdout_data_pos[inner_training_folds_p], \
                non_holdout_label_pos[inner_training_folds_p], \
                non_holdout_data_pos[inner_val_fold_p], \
                non_holdout_label_pos[inner_val_fold_p]

            inner_training_folds_n, inner_val_fold_n = kflod_inner_neg_indices[inner_fold_n]

            inner_train_data_neg, inner_train_label_neg, inner_val_data_neg, inner_val_label_neg = \
                non_holdout_data_neg[inner_training_folds_n], \
                non_holdout_label_neg[inner_training_folds_n], \
                non_holdout_data_neg[inner_val_fold_n], \
                non_holdout_label_neg[inner_val_fold_n]

            x_train = np.concatenate([inner_train_data_pos, inner_train_data_neg], 0)
            y_train = np.concatenate([inner_train_label_pos, inner_train_label_neg], 0)

            x_val = np.concatenate([inner_val_data_pos, inner_val_data_neg], 0)
            y_val = np.concatenate([inner_val_label_pos, inner_val_label_neg], 0)

            # Build vocabulary
            vocab_processor, x_train, y_train = fit_and_transform(x_train, y_train, fit=True)
            x_val, y_val = fit_and_transform(x_val, y_val, vocab_proc=vocab_processor)

            print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
            print("Train/Val split: {:d}/{:d}".format(len(y_train), len(y_val)))
            print("starts on outer fold No. {}  and innerfold No. {}".format(outer_fold_n, inner_fold_n))

            for drop in dropout:
                hp_pair = "{}".format(str(drop))
                valacc, precision_val, recall_val = tensor_graph_train(x_train, y_train, x_val, y_val, drop = drop)
                grid_accuracy[hp_pair].append(valacc)
                grid_precision[hp_pair].append(precision_val)
                grid_recall[hp_pair].append(recall_val)
                print("with keep dropout:{}   the accuracy was:{} ".format(drop , valacc))

        for drop_param in grid_accuracy:
            print("validation set accuracy for [ {} ] :".format(drop_param), grid_accuracy[drop_param])
            val_accuracy_avarage[drop_param] = np.average(grid_accuracy[drop_param])

        best_drop_str = max(val_accuracy_avarage.items(), key=operator.itemgetter(1))[0]
        best_drop =float(best_drop_str)
        print("Best parameters in the {} outer fold is : {} = {}".format(outer_fold_n, best_drop,grid_accuracy[best_drop_str]))

        x_train_pos, x_dev_pos, y_train_pos, y_dev_pos = train_test_split(non_holdout_data_pos, non_holdout_label_pos,
                                                                          test_size=.15, shuffle=True)
        x_train_neg, x_dev_neg, y_train_neg, y_dev_neg = train_test_split(non_holdout_data_neg, non_holdout_label_neg,
                                                                          test_size=.15, shuffle=True)

        x_train = np.concatenate([x_train_pos, x_train_neg], 0)
        y_train = np.concatenate([y_train_pos, y_train_neg], 0)

        x_dev = np.concatenate([x_dev_pos, x_dev_neg], 0)
        y_dev = np.concatenate([y_dev_pos, y_dev_neg], 0)

        x_test = np.concatenate([holdout_data_pos, holdout_data_neg], 0)
        y_test = np.concatenate([holdout_label_pos, holdout_label_neg], 0)

        # Build vocabulary
        vocab_processor, x_train, y_train = fit_and_transform(x_train, y_train, fit=True)
        x_dev, y_dev = fit_and_transform(x_dev, y_dev, vocab_proc=vocab_processor)
        x_test, y_test = fit_and_transform(x_test, y_test, vocab_proc=vocab_processor)


        print("compute the performance for the outerfold No. {}  ...".format(outer_fold_n))
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/dev/test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))


        # compute perfurmance over the best param
        holdoutacc, holdout_precision, holdout_recall = tensor_graph_train(x_train, y_train, x_dev, y_dev, drop = best_drop, x_test=x_test, y_test=y_test, eval=True)
        print("compute performance for the outer holdout  //  dropout:{}   the accuracy is:{} ".format(best_drop, holdoutacc))

        overall_performance.append(holdoutacc)
        print(" overall_performance  = ",overall_performance)

        overall_precision.append(holdout_precision)
        print(" overall_precision  = ",overall_precision)

        overall_recall.append(holdout_recall)
        print(" overall_recall  = ",overall_recall)

    print("#################################################################################")
    print("#################################################################################")

    print("estimate of the performance  : ", np.average(overall_performance))
    print("estimate of the precision  : ", np.average(overall_precision))
    print("estimate of the recall  : ", np.average(overall_recall))

    print("#################################################################################")
    print("#################################################################################")

    print("select the best ")

    ######################### SELECT THE BEST DROPOUT ############################

    grid_accuracy = {}
    grid_precision = {}
    grid_recall = {}
    val_accuracy_avarage = {}
    val_precision_avarage = {}
    val_recall_avarage = {}


    for drop in dropout:
        hp_pair = "{}".format(str(drop))
        grid_accuracy[hp_pair] = []
        grid_precision[hp_pair] = []
        grid_recall[hp_pair] = []

    best_param_acc = []
    best_param_recall = []
    best_param_precision = []

    # 10 folds outer_loop
    kf_pos_outer = KFold(n_splits=10, shuffle=True)
    kf_neg_outer = KFold(n_splits=10, shuffle=True)
    kf_pos_outer.get_n_splits(x_pos)
    kf_neg_outer.get_n_splits(x_neg)
    kflod_outer_neg_indices = list(kf_neg_outer.split(x_neg))
    for fold_n, [non_holdout_outer_folds_p, holdout_outer_fold_p] in enumerate(kf_pos_outer.split(x_pos)):
        non_holdout_data_pos, non_holdout_label_pos, holdout_data_pos, holdout_label_pos = x_pos[
                                                                                               non_holdout_outer_folds_p], \
                                                                                           y_pos[
                                                                                               non_holdout_outer_folds_p], \
                                                                                           x_pos[holdout_outer_fold_p], \
                                                                                           y_pos[holdout_outer_fold_p]

        non_holdout_outer_folds_n, holdout_outer_fold_n = kflod_outer_neg_indices[fold_n]
        non_holdout_data_neg, non_holdout_label_neg, holdout_data_neg, holdout_label_neg = x_neg[
                                                                                               non_holdout_outer_folds_n], \
                                                                                           y_neg[
                                                                                               non_holdout_outer_folds_n], \
                                                                                           x_neg[holdout_outer_fold_n], \
                                                                                           y_neg[holdout_outer_fold_n]

        x_train = np.concatenate([non_holdout_data_pos, non_holdout_data_neg], 0)
        y_train = np.concatenate([non_holdout_label_pos, non_holdout_label_neg], 0)

        x_val = np.concatenate([holdout_data_pos, holdout_data_neg], 0)
        y_val = np.concatenate([holdout_label_pos, holdout_label_neg], 0)

        # Build vocabulary
        vocab_processor, x_train, y_train = fit_and_transform(x_train, y_train, fit=True)
        x_val, y_val = fit_and_transform(x_val, y_val, vocab_proc=vocab_processor)

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Val split: {:d}/{:d}".format(len(y_train), len(y_val)))
        print("starts on fold No. {}".format(fold_n))

        for drop in dropout:
            hp_pair = "{}".format(str(drop))
            valacc, precision_val, recall_val = tensor_graph_train(x_train, y_train, x_val, y_val, drop = drop)
            print("compute performance for the holdout  //  dropout:{}   the accuracy is:{} ".format(drop,valacc))

            grid_accuracy[hp_pair].append(valacc)
            grid_precision[hp_pair].append(valacc)
            grid_recall[hp_pair].append(valacc)


    for drop_param in grid_accuracy:
        print("accuracy for [ {} ] :".format(drop_param), grid_accuracy[drop_param])
        val_accuracy_avarage[drop_param] = np.average(grid_accuracy[drop_param])


    for drop_param in grid_accuracy:
        print("the average accuracy for [ {} ] :".format(drop_param), val_accuracy_avarage[drop_param])




    for drop_param in grid_precision:
        print("precision for [ {} ] :".format(drop_param), grid_precision[drop_param])
        val_precision_avarage[drop_param] = np.average(grid_precision[drop_param])


    for drop_param in grid_precision:
        print("the average precision for [ {} ] :".format(drop_param), val_precision_avarage[drop_param])



    for drop_param in grid_recall:
        print("recall for [ {} ] :".format(drop_param), grid_recall[drop_param])
        val_recall_avarage[drop_param] = np.average(grid_recall[drop_param])


    for drop_param in grid_recall:
        print("the average recall for [ {} ] :".format(drop_param), val_recall_avarage[drop_param])



