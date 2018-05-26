import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.contrib import learn



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def dev_batch_iter(data,bat_size=0):
    if bat_size == 0:
        bat_size = len(data)
        dev_bat = random.sample(data, bat_size)
        return dev_bat



def precision_recall(predictions, labels):
    true_labels = []
    for i, l in enumerate(labels):
        true_labels.append(list(l)[1])

    precision, recall, f, s = precision_recall_fscore_support(true_labels, predictions, pos_label=0, average='binary')

    return precision, recall



def fit_and_transform(data, label, vocab_proc=None, fit=False):
    '''
    :param data:
    :param label:
    :param vocab_proc: if only transform fit = false you must parse the VocabularyProcessor
    :param fit: fitting and transform
    :return: data, label if fit = false "defult"
    and vocab_proc, data, label if fit = True
    '''
    if fit:
        max_document_length = max([len(x.split(" ")) for x in data])
        vocab_proc = learn.preprocessing.VocabularyProcessor(max_document_length)
        data = np.array(list(vocab_proc.fit_transform(data)))
        label = np.array(list(label))
        shuffle_indices = np.random.permutation(np.arange(len(label)))
        data = data[shuffle_indices]
        label = label[shuffle_indices]
        return vocab_proc, data, label

    elif vocab_proc != None:
        data = np.array(list(vocab_proc.transform(data)))
        label = np.array(list(label))
        return data, label
    else:
        print("ERORR IF NOT FIT YOU vocab_proc SHOULD NOT BE NONE  ")

