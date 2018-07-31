import matplotlib.pyplot as plt
import numpy as np
import itertools
import random


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_batch(dataset, batch_size, total_num_el):
    '''
    Get the indices for the batches for training.
    if total_num_el = n x dataset.shape[0], then n is the number of epoch.
    ----
    Inputs:
    batch_size = Batch size, int
    total_num_el = How many sample in total to use, int
    ----
    Outputs:
    all_batches = List of indices for the batches, list(numpy.array)
    '''
    ori_num_elements = len(dataset)
    inda = np.arange(ori_num_elements)
    all_batches = []

    batch_nb = int(total_num_el / batch_size)
    one_epoch_batch_nb = int(ori_num_elements / batch_size)
    iter_nb = max(int(batch_nb / one_epoch_batch_nb), 1)

    for i in range(iter_nb + 1):
        random.shuffle(inda)
        new_batches = [inda[u:u + batch_size]
                       for u in range(0, len(dataset), batch_size)]
        if new_batches[-1].shape[0] == batch_size:
            all_batches += new_batches
        else:
            all_batches += new_batches[:-1]

    return all_batches
