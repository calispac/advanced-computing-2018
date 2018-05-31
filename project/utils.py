import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from sklearn.metrics import confusion_matrix
from read import get_data
import numpy as np
from tqdm import tqdm


def print_score(model, train, val):

    score, acc = model.evaluate_generator(train)
    print('Train : ', score, acc)

    score, acc = model.evaluate_generator(val)
    print('Validation : ', score, acc)


def plot_history(histories, labels=[1, 2]):

    n_colors = len(histories)
    colors = cm.rainbow(np.linspace(0, 1, n_colors))

    fig_1 = plt.figure()
    fig_2 = plt.figure()

    axes_1 = fig_1.add_subplot(111)
    axes_2 = fig_2.add_subplot(111)

    for i, history in enumerate(histories):
        c = colors[i]
        label = labels[i]
        metrics = history
        epoch = np.arange(len(metrics['loss'])) + 1
        epoch = epoch / np.max(epoch)

        axes_1.semilogy(epoch, metrics['val_loss'], label='CNN {}'.format(label),
                        linestyle='--', color=c)
        axes_1.set_xlabel('epoch [a.u.]')
        axes_1.set_ylabel('categorical cross-entropy')
        axes_1.legend(loc='best')

        axes_2.semilogy(epoch, metrics['val_acc'], label='CNN {}'.format(label),
                        linestyle='--', color=c)
        axes_2.set_xlabel('epoch [a.u.]')
        axes_2.set_ylabel('accuracy')
        axes_2.legend(loc='best')


def plot_confusion_matrix(cf_matrix, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.magma):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if classes is None:

        classes = np.arange(len(cf_matrix))

    cf_matrix = cf_matrix.astype('int')

    if normalize:
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()
    plt.imshow(cf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    fmt = '.2f' if normalize else 'd'
    thresh = cf_matrix.max() / 2.
    # for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
    #    plt.text(j, i, format(cf_matrix[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cf_matrix[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_confusion_matrix(model, max_iter=10):

    _, val_generator, _, _ = get_data()

    n_classes = 29

    cf_matrix = np.zeros((n_classes, n_classes))
    i = 0

    for X, y in tqdm(val_generator):

        i += 1
        y_true = np.argmax(y, axis=-1)
        y_predict = model.predict(X)
        y_predict = np.argmax(y_predict, axis=-1)

        cf_matrix += confusion_matrix(y_true, y_predict,
                                      labels=np.arange(n_classes))

        if i > len(val_generator):
            break

    return cf_matrix


def compute_class_score_histogram(model, label=0, n_classes=29, max_iter=10):

    _, val_generator, _, _ = get_data()

    n_bins = 100
    bins = np.linspace(0, 1, num=n_bins + 1)

    histograms = np.zeros((n_classes, n_bins), dtype=int)

    count = 0

    for X, y in tqdm(val_generator):

        y_true = np.argmax(y, axis=-1)
        y_predict = model.predict(X)
        y_predict = y_predict[..., label]

        for i, class_index in enumerate(y_true):

            histograms[class_index] += np.histogram(y_predict[i], bins=bins)[0]

        count += 1

        if count > len(val_generator):

            break

    return histograms


def plot_histogram_score(histograms, label='0'):

    plt.figure()

    bins = histograms.shape[-1]
    bins = np.linspace(0, 1, num=bins)

    for i, histogram in enumerate(histograms):

        plt.step(bins, histogram, label='True class : {}'.format(i))
        plt.xlabel('class {} probability'.format(label))
        plt.ylabel('count')


if __name__ == '__main__':

    from keras.models import load_model

    model = load_model('model_6.h5')
    compute_class_score_histogram(model)