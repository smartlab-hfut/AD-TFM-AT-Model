import os
import sys
sys.path.append('src/')

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MaxAbsScaler
from tfTFM import AdaTFM
from sklearn import metrics
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EPOCHS = 800
BATCH = 256


def get_one_hot(number, digits=5):
    one_hot = [0] * digits
    one_hot[number] = 1

    return one_hot


def data_random_shuffle(x, y):
    data_num1, _, _ = x.shape
    index1 = np.arange(data_num1)
    np.random.shuffle(index1)
    x1 = x[index1]
    y1 = y[index1]
    return x1, y1


if __name__ == '__main__':

    data_dir_path = './data'
    train_data = pd.read_csv(data_dir_path + '/train_data.csv', header=None)
    test_data = pd.read_csv(data_dir_path + '/test_data.csv', header=None)
    train_np_data = train_data.values
    test_np_data = test_data.values
    train_np_data = train_np_data.T
    test_np_data = test_np_data.T

    scaler = MaxAbsScaler()
    train_np_data = scaler.fit_transform(train_np_data)
    test_np_data = scaler.fit_transform(test_np_data)
    train_np_data = train_np_data.T
    test_np_data = test_np_data.T

    train_np_data = train_np_data.reshape(1385, 100, 6)
    test_np_data = test_np_data.reshape(345, 100, 6)
    train_label = pd.read_csv(data_dir_path + '/train_label.csv', header=None)
    test_label = pd.read_csv(data_dir_path + '/test_label.csv', header=None)
    test_label_np_data = test_label.values
    train_label_np_data = train_label.values

    train_np_data, train_label_np_data = data_random_shuffle(train_np_data, train_label_np_data)
    test_np_data, test_label_np_data = data_random_shuffle(test_np_data, test_label_np_data)

    train_label_np_data_2 = np.squeeze(train_label_np_data)
    test_label_np_data_2 = np.squeeze(test_label_np_data)


    train_label_np_data_3 = [get_one_hot(x) for x in train_label_np_data_2]
    test_label_np_data_3 = [get_one_hot(x) for x in test_label_np_data_2]
    x_train, y_train, x_test, y_test = train_np_data, train_label_np_data_3, test_np_data, test_label_np_data_3

    print(x_train.shape[1])
    print(x_train.shape[2])



    unit_size = 32
    omega0 = 16
    K0 = 4
    J0 = 4

    tfm = AdaTFM(state_size=unit_size,
                 input_size=x_train.shape[1],
                 dim_size=x_train.shape[2],
                 omega=omega0,
                 target_size=5,
                 K=K0, J=J0)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            loss = 0
            mini_start = 0
            mini_end = BATCH

            for i in range(0, x_train.shape[0], BATCH):
                batch_X = x_train[mini_start:mini_end]
                batch_Y = y_train[mini_start:mini_end]

                mini_start = mini_end
                mini_end = mini_start + BATCH

                loss_, _ = sess.run([tfm.loss, tfm.train_op],
                                    feed_dict={tfm._inputs: batch_X,
                                               tfm.ys: batch_Y})

                loss += loss_
                if i % 4096 == 0:
                    train_loss, train_accuracy = sess.run([tfm.loss, tfm.accuracy],
                                             feed_dict={tfm._inputs: batch_X,
                                                        tfm.ys: batch_Y})
                    print("Batch: %s train loss: %s Train Accuracy: %s" % (i, train_loss, train_accuracy))

            test_loss, test_accuracy = sess.run([tfm.loss, tfm.accuracy],
                                     feed_dict={tfm._inputs: x_test,
                                                tfm.ys: y_test})
            print("Epoch: %s test loss: %s test Accuracy: %s" % (epoch, test_loss, test_accuracy))
            accuracy, y_pred, y_true, y_score = sess.run([tfm.accuracy, tfm.model_pred, tfm.model_True, tfm.y_score],
                                                feed_dict={tfm._inputs: x_test,
                                                           tfm.ys: y_test})


            target_names = ['SGF', 'TGF', 'IPSF', 'MTF',
                            'Normal']
            class_report = classification_report(y_true, y_pred, target_names=target_names)

            Y_valid = y_true
            y_valid = label_binarize(Y_valid, classes=[i for i in range(5)])
            y_pred1 = label_binarize(y_pred, classes=[i for i in range(5)])
            nb_classes = 5
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(nb_classes):
                fpr[i], tpr[i], dfs = roc_curve(y_valid[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(nb_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                # Finally average it and compute AUC
            mean_tpr /= nb_classes
            fpr["micro"] = all_fpr
            tpr["micro"] = mean_tpr
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Plot all ROC curves
            lw = 2
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                        label='Average (AUC = {0:0.2f})'
                            ''.format(roc_auc["micro"]),
                        color='deeppink', linestyle=':', linewidth=2)
            colors = cycle(['green', 'bisque', 'salmon', 'sienna', 'plum'])
            for i, color in zip(range(nb_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                            label=target_names[i] + '(AUC = {1:0.2f})'
                                                    ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig("../ROC_GW_ADATall_final.eps")
            if epoch == EPOCHS - 1:
                print(class_report)
                plt.show()

        saver = tf.train.Saver()
        saver.save(sess, 'net/ tfm.ckpt')
        print("model has saved,model format is saved_model !")