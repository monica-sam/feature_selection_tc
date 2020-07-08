# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:55:07 2020

@author: sam6
"""


#!/usr/bin/env python

"""Train a document classifier."""

import reuters
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import time
from sklearn.metrics import accuracy_score, fbeta_score
import pdb
import random


# dataset_module = reuters

def evaluate(dataset_module, candidate_param):
    """
    Train a classifier for a dataset.

    Parameters
    ----------
    categories : list of str
    document_ids : list of str
    """
    # Calculate feature vectors
    data = dataset_module.load_data()

    # pdb.set_trace()
    
    xs = {'train': data['x_train'], 'test': data['x_test']}
    ys = {'train': data['y_train'], 'test': data['y_test']}
    
    # pdb.set_trace()
    # rand_train = [random.randrange(1, len(data['x_train']), 1) for i in range(1000)] 
    # rand_test = [random.randrange(1, len(data['x_test']), 1) for i in range(100)] 

    # partial_train_x = data['x_train'][rand_train]
    # partial_train_y = data['y_train'][rand_train]
    
    # partial_test_x = data['x_test'][rand_test]
    # partial_test_y = data['y_test'][rand_test]
    
    partial_train_x = data['x_train'][0:1000][::]
    partial_train_y = data['y_train'][0:1000][::]

    partial_test_x = data['x_test'][0:100][::]
    partial_test_y = data['y_test'][0:100][::]
    
    candidate_param = int(candidate_param)
    
    clf_name = 'Logistic Regression (C='+str(candidate_param)+')'
    # Get classifiers
    classifier = OneVsRestClassifier(LogisticRegression(C=candidate_param))
##        ('Logistic Regression (C=1000)',
##         OneVsRestClassifier(LogisticRegression(C=10000))),
        # ('k nn 3', KNeighborsClassifier(3)),
        # ('k nn 5', KNeighborsClassifier(5)),
        # ('Naive Bayes', OneVsRestClassifier(GaussianNB())),
        # ('SVM, linear', OneVsRestClassifier(SVC(kernel="linear",
        #                                         C=0.025,
        #                                         cache_size=200))),
        # ('SVM, adj.', OneVsRestClassifier(SVC(probability=False,
        #                                       kernel="rbf",
        #                                       C=2.8,
        #                                       gamma=.0073,
        #                                       cache_size=200))),
        # ('AdaBoost', OneVsRestClassifier(AdaBoostClassifier())),  # 20 minutes to train
        # ('LDA', OneVsRestClassifier(LinearDiscriminantAnalysis())),  # took more than 6 hours
        # ('RBM 100', Pipeline(steps=[('rbm', BernoulliRBM(n_components=100)),
        #                             ('logistic', LogisticRegression(C=1))])),
        # ('RBM 100, n_iter=20',
        #  Pipeline(steps=[('rbm', BernoulliRBM(n_components=100, n_iter=20)),
        #                  ('logistic', LogisticRegression(C=1))])),
        # ('RBM 256', Pipeline(steps=[('rbm', BernoulliRBM(n_components=256)),
        #                             ('logistic', LogisticRegression(C=1))])),
        # ('RBM 512, n_iter=100',
        #  Pipeline(steps=[('rbm', BernoulliRBM(n_components=512, n_iter=10)),
        #                  ('logistic', LogisticRegression(C=1))])),


    print(("{clf_name:<30}: {score:<5}  in {train_time:>5} /  {test_time}")
          .format(clf_name="Classifier",
                  score="score",
                  train_time="train",
                  test_time="test"))
    print("-" * 70)
##    for clf_name, classifier in classifiers:
##        t0 = time.time()
##        classifier.fit(xs['train'], ys['train'])
##        t1 = time.time()
##        # score = classifier.score(xs['test'], ys['test'])
##        preds = classifier.predict(data['x_test'])
##        preds[preds >= 0.5] = 1
##        preds[preds < 0.5] = 0
##        t2 = time.time()
##        # res = get_tptnfpfn(classifier, data)
##        # acc = get_accuracy(res)
##        # f1 = get_f_score(res)
##        acc = accuracy_score(y_true=data['y_test'], y_pred=preds)
##        f1 = fbeta_score(y_true=data['y_test'], y_pred=preds, beta=1, average="weighted")
##        print(("{clf_name:<30}: {acc:0.2f}% {f1:0.2f}% in {train_time:0.2f}s"
##               " train / {test_time:0.2f}s test")
##              .format(clf_name=clf_name,
##                      acc=(acc * 100),
##                      f1=(f1 * 100),
##                      train_time=t1 - t0,
##                      test_time=t2 - t1))
##        # print("\tAccuracy={}\tF1={}".format(acc, f1))


    t0 = time.time()
    classifier.fit(partial_train_x, partial_train_y)
    t1 = time.time()
    # score = classifier.score(xs['test'], ys['test'])
    preds = classifier.predict(partial_test_x)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    t2 = time.time()
    # res = get_tptnfpfn(classifier, data)
    # acc = get_accuracy(res)
    # f1 = get_f_score(res)
    acc = accuracy_score(y_true=partial_test_y, y_pred=preds)
    f1 = fbeta_score(y_true=partial_test_y, y_pred=preds, beta=1, average="weighted")
    print(("{clf_name:<30}: {acc:0.2f}% {f1:0.2f}% in {train_time:0.2f}s"
           " train / {test_time:0.2f}s test")
          .format(clf_name=clf_name,
                  acc=(acc * 100),
                  f1=(f1 * 100),
                  train_time=t1 - t0,
                  test_time=t2 - t1))
    # print("\tAccuracy={}\tF1={}".format(acc, f1))    
    
    return ((acc * 100) + (f1 * 100)) / 2


# ##if __name__ == '__main__':
# ##    main(reuters)
# reuters.load_data()



















































