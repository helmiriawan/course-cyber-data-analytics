# Lab 1 - Fraud Detection

There are 3 tasks for this lab assignment:
- Visualization task
- Imbalance task
- Classification task

## Data set
Data set for this assignment is `data_for_student_case.csv`. It consists of
anonymous credit card transactions, some of them are fraudulent.

## Visualization task
There are several interesting relationships in the data set when comparing
fraudulent and non-fraudulent transactions. Most of the shopper and issuer
country code combinations on fraudulent (chargeback) transactions are MX-MX and
AU-AU. It is different compared with non-fraudulent (settled) transactions,
since most of the combinations are GB-GB, as can be seen in two figures below.

![Fraudulent-transactions](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/heatmap_chargeback.png)

![Non-fraudulent-transactions](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/heatmap_settled.png)

Furthermore, we also find that the amount distribution between fraudulent and non-fraudulent transactions is also different. In general, the amount of
non-fraudulent transactions is lower than fraudulent transactions, as can be
seen in figure below.

![Non-fraudulent-transactions](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/boxplot_amount.png)


## Imbalance task
The number of fraudulent transactions is very small compared with non-fraudulent
transactions, which is only 0.15% from the whole data set. A classifier that
classifies all the data as non-fraudulent transactions can have accuracy of
99.85%, which means very good. However, since the goal of this task is to detect
fraudulent transactions, it would be better if we use different metrics, such as
[true positive rate](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
and
[false positive rate](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

There are several solutions that can be used to tackle imbalanced data. In this
lab, we use oversampling technique named [Synthetic Minority Over-sampling
Technique (SMOTE)](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)
in order to adjust the class distribution of the data set. We
try this technique and test it by using three different algorithms to classify
fraudulent transactions. The result shows that not all classifiers can have the
benefit of this technique, as can be seen in figure below.

![AUC-logistic](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/roc_logistic.png)

![AUC-decision-tree](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/roc_decision_tree.png)

![AUC-KNN](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/roc_knn.png)
