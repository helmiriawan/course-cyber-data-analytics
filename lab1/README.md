# Lab 1 - Fraud Detection

There are 3 tasks for this lab assignment:
- Visualization task
- Imbalance task
- Classification task

## Dataset

Dataset for this assignment is obtained from Adyen, a global payment company. It is about anonymous credit card transactions, where some of them are fraudulent.

## Visualization task

There are several interesting relationships in the data when comparing fraudulent and non-fraudulent transactions. In this task, we show some interesting information that can be used to distinguish fraudulent and non-fraudulent transactions.

## Imbalance task

The number of fraudulent transactions is very small compared with non-fraudulent transactions, which is less than 1% of the whole dataset. A classifier that classifies all records as non-fraudulent transactions would have an accuracy of 99%, which seems very good. However, the goal of this task is to classify fraudulent transactions. Therefore we need to use other metrics, such as true positive rate and false positive rate.

## Classification task

In some cases, the interpretability of the classifier is important. In fraud detection cases, some customers might want to know the reason why their transactions are classified as fraudulent transactions. In other cases, performance is more important compared with the interpretability, where a more complex classifier might be preferred.
