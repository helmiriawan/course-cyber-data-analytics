# Lab 1 - Fraud Detection

There are 3 tasks for this lab assignment:
- Visualization task
- Imbalance task
- Classification task

## Dataset
Dataset for this assignment is `data_for_student_case.csv`. It consists of
anonymous credit card transactions, some of them are fraudulent.

## Visualization task
There are several interesting relationships in the dataset when comparing
fraudulent and non-fraudulent transactions. Most of the shopper and issuer
country code combinations on fraudulent (chargeback) transactions are MX-MX and
AU-AU. It is different compared with non-fraudulent (settled) transactions,
since most of the combinations are GB-GB.

![Fraudulent-transactions](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/heatmap_chargeback.png)

![Non-fraudulent-transactions](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/heatmap_settled.png)

The amount distribution between fraudulent and non-fraudulent transactions is
also different, as can be seen in figure below.

![Non-fraudulent-transactions](https://raw.githubusercontent.com/helmiriawan/CS4035/master/lab1/figure/boxplot_amount.png)
