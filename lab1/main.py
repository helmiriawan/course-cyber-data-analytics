#################
# Configuration #
#################

import os
import numpy as np
import pandas as pd

import visualization
import imbalance

from sklearn import tree

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

working_directory = os.getcwd() + "/"
input_file = working_directory + "data_for_student_case.csv"
figure_directory = working_directory + "figure/"
figure_whitebox = figure_directory + "tree.dot"

dataset = pd.read_csv(input_file)

# Create figure directory
if not os.path.exists(figure_directory):
    print('Creating directory for figures...')
    os.makedirs(figure_directory)


######################
# Data understanding #
######################

print("Number of columns: {}".format(len(dataset.columns)))
dataset.info()
dataset.head()
dataset.describe()

for feature in ['txvariantcode', 'currencycode', 'shopperinteraction', 'simple_journal']:
    print(dataset[feature].value_counts())
    print('---------------------------------------------')
    print('')


#######################
# Data pre-processing #
#######################

# Clean the data
dataset = dataset.dropna()
dataset = dataset[dataset.simple_journal != "Refused"]


# Change data type
for column in ['bookingdate', 'creationdate']:
    dataset[column] = pd.to_datetime(dataset.bookingdate, format='%Y-%m-%d %H:%M:%S', errors='coerce')

for column in ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
               'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']:
    dataset[column] = dataset[column].astype('category')


######################
# Visualization task #
######################

# Heat map
visualization.heatmap(dataset, 'Chargeback', 'Issuer-Shopper Country Code of Chargeback Transactions',
                      figure_directory + 'heatmap_chargeback.png')
visualization.heatmap(dataset, 'Settled', 'Issuer-Shopper Country Code of Settled Transactions',
                      figure_directory + 'heatmap_settled.png')


# Box plot
visualization.boxplot(dataset, 'Amount Distribution of Chargeback and Settled Transactions',
                      figure_directory + 'boxplot_amount.png')


##################
# Imbalance task #
##################

# Data preparation
subset = dataset[['issuercountrycode', 'txvariantcode', 'amount', 'currencycode', 'shoppercountrycode',
                  'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode',
                  'simple_journal']]

subset.loc[subset.simple_journal == 'Chargeback', 'simple_journal'] = 1
subset.loc[subset.simple_journal == 'Settled', 'simple_journal'] = 0
subset['simple_journal'] = subset['simple_journal'].astype('int')

label = subset.simple_journal

feature = subset.drop('simple_journal', axis=1)
feature = pd.get_dummies(feature)

feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.5,
                                                                        random_state=42, stratify=label)

resampling = SMOTE(ratio=float(0.5), random_state=42)
feature_resampling, label_resampling = resampling.fit_sample(feature_train, label_train)


# Generate ROC curves with different classifiers
imbalance.compare_roc(LogisticRegression(), 'AUC of Logistic Classifier', figure_directory + 'roc_logistic.png',
                      feature_train, feature_test, feature_resampling, label_train, label_test, label_resampling)

imbalance.compare_roc(tree.DecisionTreeClassifier(), 'AUC of Decision Tree Classifier',
                      figure_directory + 'roc_decision_tree.png', feature_train, feature_test, feature_resampling,
                      label_train, label_test, label_resampling)

imbalance.compare_roc(KNeighborsClassifier(n_neighbors=5), 'AUC of KNN Classifier', figure_directory + 'roc_knn.png',
                      feature_train, feature_test, feature_resampling, label_train, label_test, label_resampling)


#######################
# Classification task #
#######################

# Data preparation
columns = ['issuercountrycode', 'txvariantcode', 'amount', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
           'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode', 'simple_journal']
subset = dataset[columns]
print("Number of columns: {}".format(len(subset.columns)))

subset.loc[subset.simple_journal == 'Chargeback', 'simple_journal'] = 1
subset.loc[subset.simple_journal == 'Settled', 'simple_journal'] = 0
subset['simple_journal'] = subset['simple_journal'].astype('int')

label = subset.simple_journal

feature = subset.drop('simple_journal', axis=1)
feature = pd.get_dummies(feature)


# Create function to run cross validation
def cross_validation(method, variable, flag, number_of_fold):

    # Initiate the K-fold
    k_fold = KFold(n_splits=number_of_fold, shuffle=True, random_state=42)

    # Initiate the variables
    all_true_positives = []
    all_false_positives = []
    all_true_negatives = []
    all_false_negatives = []
    all_auc = []

    for train_index, test_index in k_fold.split(variable):

        # Split train and test data
        variable_train, variable_test = variable.iloc[train_index], variable.iloc[test_index]
        flag_train, flag_test = flag.iloc[train_index], flag.iloc[test_index]

        # Apply SMOTE
        resample = SMOTE(ratio=float(0.18), random_state=42)
        variable_resample, flag_resample = resample.fit_sample(variable_train, flag_train)

        # Train classifier
#       method.fit(variable_train, flag_train)
        method.fit(variable_resample, flag_resample)

        # Evaluate the model
        flag_prediction = method.predict(variable_test)
        table_of_confusion = confusion_matrix(flag_test, flag_prediction, labels=[1, 0])

        true_positives = table_of_confusion[0][0]
        false_positives = table_of_confusion[1][0]
        true_negatives = table_of_confusion[1][1]
        false_negatives = table_of_confusion[0][1]

        flag_prediction_probability = method.predict_proba(variable_test)[:, 1]
        area_under_curve = roc_auc_score(flag_test, flag_prediction_probability)

        # Put the evaluation result to the list
        all_true_positives.append(true_positives)
        all_false_positives.append(false_positives)
        all_true_negatives.append(true_negatives)
        all_false_negatives.append(false_negatives)
        all_auc.append(area_under_curve)

    # Change the list to numpy array
    all_true_positives = np.array(all_true_positives)
    all_false_positives = np.array(all_false_positives)
    all_true_negatives = np.array(all_true_negatives)
    all_false_negatives = np.array(all_false_negatives)
    all_auc = np.array(all_auc)

    return all_true_positives, all_false_positives, all_true_negatives, all_false_negatives, all_auc


# Create function to show the evaluation result from cross validation
def evaluation_result(true_positives, false_positives, true_negatives, false_negatives):

    accuracy = (true_positives+true_negatives) / (true_positives+false_positives+true_negatives+false_negatives)
    sensitivity = true_positives / (true_positives+false_negatives)
    specificity = true_negatives / (false_positives+true_negatives)
    precision = true_positives / (true_positives+false_positives)
    f_measure = 2 * precision * sensitivity / (precision+sensitivity)

    print("True positives: {}".format(np.mean(true_positives)))
    print("False positives: {}".format(np.mean(false_positives)))
    print("True negatives: {}".format(np.mean(true_negatives)))
    print("False negatives: {}".format(np.mean(false_negatives)))

    print("Accuracy: {}".format(np.mean(accuracy)))
    print("Sensitivity: {}".format(np.mean(sensitivity)))
    print("Specificity: {}".format(np.mean(specificity)))
    print("Precision: {}".format(np.mean(precision)))
    print("F-measure: {}".format(np.mean(f_measure)))
    print("AUC: {}".format(np.mean(auc)))


# Set algorithm
algorithm = LogisticRegression()

"""
# Alternative black-box algorithms
algorithm = KNeighborsClassifier(n_neighbors=5)
algorithm = RandomForestClassifier()

# White-box algorithm
algorithm = tree.DecisionTreeClassifier()
"""

# Evaluate the classifier with cross validation
tp, fp, tn, fn, auc = cross_validation(algorithm, feature, label, 10)
evaluation_result(tp, fp, tn, fn)


# Visualize the white-box
algorithm = tree.DecisionTreeClassifier()
algorithm.fit(feature, label)
tree.export_graphviz(algorithm, out_file=figure_whitebox, max_depth=3)   # limit the depth for pretty visualization

# After that run below line on UNIX command line to export the dot file to png
# dot -Tpng tree.dot -o tree.png
