#################
# Configuration #
#################

import os
import pandas as pd

import visualization
import imbalance
import classification

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

working_directory = os.getcwd() + "/"
input_file = working_directory + "data_for_student_case.csv"
figure_directory = working_directory + "figure/"
figure_white_box = figure_directory + "tree.dot"

data_set = pd.read_csv(input_file)

# Create figure directory
if not os.path.exists(figure_directory):
    print('Creating directory for figures...')
    os.makedirs(figure_directory)


######################
# Data understanding #
######################

print("Number of columns: {}".format(len(data_set.columns)))
data_set.info()
data_set.head()
data_set.describe()

for feature in ['txvariantcode', 'currencycode', 'shopperinteraction', 'simple_journal']:
    print(data_set[feature].value_counts())
    print('---------------------------------------------')
    print('')


#######################
# Data pre-processing #
#######################

# Clean the data
data_set = data_set.dropna()
data_set = data_set[data_set.simple_journal != "Refused"]


# Change data type
for column in ['bookingdate', 'creationdate']:
    data_set[column] = pd.to_datetime(data_set.bookingdate, format='%Y-%m-%d %H:%M:%S', errors='coerce')

for column in ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
               'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']:
    data_set[column] = data_set[column].astype('category')


######################
# Visualization task #
######################

# Heat map
visualization.heatmap(data_set, 'Chargeback', 'Issuer-Shopper Country Code of Chargeback Transactions',
                      figure_directory + 'heatmap_chargeback.png')
visualization.heatmap(data_set, 'Settled', 'Issuer-Shopper Country Code of Settled Transactions',
                      figure_directory + 'heatmap_settled.png')


# Box plot
visualization.boxplot(data_set, 'Amount Distribution of Chargeback and Settled Transactions',
                      figure_directory + 'boxplot_amount.png')


##################
# Imbalance task #
##################

# Data preparation
subset = data_set[['issuercountrycode', 'txvariantcode', 'amount', 'currencycode', 'shoppercountrycode',
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

oversampling = SMOTE(ratio=float(0.5), random_state=42)
feature_oversampling, label_oversampling = oversampling.fit_sample(feature_train, label_train)


# Generate ROC curves with different classifiers
imbalance.compare_roc(LogisticRegression(), 'AUC of Logistic Classifier', figure_directory + 'roc_logistic.png',
                      feature_train, feature_test, feature_oversampling, label_train, label_test, label_oversampling)

imbalance.compare_roc(tree.DecisionTreeClassifier(), 'AUC of Decision Tree Classifier',
                      figure_directory + 'roc_decision_tree.png', feature_train, feature_test, feature_oversampling,
                      label_train, label_test, label_oversampling)

imbalance.compare_roc(KNeighborsClassifier(n_neighbors=5), 'AUC of KNN Classifier', figure_directory + 'roc_knn.png',
                      feature_train, feature_test, feature_oversampling, label_train, label_test, label_oversampling)


#######################
# Classification task #
#######################

# Data preparation
columns = ['issuercountrycode', 'txvariantcode', 'amount', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
           'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode', 'simple_journal']
subset = data_set[columns]
print("Number of columns: {}".format(len(subset.columns)))

subset.loc[subset.simple_journal == 'Chargeback', 'simple_journal'] = 1
subset.loc[subset.simple_journal == 'Settled', 'simple_journal'] = 0
subset['simple_journal'] = subset['simple_journal'].astype('int')

label = subset.simple_journal

feature = subset.drop('simple_journal', axis=1)
feature = pd.get_dummies(feature)


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
tp, fp, tn, fn, auc = classification.cross_validation(algorithm, feature, label, 10)
classification.evaluation_result(tp, fp, tn, fn, auc)


# Visualize the white-box
algorithm = tree.DecisionTreeClassifier()
algorithm.fit(feature, label)
tree.export_graphviz(algorithm, out_file=figure_white_box, max_depth=3)   # limit the depth for pretty visualization

# After that run below line on UNIX command line to export the dot file to png
# dot -Tpng tree.dot -o tree.png
