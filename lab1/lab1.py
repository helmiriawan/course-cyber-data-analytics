#################
# Configuration #
#################

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

working_directory = os.getcwd() + "/"
input_file = working_directory + "data_for_student_case.csv"
figure_directory = working_directory + "figure/"
figure_whitebox = figure_directory + "tree.dot"

# Create figure directory
if not os.path.exists(figure_directory):
    print('Creating directory for figures...')
    os.makedirs(figure_directory)


######################
# Data understanding #
######################

dataset = pd.read_csv(input_file)

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
def generate_heatmap(flag, title, output_filename):

    # filter and aggregate data
    filtered_dataset = dataset[dataset['simple_journal'] == flag]
    aggregation_data = filtered_dataset.groupby(['shoppercountrycode', 'issuercountrycode']).size()\
        .reset_index(name='count')

    # filter country code for pretty visualization
    issuer_country = list(dataset.issuercountrycode[dataset.simple_journal == 'Chargeback'].unique())
    shopper_country = list(dataset.shoppercountrycode[dataset.simple_journal == 'Chargeback'].unique())
    aggregation_data = aggregation_data[aggregation_data['issuercountrycode'].isin(issuer_country)]
    aggregation_data = aggregation_data[aggregation_data['shoppercountrycode'].isin(shopper_country)]

    # generate pivot and heat map
    pivot_data = aggregation_data.pivot(index='shoppercountrycode', columns='issuercountrycode', values='count')
    sns.heatmap(pivot_data, cmap='viridis', linewidths=.5)
    plt.xlabel('Issuer Country Code')
    plt.ylabel('Shopper Country Code')
    plt.title(title)
    plt.savefig(figure_directory + output_filename)
    plt.show()


generate_heatmap('Chargeback', 'Issuer-Shopper Country Code of Chargeback Transactions', 'heatmap_chargeback.png')
generate_heatmap('Settled', 'Issuer-Shopper Country Code of Settled Transactions', 'heatmap_settled.png')


# Box plot
sns.boxplot(x="simple_journal", y="amount", data=dataset[dataset['amount'] <= 300000])   # for pretty visualization

plt.title('Amount Distribution of Chargeback and Settled Transactions')
plt.xlabel('')
plt.ylabel('Amount')

tick_value = [50000, 100000, 150000, 200000, 250000, 300000]
tick_label = ['50k', '100k', '150k', '200k', '250k', '300k']
plt.yticks(tick_value, tick_label)

plt.savefig(figure_directory + 'boxplot_amount.png')
plt.show()

"""
# Alternative boxplot
dataset[dataset['amount'] <= 300000].boxplot(column='amount', by='simple_journal', rot=60)
plt.show()
"""


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


# Function to generate ROC curve values
def roc_values(method, variable_train, variable_test, flag_train, flag_test):

    # Train classifier
    method.fit(variable_train, flag_train)

    # Generate the ROC curves values
    label_prediction_probability = method.predict_proba(variable_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(flag_test, label_prediction_probability)
    area_under_curve = roc_auc_score(label_test, label_prediction_probability)

    return false_positive_rate, true_positive_rate, area_under_curve


# Function to generate ROC curves
def roc_curves(title, filename, false_positive_rate, true_positive_rate, area_under_curve,
               false_positive_rate_smote, true_positive_rate_smote, area_under_curve_smote):

    plt.title(title)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
             label='AUC UNSMOTEd = %0.2f' % area_under_curve)
    plt.plot(false_positive_rate_smote, true_positive_rate_smote, color='green',
             label='AUC SMOTEd = %0.2f' % area_under_curve_smote)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.savefig(figure_directory + filename)
    plt.show()


# Generate ROC curves with logistic classifier
algorithm = LogisticRegression()
fpr, tpr, auc = roc_values(algorithm, feature_train, feature_test, label_train, label_test)
fpr_smote, tpr_smote, auc_smote = roc_values(algorithm, feature_resampling, feature_test, label_resampling, label_test)
roc_curves('AUC of Logistic Classifier', 'roc_logistic.png', fpr, tpr, auc, fpr_smote, tpr_smote, auc_smote)


# Generate ROC curves with KNN classifier
algorithm = KNeighborsClassifier(n_neighbors=5)
fpr, tpr, auc = roc_values(algorithm, feature_train, feature_test, label_train, label_test)
fpr_smote, tpr_smote, auc_smote = roc_values(algorithm, feature_resampling, feature_test, label_resampling, label_test)
roc_curves('AUC of KNN Classifier', 'roc_knn.png', fpr, tpr, auc, fpr_smote, tpr_smote, auc_smote)


# Generate ROC curves with decision tree classifier
algorithm = tree.DecisionTreeClassifier()
fpr, tpr, auc = roc_values(algorithm, feature_train, feature_test, label_train, label_test)
fpr_smote, tpr_smote, auc_smote = roc_values(algorithm, feature_resampling, feature_test, label_resampling, label_test)
roc_curves('AUC of Decision Tree Classifier', 'roc_decision_tree.png', fpr, tpr, auc, fpr_smote, tpr_smote, auc_smote)


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



# Set the classifier
classifier = LogisticRegression()                     # for black-box
#classifier = tree.DecisionTreeClassifier()           # for white-box
#classifier = KNeighborsClassifier(n_neighbors=5)
#classifier = RandomForestClassifier()


# Evaluate the classifier with cross validation
tp, fp, tn, fn, auc = cross_validation(classifier, feature, label, 10)
evaluation_result(tp, fp, tn, fn)


# Visualize the white-box
classifier = tree.DecisionTreeClassifier()
classifier.fit(feature, label)
tree.export_graphviz(classifier, out_file=figure_whitebox, max_depth=3)   # limit the depth for pretty visualization

# After that run below line on UNIX command line to export the dot file to png
# dot -Tpng tree.dot -o tree.png


#################
# Useful script #
#################


"""
# Evaluation
label_prediction = classifier.predict(feature_test)
evaluation_matrix = confusion_matrix(label_test, label_prediction, labels=['Chargeback', 'Settled'])
for row in evaluation_matrix:
    for column in row:
        print(column)

# Check new data
len(label_train[label_train == 1])
len(label_train[label_train == 0])
len(label_resampling[label_resampling == 1])
len(label_resampling[label_resampling == 0])

# Check the distribution of the data
label_train.value_counts()
label_test.value_counts()

# Evaluate classifier with sklearn function
classifier = LogisticRegression()
accuracy = cross_val_score(classifier, feature, label, cv=5, scoring="accuracy")
print("Accuracy: {}".format(np.mean(accuracy)))

# Print time
datetime.datetime.now()

# Export features to csv
feature.to_csv(working_directory+"features.csv", sep=',')
"""