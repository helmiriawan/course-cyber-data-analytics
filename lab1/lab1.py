#################
# Configuration #
#################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from imblearn.over_sampling import SMOTE

working_directory = "C:/Users/helmi/unix/CS4035/lab_python/lab1/"
input_file = working_directory + "data_for_student_case.csv"
figure_directory = working_directory + "figure/"

# Create figure directory
if not os.path.exists(figure_directory):
    print('Creating directory for figures...')
    os.makedirs(figure_directory)



######################
# Data understanding #
######################

dataset = pd.read_csv(input_file)

print("Number of columns: {}".format(len(dataset.columns)))
dataset.columns
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

for column in ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']:
    dataset[column] = dataset[column].astype('category')



######################
# Visualization task #
######################

# Heat map
for flag in ['Chargeback', 'Settled']:
    
    # filter and aggregate data
    subset = dataset[dataset['simple_journal'] == flag]
    aggregation_data = subset.groupby(['shoppercountrycode', 'issuercountrycode']).size().reset_index(name='count')
    
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
    if flag == 'Chargeback' :
        plt.title('Issuer-Shopper Country Code of Chargeback Transactions')
        plt.savefig(figure_directory + 'heatmap_chargeback.png')
    else :
        plt.title('Issuer-Shopper Country Code of Settled Transactions')
        plt.savefig(figure_directory + 'heatmap_settled.png')
    plt.show()

# Box plot
sns.boxplot(x="simple_journal", y="amount", data=dataset[dataset['amount'] <= 300000])   # filter amount for pretty visualization

plt.title('Amount Distribution of Chargeback and Settled Transactions')
plt.xlabel('')
plt.ylabel('Amount')

tick_value = [50000, 100000, 150000, 200000, 250000, 300000]
tick_label = ['50k','100k','150k', '200k', '250k', '300k']
plt.yticks(tick_value, tick_label)

plt.savefig(figure_directory + 'boxplot_amount.png')
plt.show()

# Alternative boxplot
#dataset[dataset['amount'] <= 300000].boxplot(column='amount', by='simple_journal', rot=60)
#plt.show()



##################
# Imbalance task #
##################

# Data preparation
subset = dataset[['issuercountrycode', 'txvariantcode', 'amount', 'currencycode', 'shoppercountrycode', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode', 'simple_journal']]

subset.loc[subset.simple_journal == 'Chargeback', 'simple_journal'] = 1
subset.loc[subset.simple_journal == 'Settled', 'simple_journal'] = 0
subset['simple_journal'] = subset['simple_journal'].astype('int')

label = subset.simple_journal

feature = subset.drop('simple_journal', axis=1)
feature = pd.get_dummies(feature)

feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size = 0.5, random_state=42, stratify=label)

resampling = SMOTE(ratio=float(0.5), random_state=42)
feature_resampling, label_resampling = resampling.fit_sample(feature_train, label_train)

# Set classifier
#classifier = LogisticRegression()
#classifier = KNeighborsClassifier(n_neighbors=5)
classifier = tree.DecisionTreeClassifier()

# Train classifier
classifier.fit(feature_train, label_train)

# Evaluate
label_prediction_probability = classifier.predict_proba(feature_test)[:,1]
fpr, tpr, thresholds = roc_curve(label_test, label_prediction_probability)
auc = roc_auc_score(label_test, label_prediction_probability)

# Train classifier with SMOTEd features
classifier.fit(feature_resampling, label_resampling)

# Evaluate the SMOTEd features
smote_label_prediction_probability = classifier.predict_proba(feature_test)[:,1]
smote_fpr, smote_tpr, smote_thresholds = roc_curve(label_test, smote_label_prediction_probability)
smote_auc = roc_auc_score(label_test, smote_label_prediction_probability)

# Generate the ROC curve
#plt.title('AUC of Logistic Classifier')
#plt.title('AUC of KNN Classifier')
plt.title('AUC of Decision Tree Classifier')

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, color='darkorange', label='AUC UNSMOTEd = %0.2f' % auc)
plt.plot(smote_fpr, smote_tpr, color='green', label='AUC SMOTEd = %0.2f' % smote_auc)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

#plt.savefig(figure_directory + 'roc_logistic.png')
#plt.savefig(figure_directory + 'roc_knn.png')
plt.savefig(figure_directory + 'roc_decision_tree.png')
plt.show()

#*Go back and change the classfier (also the ROC curve title and file name)





#for classifier in [logistic, knn, decision_tree]:
for classifier in [logistic]:
    for feature, label in zip(feature_list, label_list):
        
        # Train clasifier
        classifier.fit(feature, label)
        
        label_prediction_probability = classifier.predict_proba(feature_test)[:,1]
        fpr, tpr, thresholds = roc_curve(label_test, label_prediction_probability)
        
        print("AUC: {}".format(roc_auc_score(label_test, label_prediction_probability)))





#################
# Useful script #
#################

#dataset[dataset.bookingdate < datetime.datetime(2015,11,10)].head(20)


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
