#################
# Configuration #
#################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

working_directory = "C:/Users/helmi/unix/CS4035/lab_python/lab1/"
input_file = working_directory + "data_for_student_case.csv"
figure_directory = working_directory + "figure/"

# Create figure directory
if not os.path.exists(figure_directory):
    os.makedirs(figure_directory)



######################
# Data understanding #
######################

dataset = pd.read_csv(input_file)
dataset.columns
dataset.info()
dataset.head()
dataset.describe()
dataset['simple_journal'].value_counts()



####################
# Data preparation #
####################

# Clean the data
dataset = dataset.dropna()
dataset = dataset[dataset.simple_journal != "Refused"]

# Change the label (simple_journal)
dataset.simple_journal[dataset['simple_journal'] == 'Chargeback'] = 'Fraudulent'
dataset.simple_journal[dataset['simple_journal'] == 'Settled'] = 'Legitimate'

# Change data type
for column in ['bookingdate', 'creationdate']:
    dataset[column] = pd.to_datetime(dataset.bookingdate, format='%Y-%m-%d', errors='coerce')

for column in ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']:
    dataset[column] = dataset[column].astype('category')



######################
# Visualization task #
######################

# Heat map
for flag in ['Fraudulent', 'Legitimate']:
    
    # filter and aggregate data
    subset = dataset[dataset['simple_journal'] == flag]
    aggregation_data = subset.groupby(['shoppercountrycode', 'issuercountrycode']).size().reset_index(name='count')
    
    # filter country code for pretty visualization
    issuer_country = list(dataset.issuercountrycode[dataset.simple_journal == 'Fraudulent'].unique())
    shopper_country = list(dataset.shoppercountrycode[dataset.simple_journal == 'Fraudulent'].unique())
    aggregation_data = aggregation_data[aggregation_data['issuercountrycode'].isin(issuer_country)]
    aggregation_data = aggregation_data[aggregation_data['shoppercountrycode'].isin(shopper_country)]
    
    # generate pivot and heat map
    pivot_data = aggregation_data.pivot(index='shoppercountrycode', columns='issuercountrycode', values='count')
    sns.heatmap(pivot_data, cmap='viridis', linewidths=.5)
    plt.xlabel('Issuer Country Code')
    plt.ylabel('Shopper Country Code')
    if flag == 'Fraudulent' :
        plt.title('Issuer-Shopper Country Code of Fraudulent Transactions')
        plt.savefig(figure_directory + 'heatmap_fraudulent.png')
    else :
        plt.title('Issuer-Shopper Country Code of Legitimate Transactions')
        plt.savefig(figure_directory + 'heatmap_legitimate.png')
    plt.show()

# Box plot
sns.boxplot(x="simple_journal", y="amount", data=dataset[dataset['amount'] <= 300000])   # filter amount for pretty visualization
plt.title('Amount Distribution of Fraudulent and Legitimate Transactions')
plt.xlabel('')
plt.ylabel('Amount')
plt.show()


