#################
# Configuration #
#################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

working_directory = "C:/Users/helmi/unix/CS4035/lab_python/lab1/"
input_file = working_directory + "data_for_student_case.csv"
output_file = working_directory + "test.png"



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

# Change data type
for column in ['bookingdate', 'creationdate']:
    dataset[column] = pd.to_datetime(dataset.bookingdate, format='%Y-%m-%d', errors='coerce')

for column in ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']:
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
    if flag == 'Chargeback' :
        plt.title('Heat Map of Issuer and Shopper Country Code from Fraudulent Transaction')
    else :
        plt.title('Heat Map of Issuer and Shopper Country Code from Legitimate Transaction')
    plt.show()
