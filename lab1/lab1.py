#################
# Configuration #
#################

import pandas as pd

working_directory = "C:/Users/helmi/unix/CS4035/lab_python/lab1/"
input_file = working_directory + "data_for_student_case.csv"



######################
# Data understanding #
######################

dataset = pd.read_csv(input_file)
dataset.columns
dataset.info()
dataset.head()
dataset.describe()

dataset['simple_journal'].value_counts()

# to be changed: bookingdate, issuercountrycode, txvariantcode, currencycode, 
#                shoppercountrycode, shopperinteraction, simple_journal,
#                cardverificationcodesupplied, cvcresponsecode, creationdate, 
#                accountcode


####################
# Data preparation #
####################

# Remove some data
dataset = dataset.dropna()
dataset = dataset[dataset.simple_journal != "Refused"]

# Change data type
for column in ['bookingdate', 'creationdate']:
    dataset[column] = pd.to_datetime(dataset.bookingdate, format='%Y-%m-%d', errors='coerce')

for column in ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']:
    dataset[column] = dataset[column].astype('category')

