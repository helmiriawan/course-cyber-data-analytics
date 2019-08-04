import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from math import sqrt, ceil
from sklearn.metrics import mean_squared_error

def filter_list(data, patterns, exclude=False):
    '''
    data is list of original columns
    patterns is list of patterns, which indicate columns that will be excluded 
    (if exclude flag is true) or included (if exclude flag is false)
    '''
    if type(patterns) != list:
        patterns = [patterns]
    
    for pattern in patterns:
        if exclude:
            data = [item for item in data if pattern not in item]
        else:
            data = [item for item in data if pattern in item]
    
    return(data)

def filter_time(data, start_date=None, end_date=None, time_column='Timestamp'):
    '''
    data is a data frame
    start_date and end_date determine the range of period that will be selected
    time_column is the name of column that defines time
    '''
    
    column_index = data.columns.values.tolist().index(time_column)
    
    if start_date is None:
        start_date = min(data.iloc[:,column_index])
    if end_date is None:
        end_date = max(data.iloc[:,column_index])
    
    index = (
        data.iloc[:,column_index] >= start_date
    ) & (
        data.iloc[:,column_index] <= end_date
    )
    filtered_data = data.loc[index]
    filtered_data = filtered_data.reset_index(drop=True)
    
    return(filtered_data)

def get_xticks(input_list, number_ticks, time_format='%d-%b %H'):
    '''
    input_list is a list of xtick labels
    number_ticks is number of ticks that is desired
    '''
    
    xindex = range(0, len(input_list), int(len(input_list)/number_ticks))
    xlabel = [input_list[i].strftime(time_format) for i in xindex]
    
    return(xindex, xlabel)

def plot_prediction(actual, prediction, model, variable, time, filename, 
                    fig_size=(15, 8)):
    '''
    actual, prediction, and time are lists
    model and variable are strings
    produce a chart about actual vs prediction data
    '''
    
    plt.figure(figsize=fig_size)
    plt.title(model+' Prediction vs Actual', fontsize=16)
    plt.plot(actual, label=variable)
    plt.plot(prediction, label='Prediction')
    xindex, xlabel = get_xticks(time, 12, time_format='%H:%M')
    plt.xticks(xindex, xlabel, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16)
    plt.savefig('plot/prediction-' + filename);

def batch_data(data, batch_size):
    '''
    data is a list
    returns a list of size-batch_size lists
    '''
    
    batch = []
    for i in range(0, len(data), batch_size):
        batch.append(data[i:i+batch_size])
    
    return batch

def resample(data, interval, time_column='Timestamp', fill_null=False):
    '''
    data is a data frame
    interval is the new sampling rate (H/T/S for hour/minute/second)
    time_column is the name of column that indicate the time
    '''
    
    data = data.set_index(time_column)
    data = data.resample(interval).first()
    data = data.reset_index()
    
    if fill_null:
        data = data.fillna(0)
    
    return(data)

def get_sequences(features, targets, length=2):
    '''
    data is a list of lists
    length is the length of the sequences
    '''

    # Create placeholders
    feature_sequences = []
    target_sequences = []
    window = []
    
    # Initialize the sequences
    for index in range(0, length):
        window.append(features[index])
    feature_sequences.append(list(window))
    target_sequences.append([targets[length-1]])
    
    # Move the window through all the data
    for index in range(length, len(features)):
        window.pop(0)
        window.append(features[index])
        feature_sequences.append(list(window))
        target_sequences.append([targets[index]])

    return(feature_sequences, target_sequences)

def get_attacks(list_attack, attack_data, length=-1):
    '''
    list_attack is a list of records about all attacks
    attack_data is sensor data collected when attacks happened
    length needs to be specified when using RNNs model
    returns a list of lists that indicates the time of attacks
    '''
    
    # Get list of attacks (represented by indices)
    attack_indices = []
    for attack in list_attack.loc[:, ['start_time', 'end_time']].values:
        indices = []
        for time in attack:
            flags = attack_data.loc[attack_data['Timestamp']>=time, 'Timestamp']
            if not flags.empty:
                index = flags.index[0]
                if index != 0:
                    indices.append(index-length+1)
        attack_indices.append(indices)

    # Filter missing indices
    attack_indices = [x for x in attack_indices if len(x) > 0]
    
    return(attack_indices)

def plot_comparison(actual, prediction, error, max_error, min_error, dataframe,
                    title, attack_indices=None, number_ticks=12):
    '''
    actual is list of actual data
    prediction is list of prediction data
    error is list of error data
    max_error and min error are the threshold obtained from evaluation
    dataframe is source of data that still has time information
    title is title of the plot
    attack_indices is list of lists that indicates the attacks
    number_ticks is number of ticks in x axis
    '''

    plt.figure(figsize=(15,4))
    plt.title(title, fontsize=16)

    plt.plot(actual, label='Actual')
    plt.plot(prediction, label='Prediction')
    plt.plot(error, label='Error')
    plt.axhline(y=max_error, linestyle='dashed', linewidth=1.0)
    plt.axhline(y=min_error, linestyle='dashed', linewidth=1.0)

    xindex, xlabel = get_xticks(
        input_list = dataframe.loc[:, 'Timestamp'], 
        number_ticks = number_ticks, 
        time_format = '%H:%M'
    )
    plt.xticks(xindex, xlabel, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-500, 1000)
    if attack_indices is not None:
        for attack in attack_indices:
            plt.axvspan(attack[0], attack[1], color='grey', alpha=0.2)
    plt.legend(fontsize=12);

class nn:
    
    def __init__(self, input_neurons=4, hidden_neurons=100, length=5,
                 number_layers= 2, output_neurons=1, learning_rate=0.001):
        self.sess = tf.Session()
        self.networks = self.rnn(
            input_neurons, 
            hidden_neurons, 
            length, 
            number_layers,
            output_neurons
        )
        self.optimizer = self.init_optimizer(learning_rate)
        self.saver = tf.train.Saver()
        
        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())
        
    def rnn(self, input_neurons, hidden_neurons, length, number_layers, 
            output_neurons):
        
        # Create placeholders for input data
        self.features = tf.placeholder(
            tf.float32, 
            shape=[None, length, input_neurons], 
            name='features'
        )
        self.targets = tf.placeholder(
            tf.float32, 
            shape=[None, output_neurons], 
            name='target'
        )

        # Define recurrent networks
        stacked_recurrent = []
        for _ in range(number_layers):
            recurrent_cell = tf.nn.rnn_cell.LSTMCell(
                hidden_neurons, 
                activation='tanh'
            )
            stacked_recurrent.append(recurrent_cell)
        stacked_recurrent = tf.nn.rnn_cell.MultiRNNCell(stacked_recurrent)
        recurrent_output, state = tf.nn.dynamic_rnn(
            stacked_recurrent, 
            inputs=self.features, 
            dtype=tf.float32
        )

        # Take the last sequence of unrolled recurrent cell
        recurrent_output = tf.reshape(
            tf.split(recurrent_output, length, axis=1, num=None, name='split')[-1],
            [-1, hidden_neurons]
        )

        # Define the output layer
        weight = tf.Variable(tf.random_normal([hidden_neurons, 1]))
        bias = tf.Variable(tf.random_normal([1]))
        output = tf.add(tf.matmul(recurrent_output, weight), bias)

        return(output)


    def init_optimizer(self, learning_rate):
        
        # Define loss function
        self.loss = tf.losses.mean_squared_error(self.targets, self.networks)

        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        return(optimizer.minimize(self.loss))
    
    def train(self, features, targets, test_features=[], test_targets=[], epochs=20, 
                batch_size=50):
        
        losses = []
        training_error = []
        validation_error = []
        
        training_data = list(zip(features, targets))
        for epoch in range(epochs):
            
            # Minibatches selected randomly
            random.shuffle(training_data)
            
            loss_epoch = []
            for batch in range(ceil(len(features) / batch_size)):
                indices = [batch*batch_size, batch*batch_size+batch_size]
                x_batch, y_batch = zip(*training_data[indices[0]:indices[1]])
                                
                optimization, train_loss = self.sess.run(
                    [self.optimizer, self.loss], 
                    feed_dict={
                        self.features:x_batch, 
                        self.targets:y_batch
                    }
                )
                loss_epoch.append(train_loss)
            
            # Log the epoch
            loss_epoch = np.mean(np.array(loss_epoch))
            losses.append(loss_epoch)

            # Log the training error
            train_prediction = self.predict(features)
            training_error.append(sqrt(mean_squared_error(targets, train_prediction)))

            # Log the validation error
            validation_prediction =self.predict(test_features)
            rmse = sqrt(mean_squared_error(test_targets, validation_prediction))
            validation_error.append(rmse)

            # Show status
            print('Epoch: %d of %d, validation error: %4.2f' % (
                epoch+1, epochs, validation_error[-1]
            ))
    
    def predict(self, features):
        
        prediction = self.sess.run(
            self.networks,
            feed_dict={
                self.features:features
            }
        )
        
        return(prediction)
    
    def save(self, filename):
        self.saver.save(self.sess, filename + '.ckpt')
        
    def load(self, filename):
        self.saver.restore(self.sess, filename + '.ckpt')