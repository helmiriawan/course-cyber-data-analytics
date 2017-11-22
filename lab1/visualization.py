import seaborn as sns
import matplotlib.pyplot as plt


def heatmap(dataset, label, title, filename):

    # filter and aggregate data
    filtered_dataset = dataset[dataset['simple_journal'] == label]
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
    plt.savefig(filename)
    plt.show()


def boxplot(dataset, title, filename):
    sns.boxplot(x="simple_journal", y="amount", data=dataset[dataset['amount'] <= 300000])  # for pretty visualization

    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Amount')

    tick_value = [50000, 100000, 150000, 200000, 250000, 300000]
    tick_label = ['50k', '100k', '150k', '200k', '250k', '300k']
    plt.yticks(tick_value, tick_label)

    plt.savefig(filename)
    plt.show()


"""
# Alternative boxplot
dataset[dataset['amount'] <= 300000].boxplot(column='amount', by='simple_journal', rot=60)
plt.show()
"""