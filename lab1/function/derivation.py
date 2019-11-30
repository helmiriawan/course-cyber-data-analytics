import datetime

def get_derived_attributes(dataset):

    # Initiate the variables
    transactions_same_day = []
    amount_same_day = []
    transactions_last_30_days = []
    average_amount_daily_30 = []
    average_amount_transaction_30 = []
    average_amount_weekly_90 = []
    transactions_same_currency = []
    average_amount_same_currency = []
    transactions_same_ip = []
    average_amount_same_ip = []
    transactions_same_mail = []
    average_amount_same_mail = []
    transactions_same_issuer = []
    average_amount_same_issuer = []
    transactions_same_shopper = []
    average_amount_same_shopper = []
    same_country = []
    count = 0
    
    for row in dataset.values.tolist():
        
        # Get parameters
        card_id = row[dataset.columns.tolist().index('card_id')]
        creation_datetime = row[dataset.columns.tolist().index('creationdate')]
        creation_date = creation_datetime.date()
        creation_datetime_last_30 = creation_datetime + datetime.timedelta(-30)
        creation_datetime_last_90 = creation_datetime + datetime.timedelta(-90)
        
        currency = row[dataset.columns.tolist().index('currencycode')]
        ip = row[dataset.columns.tolist().index('ip_id')]
        mail = row[dataset.columns.tolist().index('mail_id')]
        issuercountry = row[dataset.columns.tolist().index('issuercountrycode')]
        shoppercountry = row[dataset.columns.tolist().index('shoppercountrycode')]

        # Get information from the same day
        last_transactions_1 = dataset.loc[
            (dataset['card_id'] == card_id)
            & (dataset['creationdate'] < creation_datetime)
            & (dataset['creationdate'] >= creation_date)
        ]
        
        # Get information from the last 30 days
        last_transactions_30 = dataset.loc[
            (dataset['card_id'] == card_id)
            & (dataset['creationdate'] < creation_datetime)
            & (dataset['creationdate'] >= creation_datetime_last_30)
        ]
        
        # Get information from the last 90 days
        last_transactions_90 = dataset.loc[
            (dataset['card_id'] == card_id)
            & (dataset['creationdate'] < creation_datetime)
            & (dataset['creationdate'] >= creation_datetime_last_90)
        ]
        
        # Get information from transactions with similar currency
        same_currency = last_transactions_30.loc[
            last_transactions_30['currencycode'] == currency
        ]
        
        # Get information from transactions with similar IP
        same_ip = last_transactions_30.loc[
            last_transactions_30['ip_id'] == ip
        ]
        
        # Get information from transactions with similar mail
        same_mail = last_transactions_30.loc[
            last_transactions_30['mail_id'] == mail
        ]
        
        # Get information from transactions with similar country of issuer
        same_issuer = last_transactions_30.loc[
            last_transactions_30['issuercountrycode'] == issuercountry
        ]
        
        # Get information from transactions with similar country of shopper
        same_shopper = last_transactions_30.loc[
            last_transactions_30['shoppercountrycode'] == shoppercountry
        ]
        
        # Derive some information
        transactions_1 = last_transactions_1.shape[0]
        amount_1 = last_transactions_1.sum()['amount']
        transactions_30 = last_transactions_30.shape[0]
        amount_30 = last_transactions_30.sum()['amount']
        amount_90 = last_transactions_90.sum()['amount']
        transactions_currency = same_currency.shape[0]
        amount_currency = same_currency.sum()['amount']
        transactions_ip = same_ip.shape[0]
        amount_ip = same_ip.sum()['amount']
        transactions_mail = same_mail.shape[0]
        amount_mail = same_mail.sum()['amount']
        transactions_issuer = same_issuer.shape[0]
        amount_issuer = same_issuer.sum()['amount']
        transactions_shopper = same_shopper.shape[0]
        amount_shopper = same_shopper.sum()['amount']

        # Derive further and store the information
        transactions_same_day.append(transactions_1)
        amount_same_day.append(amount_1)
        transactions_last_30_days.append(transactions_30)
        average_amount_daily_30.append(amount_30/30)
        if transactions_30 > 0:
            average_amount_transaction_30.append(amount_30/transactions_30)
        else:
            average_amount_transaction_30.append(0)
        average_amount_weekly_90.append(amount_90/12)
        transactions_same_currency.append(transactions_currency)
        average_amount_same_currency.append(amount_currency/30)
        transactions_same_ip.append(transactions_ip)
        average_amount_same_ip.append(amount_ip/30)
        transactions_same_mail.append(transactions_mail)
        average_amount_same_mail.append(amount_mail/30)
        transactions_same_issuer.append(transactions_issuer)
        average_amount_same_issuer.append(amount_issuer/30)
        transactions_same_shopper.append(transactions_shopper)
        average_amount_same_shopper.append(amount_shopper/30)
        if issuercountry == shoppercountry:
            same_country.append(True)
        else:
            same_country.append(False)
        
        # Print log
        #count = count + 1
        #print('row: ', count)
    
    # Add to new columns
    dataset['transactions_1'] = transactions_same_day
    dataset['total_amount_1'] = amount_same_day
    dataset['transactions_30'] = transactions_last_30_days
    dataset['average_amount_daily_30'] = average_amount_daily_30
    dataset['average_amount_transaction_30'] = average_amount_transaction_30
    dataset['average_amount_weekly_90'] = average_amount_weekly_90
    dataset['transactions_same_currency'] = transactions_same_currency
    dataset['average_amount_same_currency'] = average_amount_same_currency
    dataset['transactions_same_ip'] = transactions_same_ip
    dataset['average_amount_same_ip'] = average_amount_same_ip
    dataset['transactions_same_mail'] = transactions_same_mail
    dataset['average_amount_same_mail'] = average_amount_same_mail
    dataset['transactions_same_issuer'] = transactions_same_issuer
    dataset['average_amount_same_issuer'] = average_amount_same_issuer
    dataset['transactions_same_shopper'] = transactions_same_shopper
    dataset['average_amount_same_shopper'] = average_amount_same_shopper
    dataset['same_country'] = same_country
    
    return dataset