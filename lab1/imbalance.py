from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


# Function to generate ROC curve values
def roc_values(method, variable_train, variable_test, flag_train, flag_test):

    # Train classifier
    method.fit(variable_train, flag_train)

    # Generate the ROC curves values
    label_prediction_probability = method.predict_proba(variable_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(flag_test, label_prediction_probability)
    area_under_curve = roc_auc_score(flag_test, label_prediction_probability)

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

    plt.savefig(filename)
    plt.show()