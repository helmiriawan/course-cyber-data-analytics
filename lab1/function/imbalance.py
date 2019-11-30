from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


# Function to generate ROC curve values
def roc_values(method, feature_train, feature_test, label_train, label_test):

    # Train classifier
    method.fit(feature_train, label_train)

    # Generate the ROC curves values
    label_prediction_probability = method.predict_proba(feature_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(label_test, label_prediction_probability)
    auc = roc_auc_score(label_test, label_prediction_probability)

    return fpr, tpr, auc


# Function to generate ROC curves
def roc_curves(
        title, filename, fpr, tpr, auc,
        fpr_smote, tpr_smote, auc_smote):

    plt.title(title)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, color='darkorange',
             label='AUC UNSMOTEd = %0.2f' % auc)
    plt.plot(fpr_smote, tpr_smote, color='green',
             label='AUC SMOTEd = %0.2f' % auc_smote)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.savefig(filename)
    plt.show()


# Function to compare ROC curves
def compare_roc(
        algorithm, title, filename, feature_train, feature_test,
        feature_resampled, label_train, label_test, label_resampled):

    fpr, tpr, auc = roc_values(
        algorithm,
        feature_train,
        feature_test,
        label_train,
        label_test
    )

    fpr_smote, tpr_smote, auc_smote = roc_values(
        algorithm,
        feature_resampled,
        feature_test,
        label_resampled,
        label_test
    )

    roc_curves(
        title,
        filename,
        fpr,
        tpr,
        auc,
        fpr_smote,
        tpr_smote,
        auc_smote
    )