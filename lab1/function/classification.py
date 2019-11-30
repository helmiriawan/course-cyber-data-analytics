import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


# Create function to run cross validation
def cross_validation(method, x, y, number_of_fold, smote_ratio=None):

    # Initiate the K-fold
    k_fold = KFold(n_splits=number_of_fold, shuffle=True, random_state=42)

    # Initiate the variables
    all_true_positives = []
    all_false_positives = []
    all_true_negatives = []
    all_false_negatives = []
    all_auc = []

    for train_index, test_index in k_fold.split(x):

        # Split train and test data
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Apply SMOTE
        if smote_ratio:
            oversampling = SMOTE(ratio=float(smote_ratio), random_state=42)
            x_smoted, y_smoted = oversampling.fit_sample(x_train, y_train)

        # Normalize the input variables (only based on training data)
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        if smote_ratio:
            x_smoted = scaler.transform(x_smoted)

        # Train classifier
        if smote_ratio:
            method.fit(x_smoted, y_smoted)
        else:
            method.fit(x_train, y_train)

        # Evaluate the model
        flag_prediction = method.predict(x_test)
        table_of_confusion = confusion_matrix(
            y_test,
            flag_prediction,
            labels=[1, 0]
        )

        true_positives = table_of_confusion[0][0]
        false_positives = table_of_confusion[1][0]
        true_negatives = table_of_confusion[1][1]
        false_negatives = table_of_confusion[0][1]

        flag_prediction_probability = method.predict_proba(x_test)[:, 1]
        area_under_curve = roc_auc_score(y_test, flag_prediction_probability)

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

    return all_true_positives, all_false_positives, all_true_negatives, \
        all_false_negatives, all_auc


# Create function to show the evaluation result from cross validation
def evaluate_result(
        true_positives, false_positives, true_negatives,
        false_negatives, print_result=False):

    true_predictions = true_positives + true_negatives
    all_predictions = true_positives + false_positives\
        + true_negatives + false_negatives

    accuracy = np.mean(true_predictions/all_predictions)
    precision = np.mean(true_positives/(true_positives+false_positives))
    sensitivity = np.mean((true_positives/(true_positives+false_negatives)))
    fpr = np.mean(false_positives/(false_positives+true_negatives))
    f_measure = np.mean(2*precision*sensitivity/(precision+sensitivity))

    if print_result:

        print("True positives\t: {:.0f}".format(np.mean(true_positives)))
        print("False positives\t: {:.0f}".format(np.mean(false_positives)))
        print("True negatives\t: {:.0f}".format(np.mean(true_negatives)))
        print("False negatives\t: {:.0f}".format(np.mean(false_negatives)))
        print("")

        print("Accuracy\t: {:.3f}".format(accuracy))
        print("Precision\t: {:.3f}".format(precision))
        print("Sensitivity\t: {:.3f}".format(sensitivity))
        print("FPR\t: {:.3f}".format(fpr))
        print("F-measure\t: {:.3f}".format(f_measure))

    return [
        np.mean(true_positives), np.mean(false_positives),
        np.mean(true_negatives), np.mean(false_negatives),
        accuracy, precision, sensitivity, fpr, f_measure
    ]