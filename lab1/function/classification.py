import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE


# Create function to run cross validation
def cross_validation(method, variable, flag, number_of_fold):

    # Initiate the K-fold
    k_fold = KFold(n_splits=number_of_fold, shuffle=True, random_state=42)

    # Initiate the variables
    all_true_positives = []
    all_false_positives = []
    all_true_negatives = []
    all_false_negatives = []
    all_auc = []

    for train_index, test_index in k_fold.split(variable):

        # Split train and test data
        variable_train, variable_test = variable.iloc[train_index], variable.iloc[test_index]
        flag_train, flag_test = flag.iloc[train_index], flag.iloc[test_index]

        # Apply SMOTE
        oversampling = SMOTE(ratio=float(0.18), random_state=42)
        variable_oversampling, flag_oversampling = oversampling.fit_sample(variable_train, flag_train)

        # Train classifier
#       method.fit(variable_train, flag_train)
        method.fit(variable_oversampling, flag_oversampling)

        # Evaluate the model
        flag_prediction = method.predict(variable_test)
        table_of_confusion = confusion_matrix(flag_test, flag_prediction, labels=[1, 0])

        true_positives = table_of_confusion[0][0]
        false_positives = table_of_confusion[1][0]
        true_negatives = table_of_confusion[1][1]
        false_negatives = table_of_confusion[0][1]

        flag_prediction_probability = method.predict_proba(variable_test)[:, 1]
        area_under_curve = roc_auc_score(flag_test, flag_prediction_probability)

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

    return all_true_positives, all_false_positives, all_true_negatives, all_false_negatives, all_auc


# Create function to show the evaluation result from cross validation
def evaluation_result(true_positives, false_positives, true_negatives, false_negatives, auc):

    accuracy = (true_positives+true_negatives) / (true_positives+false_positives+true_negatives+false_negatives)
    sensitivity = true_positives / (true_positives+false_negatives)
    specificity = true_negatives / (false_positives+true_negatives)
    precision = true_positives / (true_positives+false_positives)
    f_measure = 2 * precision * sensitivity / (precision+sensitivity)

    print("True positives\t: {:.0f}".format(np.mean(true_positives)))
    print("False positives\t: {:.0f}".format(np.mean(false_positives)))
    print("True negatives\t: {:.0f}".format(np.mean(true_negatives)))
    print("False negatives\t: {:.0f}".format(np.mean(false_negatives)))
    print("")

    print("Accuracy\t: {:.3f}".format(np.mean(accuracy)))
    print("Sensitivity\t: {:.3f}".format(np.mean(sensitivity)))
    print("Specificity\t: {:.3f}".format(np.mean(specificity)))
    print("Precision\t: {:.3f}".format(np.mean(precision)))
    print("F-measure\t: {:.3f}".format(np.mean(f_measure)))
    print("AUC\t\t: {:.3f}".format(np.mean(auc)))