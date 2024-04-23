import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

RAND_STATE = 22
FEAT_COLS = ["alpha","delta","u","g","r","i","z","redshift"]
COLS = FEAT_COLS + ["class"]
CLASS_NUMS = {"STAR": 0, "GALAXY": 1, "QSO": 2}
CM_LABELS = ["STAR", "GALAXY", "QSO"]


def get_confusion_matrix_dict(y_true, y_pred, labels) -> dict:
    """Returns a dictionary containing TN, FP, FN, TP, Accuracy, TNR, and TPR,
    for given y_true and y_pred arrays."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ts, fs_g, fs_q, fg_s, tg, fg_q, fq_s, fq_g, tq, = cm.ravel()
    tsr = ts / (ts+fs_g+fs_q)
    tgr = tg / (tg+fg_s+fg_q)
    tqr = tq / (tq+fq_s+fq_g)
    acc = (ts+tg+tq) / sum(cm.ravel())
    cm_dict = {
        "TS": ts, "FS-G": fs_g, "FS-Q": fs_q,
        "TG": tg, "FG-S": fg_s, "FG-Q": fg_q,
        "TQ": tq, "FQ-S": fq_s, "FQ-G": fq_g,
        "TSR": tsr, "TGR": tgr, "TQR": tqr,
        "ACCURACY": acc
    }
    return cm_dict

def accuracy_str(accuracy: float):
    """Get a print-friendly string form of the provided decimal accuracy."""
    return "{:.4f}%".format(accuracy*100)

def show_accuracy_chart(df: pd.DataFrame, title: str, rotate_x_ticks: bool=False, xticks=None):
    plt.bar(df.index, df["Accuracy"])
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Hyper-Parameter Settings")
    if xticks is not None:
        plt.xticks(xticks)
    if rotate_x_ticks:
        plt.xticks(rotation=90)
    plt.yscale('log')
    plt.show()

def conf_mtrx_heatmap(cm: np.ndarray, rows: list, cols: list, title: str):
    # Note: CM rows are True labels, cols are Predicted labels
    cm_df = pd.DataFrame(cm, columns=cols, index=rows)
    plt.figure()
    sns.heatmap(cm_df, annot=True, fmt="d")
    plt.title(title)
    plt.show()

def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    temp_df = pd.DataFrame(zip(y_true, y_pred),
                           columns=["Actual", "Predicted"])
    all_count = len(temp_df.values)
    correct = len(temp_df.loc[temp_df["Predicted"]==temp_df["Actual"]])
    return correct / all_count

def get_linear_model(x_train, y_train, deg: int):
    weights = np.polyfit(x_train, y_train, deg)
    model = np.poly1d(weights)
    return model

def main():
    # Read the star classification data
    df = pd.read_csv("star_classification.csv").loc[:,COLS]
    # Split into Train and Test data sets
    train_df, test_df = train_test_split(df, test_size=0.25,
                                    random_state=RAND_STATE,
                                    stratify=df["class"].values)

    # Print information on the data set:
    df_description = df.describe()
    print("\n## Data Set Info ##")
    print()
    print(df_description)
    print()
    print(f"Full Data Rows: {len(df)}")
    print(f"Stars: {len(df.loc[df['class']=='STAR'])}")
    print(f"Galaxies: {len(df.loc[df['class']=='GALAXY'])}")
    print(f"Quasars: {len(df.loc[df['class']=='QSO'])}")
    print()
    print(f"Training Data Rows: {len(train_df)}")
    print(f"Test Data Rows: {len(test_df)}")
    print("\nWriting star classification data set summary to csv...")
    df_description.to_csv("StarClassificationsDataSummary.csv", float_format="%.6f")

    # Get X_train and Y_train
    X_trn = train_df[FEAT_COLS].values
    Y_trn = train_df[['class']].values.ravel()

    # Get X_test & Y_test
    X_tst = test_df[FEAT_COLS].values
    Y_tst = test_df[['class']].values.ravel()

    # Use a scaler for X_train
    scaler = StandardScaler()
    scaler.fit(X_trn)
    X_trn = scaler.transform(X_trn)
    # Scale X_test
    X_tst = scaler.transform(X_tst)

    # K-Nearest-Neighbors
    print("\n## K-NEAREST-NEIGHBORS ##")
    k_vals = [1,3,5,7,9,11,13]
    # Setup placeholder for best model, and dict for comparisons
    best_knn = None
    best_knn_preds = None
    k_accuracies = {}
    for k in k_vals:
        # Instantiate and fit the k-NN classifier for k
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_trn,Y_trn)
        # Predict labels for X_test
        predictions = knn_classifier.predict(X_tst)
        # Get Accuracy
        accrcy = get_accuracy(Y_tst, predictions)
        print(f"K={k} Accuracy: {accuracy_str(accrcy)}")
        # Save the best classifier and best set of predictions for later use
        if best_knn is None or accrcy > max(k_accuracies.values()):
            best_knn = knn_classifier
            best_knn_preds = predictions
        k_accuracies.update({k: accrcy})
    knns_df = pd.DataFrame.from_dict(k_accuracies, orient='index',
                                     columns=["Accuracy"])
    print("\nShowing KNN accuracies bar chart...")
    show_accuracy_chart(knns_df, "K-NN Accuracy by K Value", xticks=k_vals)
    # Report the best KNN classifier by accuracy:
    best_k = best_knn.get_params().get("n_neighbors")
    best_acc = max(k_accuracies.values())*100
    print("Best K-NN Model: K={:d}, Accuracy={:.4f}".format(best_k, best_acc))
    knn_cm_dict = get_confusion_matrix_dict(Y_tst, best_knn_preds,
                                            CM_LABELS)
    print("K-Nearest-Neighbors Stats:")
    print(knn_cm_dict)
    print(f"Accuracy: {accuracy_str(knn_cm_dict['ACCURACY'])}")

    cm = confusion_matrix(Y_tst, best_knn_preds, labels=CM_LABELS)
    print("\nShowing KNN confusion matrix chart...")
    conf_mtrx_heatmap(cm, ["True Star", "True Galaxy", "True Quasar"],
                    ["Predicted Star", "Predicted Galaxy", "Predicted Quasar"],
                    "K-NN Confusion Matrix")

    # Logistic Regression
    print("\n## Logistic Regression ##")
    # Set max iterations higher since lbfgs failed to converge within 100 iterations.
    logreg = LogisticRegression(max_iter=1000, random_state=RAND_STATE)
    logreg.fit(X_trn,Y_trn)
    logreg_predictions = logreg.predict(X_tst)
    logreg_cm_dict = get_confusion_matrix_dict(Y_tst, logreg_predictions,
                                               CM_LABELS)
    print("Logistic Regression Stats:")
    print(logreg_cm_dict)
    print(f"Accuracy: {accuracy_str(logreg_cm_dict['ACCURACY'])}")
    cm = confusion_matrix(Y_tst, logreg_predictions, labels=CM_LABELS)
    print("\nShowing Logistic Regression confusion matrix chart...")
    conf_mtrx_heatmap(cm, ["True Star", "True Galaxy", "True Quasar"],
                    ["Predicted Star", "Predicted Galaxy", "Predicted Quasar"],
                    "Logistic Regression Confusion Matrix")

    # Naive Bayes
    print("\n## Naive Bayes ##")
    gnb_classifier = GaussianNB().fit(X_trn, Y_trn)
    gnb_predictions = gnb_classifier.predict(X_tst)
    gnb_cm_dict = get_confusion_matrix_dict(Y_tst, gnb_predictions,
                                            CM_LABELS)
    print("Gaussian Naive Bayes Stats:")
    print(gnb_cm_dict)
    print(f"Accuracy: {accuracy_str(gnb_cm_dict['ACCURACY'])}")
    cm = confusion_matrix(Y_tst, gnb_predictions, labels=CM_LABELS)
    print("\nShowing Naive Bayes confusion matrix chart...")
    conf_mtrx_heatmap(cm, ["True Star", "True Galaxy", "True Quasar"],
                    ["Predicted Star", "Predicted Galaxy", "Predicted Quasar"],
                    "Naive Bayes Confusion Matrix")

    # Random Forest
    print("\n## RANDOM FOREST ##")
    rfc_accuracies = {}
    best_rfc = None
    best_rfc_predictions = None
    for n in range(6,13):
        for d in range(6, 10):
            rfc = RandomForestClassifier(n_estimators=n,
                                         criterion='entropy', max_depth=d,
                                         random_state=RAND_STATE)
            rfc.fit(X_trn, Y_trn)
            predictions = rfc.predict(X_tst)
            accuracy = get_accuracy(Y_tst, predictions)
            if best_rfc is None or accuracy >= max(rfc_accuracies.values()):
                best_rfc = rfc
                best_rfc_predictions = predictions
            rfc_accuracies.update({f"N={n}, D={d}": accuracy})
    # Plot accuracies by hyper-params
    rfcs_df = pd.DataFrame.from_dict(rfc_accuracies, orient='index',
                                     columns=["Accuracy"])
    print("Showing RFC accuracies bar chart...")
    show_accuracy_chart(rfcs_df, "Random Forest Accuracy by Hyperparameters",
                        rotate_x_ticks=True)
    # Report best RFC by accuracy
    d = best_rfc.get_params().get('max_depth')
    n = best_rfc.get_params().get('n_estimators')
    print(f"Best Random Forest hyper-parameters: N={n}, D={d}")
    max_acc = max(rfc_accuracies.values())*100
    max_acc = accuracy_str(max(rfc_accuracies.values()))
    print(f"Best Random Forest Accuracy: {max_acc}")
    rfc_cm_dict = get_confusion_matrix_dict(Y_tst, best_rfc_predictions,
                                            CM_LABELS)
    print("Random Forest Classifier Stats:")
    print(rfc_cm_dict)

    cm = confusion_matrix(Y_tst, best_rfc_predictions, labels=CM_LABELS)
    print("\nShowing Random Forest confusion matrix chart...")
    conf_mtrx_heatmap(cm, ["True Star", "True Galaxy", "True Quasar"],
                    ["Predicted Star", "Predicted Galaxy", "Predicted Quasar"],
                    "Random Forest Confusion Matrix")

    print("\n## SUMMARY ##")
    combined_cms = {
        "K-Nearest Neighbors": knn_cm_dict,
        "Logistic Regressions": logreg_cm_dict,
        "Naive Bayes": gnb_cm_dict,
        "Random Forest": rfc_cm_dict
    }

    cm_df = pd.DataFrame(combined_cms)
    print(cm_df)
    print("\nWriting classifiers comparison to csv report...")
    cm_df.to_csv("StarClassifiersComparison.csv", float_format="%.6f")


if __name__ == "__main__":
    main()