from sklearn.metrics import classification_report, roc_auc_score

def evaluate(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))