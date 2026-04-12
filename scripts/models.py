import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def get_svm_model(kernel='rbf', C=1.0):
    """Returns an SVM model. Defaulting to RBF kernel."""
    return SVC(
        kernel=kernel, 
        C=C, 
        class_weight='balanced', 
        probability=True,
        random_state=42
    )


def evaluate_model(model, X_set, y_true, set_name="Validation"):
    """Prints a detailed performance report for the model."""
    predictions = model.predict(X_set)
    print(f"--- {set_name} Performance ---")
    report = classification_report(y_true, predictions)
    print(report)
    
    # The Confusion Matrix shows exactly how many All-Stars were missed
    cm = confusion_matrix(y_true, predictions)
    print("Confusion Matrix:")
    print(cm)
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual: Not All-Star', 'Actual: All-Star'], 
        columns=['Predicted: Not All-Star', 'Predicted: All-Star']
    )
    
    print("--- Detailed Confusion Matrix ---")
    print(cm_df)
    return classification_report(y_true, predictions, output_dict=True)