import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch

def plot_history(history, optimizer_name, save_path=None):
    """
    Plots training and validation accuracy and loss.
    """
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy - {optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss - {optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plots a confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_comparative_history(histories, metric='val_acc', save_path=None):
    """
    Plots the same metric for multiple optimizers for comparison.
    """
    plt.figure(figsize=(10, 6))
    for optimizer_name, history in histories.items():
        plt.plot(history[metric], label=optimizer_name)
    
    plt.title(f'Comparative {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.split('_')[-1].capitalize())
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
