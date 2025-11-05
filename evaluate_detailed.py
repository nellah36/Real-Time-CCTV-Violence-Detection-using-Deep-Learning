import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve,
                            f1_score, precision_score, recall_score)
import models
from dataGenerator import *
from tensorflow.keras.optimizers import Adam
import argparse
import json
from datetime import datetime
import pandas as pd

def plot_confusion_matrix(cm, classes, save_path, normalize=False):
    """Plot confusion matrix"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot ROC curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved: {save_path}")
    plt.close()

def plot_precision_recall_curve(precision, recall, avg_precision, save_path):
    """Plot Precision-Recall curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Precision-Recall curve saved: {save_path}")
    plt.close()

def plot_prediction_distribution(y_pred, y_true, save_path):
    """Plot distribution of prediction scores"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    axes[0].hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='NonFight', color='green')
    axes[0].hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Fight', color='red')
    axes[0].axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    axes[0].set_xlabel('Prediction Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Prediction Scores', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Violin plot
    data_for_violin = pd.DataFrame({
        'Prediction Score': y_pred,
        'True Class': ['Fight' if label == 1 else 'NonFight' for label in y_true]
    })
    sns.violinplot(data=data_for_violin, x='True Class', y='Prediction Score', ax=axes[1])
    axes[1].axhline(y=0.5, color='black', linestyle='--', label='Threshold=0.5')
    axes[1].set_title('Prediction Score Distribution by Class', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Prediction distribution saved: {save_path}")
    plt.close()

def save_misclassified_samples(y_true, y_pred, generator, save_dir, top_k=20):
    """Save information about misclassified samples"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    misclassified_idx = np.where(y_true != y_pred_binary)[0]
    
    # Sort by confidence (how wrong the model was)
    confidence_errors = np.abs(y_pred[misclassified_idx] - y_true[misclassified_idx])
    sorted_indices = misclassified_idx[np.argsort(confidence_errors)[::-1]]
    
    misclassified_data = []
    for idx in sorted_indices[:top_k]:
        file_path = generator.X_path[idx]
        true_label = 'Fight' if y_true[idx] == 1 else 'NonFight'
        pred_label = 'Fight' if y_pred_binary[idx] == 1 else 'NonFight'
        confidence = y_pred[idx]
        
        misclassified_data.append({
            'Index': int(idx),
            'File': os.path.basename(file_path),
            'True_Label': true_label,
            'Predicted_Label': pred_label,
            'Confidence': float(confidence),
            'Error_Magnitude': float(confidence_errors[np.where(sorted_indices == idx)[0][0]])
        })
    
    df = pd.DataFrame(misclassified_data)
    csv_path = os.path.join(save_dir, 'misclassified_samples.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Misclassified samples saved: {csv_path}")
    return df

def evaluate_detailed(args):
    # Setup
    mode = args.mode
    dataset = args.dataset
    batch_size = args.batchSize
    vid_len = args.modelVidLen
    input_frame_size = 160
    lstm_type = args.lstmType
    frame_diff_interval = 1
    
    if args.fusionType == 'C':
        model_function = models.getProposedModelC
    elif args.fusionType == 'A':
        model_function = models.getProposedModelA
    elif args.fusionType == 'M':
        model_function = models.getProposedModelM
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.resultsPath, f'evaluation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("DETAILED EVALUATION")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Model VidLen: {vid_len}")
    print(f"Input Frame Size: {input_frame_size}x{input_frame_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Results Directory: {results_dir}")
    print("="*70 + "\n")
    
    # Load test data
    if args.dataPath != 'NOT_SET':
        test_data_dir = os.path.join(args.dataPath, 'val')
    else:
        test_data_dir = f'{dataset}/processed/val'
    
    test_generator = DataGenerator(
        directory=test_data_dir,
        batch_size=batch_size,
        data_augmentation=False,
        shuffle=False,
        one_hot=False,
        sample=True,
        resize=input_frame_size,
        target_frames=vid_len,
        background_suppress=True,
        dataset=dataset,
        mode=mode
    )
    
    print(f"✓ Test samples loaded: {len(test_generator.X_path)}")
    
    # Build and load model
    print("\n> Building model architecture...")
    model = model_function(
        size=input_frame_size,
        seq_len=vid_len,
        frame_diff_interval=frame_diff_interval,
        mode=mode,
        lstm_type=lstm_type
    )
    
    optimizer = Adam(lr=4e-4, amsgrad=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    
    best_model_path = os.path.join(args.weightsPath, 'rwf2000_best_val_acc_Model')
    print(f"\n> Loading weights from: {best_model_path}")
    model.load_weights(best_model_path).expect_partial()
    model.trainable = False
    print("✓ Weights loaded successfully!")
    
    # Get predictions
    print("\n> Generating predictions...")
    y_pred = model.predict(test_generator, verbose=1, workers=8, use_multiprocessing=False)
    y_pred = y_pred.flatten()
    
    # Get true labels
    y_true = np.array([test_generator.Y_dict[path] for path in test_generator.X_path])
    
    # Binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    print("\n> Calculating metrics...")
    accuracy = np.mean(y_pred_binary == y_true)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = np.mean(precision_curve)
    
    # Classification report
    class_names = ['NonFight', 'Fight']
    report = classification_report(y_true, y_pred_binary, target_names=class_names, digits=4)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("="*70 + "\n")
    
    # Save metrics to JSON
    metrics = {
        'timestamp': timestamp,
        'dataset': dataset,
        'model_vidlen': vid_len,
        'input_frame_size': input_frame_size,
        'batch_size': batch_size,
        'total_samples': len(y_true),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'avg_precision': float(avg_precision),
        'confusion_matrix': cm.tolist(),
        'weights_path': best_model_path
    }
    
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Save classification report
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Classification report saved: {report_path}")
    
    # Generate visualizations
    print("\n> Generating visualizations...")
    
    # Confusion Matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path, normalize=False)
    
    cm_norm_path = os.path.join(results_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, class_names, cm_norm_path, normalize=True)
    
    # ROC Curve
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curve(fpr, tpr, roc_auc, roc_path)
    
    # Precision-Recall Curve
    pr_path = os.path.join(results_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(precision_curve, recall_curve, avg_precision, pr_path)
    
    # Prediction Distribution
    dist_path = os.path.join(results_dir, 'prediction_distribution.png')
    plot_prediction_distribution(y_pred, y_true, dist_path)
    
    # Save misclassified samples
    print("\n> Analyzing misclassified samples...")
    misclassified_df = save_misclassified_samples(y_true, y_pred, test_generator, 
                                                   results_dir, top_k=args.topKMisclassified)
    
    print("\n" + "="*70)
    print(f"EVALUATION COMPLETE!")
    print(f"All results saved to: {results_dir}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelVidLen', type=int, default=8,
                       help='Sequence length used during training')
    parser.add_argument('--batchSize', type=int, default=16,
                       help='Evaluation batch size')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['both', 'only_frames', 'only_differences'])
    parser.add_argument('--dataset', type=str, default='rwf2000',
                       choices=['rwf2000', 'movies', 'hockey'])
    parser.add_argument('--lstmType', type=str, default='sepconv',
                       choices=['sepconv', 'asepconv'])
    parser.add_argument('--fusionType', type=str, default='C',
                       choices=['C', 'A', 'M'])
    parser.add_argument('--dataPath', type=str, default='NOT_SET',
                       help='Path to preprocessed data folder')
    parser.add_argument('--weightsPath', type=str, default='NOT_SET',
                       help='Path to trained weights folder')
    parser.add_argument('--resultsPath', type=str, default='D:/evaluation_results',
                       help='Path to save evaluation results')
    parser.add_argument('--topKMisclassified', type=int, default=20,
                       help='Number of top misclassified samples to save')
    
    args = parser.parse_args()
    
    if args.weightsPath == "NOT_SET" or args.dataPath == "NOT_SET":
        parser.error("Both --weightsPath and --dataPath are required!")
    
    evaluate_detailed(args)

if __name__ == "__main__":
    main()
