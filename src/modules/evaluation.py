"""
Model Evaluation Module
========================

This module provides comprehensive evaluation metrics, visualization, and analysis tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score,
    average_precision_score, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation following the 3rd place solution approach.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the evaluator.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        
    def evaluate_single_model(self, y_true, y_pred_proba, y_pred=None, 
                             model_name="Model", threshold=0.5):
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            y_pred (array-like, optional): Predicted labels
            model_name (str): Name of the model
            threshold (float): Classification threshold
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Basic classification metrics
        metrics = {
            'model_name': model_name,
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
        })
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def evaluate_cross_validation(self, cv_scores, model_name="Model"):
        """
        Evaluate cross-validation results.
        
        Args:
            cv_scores (list): Cross-validation scores
            model_name (str): Name of the model
            
        Returns:
            dict: CV evaluation metrics
        """
        cv_metrics = {
            'model_name': model_name,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_min': np.min(cv_scores),
            'cv_max': np.max(cv_scores),
            'cv_scores': cv_scores
        }
        
        return cv_metrics
    
    def compare_models(self, models_results):
        """
        Compare multiple models performance.
        
        Args:
            models_results (dict): Dictionary with model results
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in models_results.items():
            if 'metrics' in results:
                metrics = results['metrics'].copy()
                comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        return comparison_df
    
    def plot_roc_curves(self, models_results=None, figsize=(12, 8), save_path=None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_results (dict, optional): Models results, uses self.results if None
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        if models_results is None:
            models_results = self.results
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
        
        for (model_name, results), color in zip(models_results.items(), colors):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            plt.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{model_name} (AUC = {auc_score:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self, models_results=None, figsize=(12, 8), save_path=None):
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models_results (dict, optional): Models results, uses self.results if None
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        if models_results is None:
            models_results = self.results
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
        
        for (model_name, results), color in zip(models_results.items(), colors):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            ap_score = average_precision_score(y_true, y_pred_proba)
            
            plt.plot(recall, precision, color=color, linewidth=2,
                    label=f'{model_name} (AP = {ap_score:.4f})')
        
        # Baseline (random classifier)
        baseline = np.mean([results['y_true'] for results in models_results.values()][0])
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Random Classifier (AP = {baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrices(self, models_results=None, figsize=(15, 10), save_path=None):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            models_results (dict, optional): Models results, uses self.results if None
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        if models_results is None:
            models_results = self.results
        
        n_models = len(models_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            if idx >= len(axes):
                break
                
            cm = results['confusion_matrix']
            ax = axes[idx]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Churned', 'Churned'],
                       yticklabels=['Not Churned', 'Churned'])
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_calibration_curves(self, models_results=None, n_bins=10, figsize=(12, 8), save_path=None):
        """
        Plot calibration curves for probability calibration analysis.
        
        Args:
            models_results (dict, optional): Models results, uses self.results if None
            n_bins (int): Number of bins for calibration
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        if models_results is None:
            models_results = self.results
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
        
        for (model_name, results), color in zip(models_results.items(), colors):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    color=color, label=model_name, linewidth=2, markersize=8)
        
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_distributions(self, models_results=None, figsize=(15, 10), save_path=None):
        """
        Plot prediction probability distributions.
        
        Args:
            models_results (dict, optional): Models results, uses self.results if None
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        if models_results is None:
            models_results = self.results
        
        n_models = len(models_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            if idx >= len(axes):
                break
                
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            ax = axes[idx]
            
            # Plot distributions for both classes
            ax.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, 
                   label='Not Churned', color='blue', density=True)
            ax.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, 
                   label='Churned', color='red', density=True)
            
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_model_report(self, model_name, detailed=True):
        """
        Generate a comprehensive model report.
        
        Args:
            model_name (str): Name of the model
            detailed (bool): Whether to include detailed analysis
            
        Returns:
            str: Formatted report
        """
        if model_name not in self.results:
            return f"Model '{model_name}' not found in results."
        
        results = self.results[model_name]
        metrics = results['metrics']
        
        report = f"""
{'='*60}
MODEL EVALUATION REPORT: {model_name.upper()}
{'='*60}

🎯 CLASSIFICATION METRICS:
   • ROC AUC Score:      {metrics['roc_auc']:.4f}
   • Precision:          {metrics['precision']:.4f}
   • Recall:             {metrics['recall']:.4f}
   • F1 Score:           {metrics['f1_score']:.4f}
   • Average Precision:  {metrics['average_precision']:.4f}

📊 PROBABILITY METRICS:
   • Log Loss:           {metrics['log_loss']:.4f}
   • Brier Score:        {metrics['brier_score']:.4f}

🔍 CONFUSION MATRIX:
   • True Negatives:     {metrics['true_negatives']:,}
   • False Positives:    {metrics['false_positives']:,}
   • False Negatives:    {metrics['false_negatives']:,}
   • True Positives:     {metrics['true_positives']:,}

📈 ADDITIONAL METRICS:
   • Specificity:        {metrics['specificity']:.4f}
   • Sensitivity:        {metrics['sensitivity']:.4f}
"""
        
        if detailed:
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            # Additional analysis
            churn_rate = np.mean(y_true)
            pred_churn_rate = np.mean(y_pred_proba)
            
            report += f"""
🔢 DATA ANALYSIS:
   • Actual Churn Rate:   {churn_rate:.4f}
   • Predicted Churn Rate: {pred_churn_rate:.4f}
   • Total Samples:       {len(y_true):,}
   • Churned Samples:     {np.sum(y_true):,}
   • Non-Churned Samples: {len(y_true) - np.sum(y_true):,}

📋 CLASSIFICATION REPORT:
{classification_report(y_true, results['y_pred'])}
"""
        
        report += "=" * 60
        return report
    
    def save_evaluation_results(self, filepath, include_predictions=False):
        """
        Save evaluation results to file.
        
        Args:
            filepath (str): Path to save results
            include_predictions (bool): Whether to include prediction arrays
        """
        results_to_save = {}
        
        for model_name, results in self.results.items():
            model_results = {
                'metrics': results['metrics'],
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
            
            if include_predictions:
                model_results.update({
                    'y_true': results['y_true'].tolist(),
                    'y_pred_proba': results['y_pred_proba'].tolist(),
                    'y_pred': results['y_pred'].tolist()
                })
            
            results_to_save[model_name] = model_results
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"✅ Evaluation results saved to: {filepath}")


class CrossValidationEvaluator:
    """
    Specialized evaluator for cross-validation results following 3rd place solution.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.cv_results = {}
    
    def add_cv_results(self, model_name, cv_scores, oof_predictions, y_true):
        """
        Add cross-validation results for a model.
        
        Args:
            model_name (str): Name of the model
            cv_scores (list): Cross-validation scores
            oof_predictions (array): Out-of-fold predictions
            y_true (array): True labels
        """
        self.cv_results[model_name] = {
            'cv_scores': cv_scores,
            'oof_predictions': oof_predictions,
            'y_true': y_true,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'oof_auc': roc_auc_score(y_true, oof_predictions)
        }
    
    def plot_cv_scores(self, figsize=(12, 8), save_path=None):
        """
        Plot cross-validation scores comparison.
        
        Args:
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        model_names = list(self.cv_results.keys())
        cv_means = [self.cv_results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.cv_results[name]['cv_std'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        
        bars = plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
                      color='lightblue', edgecolor='navy', alpha=0.7)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        plt.title('Cross-Validation Scores Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, cv_means, cv_stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                    f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cv_distributions(self, figsize=(12, 8), save_path=None):
        """
        Plot distribution of CV scores for each model.
        
        Args:
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        data_for_plot = []
        for model_name, results in self.cv_results.items():
            for score in results['cv_scores']:
                data_for_plot.append({'Model': model_name, 'CV_Score': score})
        
        df_plot = pd.DataFrame(data_for_plot)
        
        sns.boxplot(data=df_plot, x='Model', y='CV_Score', palette='Set2')
        sns.stripplot(data=df_plot, x='Model', y='CV_Score', 
                     color='black', alpha=0.6, size=4)
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('ROC AUC Score', fontsize=12)
        plt.title('Cross-Validation Score Distributions', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_cv_summary(self):
        """
        Get summary of cross-validation results.
        
        Returns:
            pd.DataFrame: Summary dataframe
        """
        summary_data = []
        
        for model_name, results in self.cv_results.items():
            summary_data.append({
                'Model': model_name,
                'CV_Mean': results['cv_mean'],
                'CV_Std': results['cv_std'],
                'CV_Min': np.min(results['cv_scores']),
                'CV_Max': np.max(results['cv_scores']),
                'OOF_AUC': results['oof_auc'],
                'Score_Range': f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('CV_Mean', ascending=False)
        
        return summary_df


class EnsembleEvaluator:
    """
    Specialized evaluator for ensemble methods.
    """
    
    def __init__(self):
        self.ensemble_results = {}
    
    def evaluate_ensemble_weights(self, model_weights, model_names):
        """
        Analyze ensemble model weights.
        
        Args:
            model_weights (array): Weights for each model
            model_names (list): Names of the models
            
        Returns:
            pd.DataFrame: Weights analysis
        """
        weights_df = pd.DataFrame({
            'Model': model_names,
            'Weight': model_weights,
            'Weight_Percentage': model_weights * 100
        }).sort_values('Weight', ascending=False)
        
        return weights_df
    
    def plot_ensemble_weights(self, model_weights, model_names, 
                            figsize=(10, 6), save_path=None):
        """
        Plot ensemble model weights.
        
        Args:
            model_weights (array): Weights for each model
            model_names (list): Names of the models
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        weights_df = self.evaluate_ensemble_weights(model_weights, model_names)
        
        plt.figure(figsize=figsize)
        
        # Pie chart
        plt.subplot(1, 2, 1)
        plt.pie(weights_df['Weight'], labels=weights_df['Model'], autopct='%1.1f%%',
               startangle=90, colors=plt.cm.Set3(range(len(weights_df))))
        plt.title('Ensemble Model Weights', fontweight='bold')
        
        # Bar chart
        plt.subplot(1, 2, 2)
        bars = plt.bar(weights_df['Model'], weights_df['Weight_Percentage'], 
                      color=plt.cm.Set3(range(len(weights_df))))
        plt.xlabel('Models')
        plt.ylabel('Weight (%)')
        plt.title('Model Contribution to Ensemble', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights_df['Weight_Percentage']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{weight:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return weights_df


def create_comprehensive_report(evaluator, cv_evaluator=None, ensemble_evaluator=None,
                               output_dir="evaluation_results"):
    """
    Create a comprehensive evaluation report with all plots and metrics.
    
    Args:
        evaluator (ModelEvaluator): Main model evaluator
        cv_evaluator (CrossValidationEvaluator, optional): CV evaluator
        ensemble_evaluator (EnsembleEvaluator, optional): Ensemble evaluator
        output_dir (str): Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Generating comprehensive evaluation report...")
    
    # Generate all plots
    evaluator.plot_roc_curves(save_path=f"{output_dir}/roc_curves.png")
    evaluator.plot_precision_recall_curves(save_path=f"{output_dir}/pr_curves.png")
    evaluator.plot_confusion_matrices(save_path=f"{output_dir}/confusion_matrices.png")
    evaluator.plot_calibration_curves(save_path=f"{output_dir}/calibration_curves.png")
    evaluator.plot_prediction_distributions(save_path=f"{output_dir}/prediction_distributions.png")
    
    # CV plots if available
    if cv_evaluator:
        cv_evaluator.plot_cv_scores(save_path=f"{output_dir}/cv_scores.png")
        cv_evaluator.plot_cv_distributions(save_path=f"{output_dir}/cv_distributions.png")
    
    # Save evaluation results
    evaluator.save_evaluation_results(f"{output_dir}/evaluation_results.json", 
                                     include_predictions=True)
    
    # Generate summary report
    summary_report = "BANK CUSTOMER CHURN PREDICTION - EVALUATION REPORT\n"
    summary_report += "=" * 60 + "\n\n"
    
    # Model comparison
    if len(evaluator.results) > 1:
        comparison_df = evaluator.compare_models(evaluator.results)
        summary_report += "MODEL COMPARISON:\n"
        summary_report += comparison_df.to_string(index=False) + "\n\n"
    
    # Individual model reports
    for model_name in evaluator.results.keys():
        summary_report += evaluator.generate_model_report(model_name, detailed=True)
        summary_report += "\n\n"
    
    # CV summary if available
    if cv_evaluator:
        cv_summary = cv_evaluator.get_cv_summary()
        summary_report += "CROSS-VALIDATION SUMMARY:\n"
        summary_report += cv_summary.to_string(index=False) + "\n\n"
    
    # Save summary report
    with open(f"{output_dir}/evaluation_summary.txt", "w") as f:
        f.write(summary_report)
    
    print(f"✅ Comprehensive evaluation report saved to: {output_dir}")
    print("📁 Generated files:")
    print("   • roc_curves.png")
    print("   • pr_curves.png") 
    print("   • confusion_matrices.png")
    print("   • calibration_curves.png")
    print("   • prediction_distributions.png")
    if cv_evaluator:
        print("   • cv_scores.png")
        print("   • cv_distributions.png")
    print("   • evaluation_results.json")
    print("   • evaluation_summary.txt")


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing evaluation module...")
    
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_proba_1 = np.random.beta(2, 5, n_samples)  # Model 1
    y_pred_proba_2 = np.random.beta(3, 4, n_samples)  # Model 2
    
    # Test evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    metrics_1 = evaluator.evaluate_single_model(y_true, y_pred_proba_1, model_name="Model_1")
    metrics_2 = evaluator.evaluate_single_model(y_true, y_pred_proba_2, model_name="Model_2")
    
    print("✅ Model evaluation completed!")
    print(f"Model 1 AUC: {metrics_1['roc_auc']:.4f}")
    print(f"Model 2 AUC: {metrics_2['roc_auc']:.4f}")
    
    # Test CV evaluator
    cv_evaluator = CrossValidationEvaluator()
    cv_scores_1 = np.random.normal(0.85, 0.02, 10)
    cv_scores_2 = np.random.normal(0.87, 0.03, 10)
    
    cv_evaluator.add_cv_results("Model_1", cv_scores_1, y_pred_proba_1, y_true)
    cv_evaluator.add_cv_results("Model_2", cv_scores_2, y_pred_proba_2, y_true)
    
    print("✅ CV evaluation completed!")
    
    # Test ensemble evaluator
    ensemble_evaluator = EnsembleEvaluator()
    weights = np.array([0.6, 0.4])
    model_names = ["Model_1", "Model_2"]
    weights_df = ensemble_evaluator.evaluate_ensemble_weights(weights, model_names)
    
    print("✅ Ensemble evaluation completed!")
    print(weights_df)
    
    print("\n✅ Evaluation module ready for use!")