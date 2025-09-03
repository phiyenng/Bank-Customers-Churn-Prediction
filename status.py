"""
Project Summary & Usage Guide
=============================

Hiển thị tổng quan về project sau khi đơn giản hóa.
"""

import pandas as pd
import os

def show_project_status():
    """Hiển thị tình trạng hiện tại của project."""
    print("🏦 BANK CUSTOMER CHURN PREDICTION - SIMPLIFIED VERSION")
    print("=" * 70)
    
    print("\n📁 PROJECT STRUCTURE (Simplified):")
    print("-" * 40)
    print("📄 pipeline.py        - Main training pipeline")
    print("⚙️  config.yaml       - Configuration file") 
    print("📖 README_SIMPLE.md   - Simple usage guide")
    print("📊 results/           - Training results")
    print("🤖 artifacts/         - Trained models")
    print("📈 plots/             - Visualizations")
    
    print("\n🎯 HOW TO USE:")
    print("-" * 40)
    print("1️⃣ Basic run:           python pipeline.py")
    print("2️⃣ Customize:          Edit config.yaml → python pipeline.py")
    print("3️⃣ Check results:      Check results/ and plots/ directories")
    
    # Check if training has been done
    if os.path.exists("results/training_results.csv"):
        print("\n📊 LATEST TRAINING RESULTS:")
        print("-" * 40)
        
        df = pd.read_csv("results/training_results.csv")
        print(f"🔢 Total models trained: {len(df)}")
        
        # Show top 5
        top_5 = df.nlargest(5, 'roc_auc')
        print("\n🏆 Top 5 Models:")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"  {i}. {row['combination']:25} ROC-AUC: {row['roc_auc']:.4f}  F1: {row['f1_score']:.4f}")
        
        best = df.loc[df['roc_auc'].idxmax()]
        print(f"\n🥇 Best Model: {best['combination']}")
        print(f"   📈 ROC-AUC: {best['roc_auc']:.4f}")
        print(f"   📈 F1-Score: {best['f1_score']:.4f}")
        print(f"   📈 Precision: {best['precision']:.4f}")
        print(f"   📈 Recall: {best['recall']:.4f}")
        
        # Model comparison
        print(f"\n📊 PERFORMANCE BY MODEL TYPE:")
        model_avg = df.groupby('model')[['roc_auc', 'f1_score']].mean().round(4)
        print(model_avg.to_string())
        
        print(f"\n⚖️  PERFORMANCE BY IMBALANCE METHOD:")
        imbalance_avg = df.groupby('imbalance_method')[['roc_auc', 'f1_score']].mean().round(4)
        print(imbalance_avg.to_string())
        
    else:
        print("\n⚠️  No training results found. Run 'python pipeline.py' first!")
    
    print(f"\n💡 QUICK TIPS:")
    print("-" * 40)
    print("• Fast test: Set small n_estimators in config.yaml")
    print("• Skip slow models: Set enabled: false for catboost/xgboost")
    print("• Add more methods: Add 'smoteenn', 'smotetomek' to imbalance_methods")
    print("• Tune parameters: Modify params in models section")
    
    print(f"\n📋 AVAILABLE FEATURES:")
    print("-" * 40)
    print("✅ Data preprocessing (cleaning, encoding, scaling)")
    print("✅ Imbalance handling (SMOTE, oversampling, undersampling)")
    print("✅ Multiple ML models (Logistic, RF, XGBoost, LightGBM, CatBoost)")
    print("✅ Automated evaluation (ROC-AUC, F1, Precision, Recall)")
    print("✅ Visualizations (model comparison, ROC curves)")
    print("✅ Easy configuration via YAML file")

def show_config_help():
    """Hướng dẫn cấu hình."""
    print(f"\n⚙️  CONFIGURATION HELP:")
    print("=" * 50)
    
    print(f"\n🎯 Quick config examples:")
    print("-" * 30)
    
    print("# For fast testing:")
    print("models:")
    print("  logistic_regression:")
    print("    enabled: true")
    print("  catboost:")
    print("    enabled: false  # Skip slow model")
    print("    params:")
    print("      iterations: 50  # Reduce for speed")
    
    print("\n# For best performance:")
    print("imbalance_methods:")
    print("  - 'smote'")
    print("  - 'smoteenn'")
    print("  - 'oversample'")
    print("preprocessing:")
    print("  apply_pca: true")
    
    print("\n# For specific data:")
    print("data:")
    print("  path: 'your_data.csv'")
    print("  target_column: 'your_target'")
    print("  test_size: 0.3")

def main():
    """Main function."""
    show_project_status()
    
    response = input(f"\n❓ Show configuration help? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        show_config_help()
    
    print(f"\n🎉 Ready to go! Run 'python pipeline.py' to start training.")

if __name__ == "__main__":
    main()
