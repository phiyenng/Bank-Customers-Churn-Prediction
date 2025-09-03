# Bank Customer Churn Prediction

Pipeline đơn giản để dự đoán churn khách hàng ngân hàng.

## 📁 Files chính

- `pipeline.py` - Main pipeline script
- `config.yaml` - Configuration file (điều chỉnh tham số ở đây)

## 🚀 Cách chạy

### Chạy với dữ liệu có sẵn:
```bash
python pipeline.py
```

### Chạy với config tùy chỉnh:
Chỉnh sửa file `config.yaml` rồi chạy:
```bash
python pipeline.py
```

## ⚙️ Tùy chỉnh trong config.yaml

### 1. Data settings:
```yaml
data:
  path: "data/raw/Churn_Modelling.csv"  # Đường dẫn data
  target_column: "Exited"                # Tên cột target
  test_size: 0.2                         # Tỷ lệ test set
```

### 2. Preprocessing:
```yaml
preprocessing:
  encode_categorical: "onehot"    # "onehot" hoặc "label"
  scale_features: "standard"      # "standard", "minmax", hoặc "none"
  apply_pca: false               # true/false
```

### 3. Imbalance methods:
```yaml
imbalance_methods:
  - "none"
  - "smote"
  - "oversample"
  - "undersample"
```

### 4. Models:
```yaml
models:
  logistic_regression:
    enabled: true               # Bật/tắt model
    params:
      max_iter: 1000
      
  xgboost:
    enabled: true
    params:
      n_estimators: 100
      max_depth: 6
```

## 📊 Kết quả

Sau khi chạy xong:
- `results/training_results.csv` - Kết quả chi tiết
- `artifacts/best_model.pkl` - Model tốt nhất
- `plots/model_comparison.png` - Biểu đồ so sánh
- `plots/roc_curves.png` - ROC curves

## 💡 Tips

1. **Test nhanh**: Giảm `n_estimators` trong config
2. **Tắt models chậm**: Set `enabled: false` cho CatBoost/XGBoost
3. **Thêm methods**: Thêm `"smoteenn"`, `"smotetomek"` vào `imbalance_methods`
4. **Tune parameters**: Chỉnh các params trong mỗi model

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn PyYAML joblib
pip install xgboost lightgbm catboost imbalanced-learn  # Optional
```
