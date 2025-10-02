# 🏦 Bank Customer Churn Prediction - Streamlit Web App

A comprehensive web application for predicting bank customer churn using machine learning techniques.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Run the Application
```bash
python run_app.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

### 3. Access the App
Open your web browser and go to: `http://localhost:8501`

## 📱 Features

### 🏠 Project Information Page
- **Project Overview**: Detailed description of the churn prediction project
- **Dataset Summary**: Key statistics about the banking dataset
- **Model Performance**: Comprehensive metrics and evaluation results
- **Technical Implementation**: Details about data processing and model training

### 📊 Interactive Dashboard
- **Key Metrics**: Real-time statistics about customers and churn rates
- **Visualizations**: 
  - Churn distribution pie chart
  - Geographic distribution
  - Age vs Balance analysis
  - Credit Score analysis
  - Product usage patterns
- **Interactive Charts**: Built with Plotly for smooth interactions

### 🔮 Prediction Interface
- **Customer Input Form**: Easy-to-use form for entering customer details
- **Real-time Prediction**: Instant churn probability calculation
- **Risk Assessment**: Color-coded risk levels (Low/Medium/High)
- **Recommendations**: Actionable insights based on prediction results

## 🎨 UI/UX Features

- **Modern Design**: Clean, professional interface with gradient backgrounds
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Color-coded Metrics**: Intuitive visual indicators for different risk levels
- **Smooth Animations**: Hover effects and transitions for better user experience
- **Sidebar Navigation**: Easy switching between different pages

## 🔧 Technical Details

### Model Information
- **Best Model**: LightGBM (Hyperparameter tuned with Optuna)
- **Performance**: ROC-AUC: 0.884, Accuracy: 86.5%, F1-Score: 86.1%
- **Features**: 30 engineered features after transformation and selection
- **Data Processing**: Outlier removal, feature engineering, imbalanced data handling

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with scikit-learn, LightGBM, and joblib
- **Visualization**: Plotly for interactive charts
- **Caching**: Streamlit caching for optimal performance

## 📊 Dataset Information

The application uses a banking dataset with the following features:
- **Personal**: CreditScore, Geography, Gender, Age
- **Financial**: Balance, EstimatedSalary
- **Banking**: Tenure, NumOfProducts, HasCrCard, IsActiveMember
- **Target**: Exited (Churn status)

## 🛠️ Customization

### Adding New Features
1. Modify the `apply_feature_engineering()` function in `app.py`
2. Update the input form in `show_prediction_page()`
3. Ensure feature names match the trained model

### Styling Changes
1. Edit the CSS in the `st.markdown()` section at the top of `app.py`
2. Modify color schemes, fonts, and layout as needed

### Adding New Pages
1. Add new page option to the sidebar selectbox
2. Create corresponding function (e.g., `show_new_page()`)
3. Add page logic to the main navigation

## 📈 Performance

- **Fast Loading**: Cached data and model loading
- **Responsive**: Optimized for different screen sizes
- **Interactive**: Real-time updates and smooth animations
- **Scalable**: Can handle large datasets efficiently

## 🔍 Troubleshooting

### Common Issues

1. **Model not found**: Ensure `smote_saved_models/best_model_tuned_optuna.joblib` exists
2. **Data loading error**: Check if data files are in the correct location
3. **Feature mismatch**: Verify feature engineering matches training pipeline

### Debug Mode
Run with debug information:
```bash
streamlit run app.py --logger.level debug
```

## 📝 Notes

- The app requires the trained model and processed data files
- Feature engineering is simplified for the web interface
- For production use, consider adding input validation and error handling
- The app is designed for demonstration and analysis purposes

## 🤝 Contributing

Feel free to contribute by:
- Adding new visualizations
- Improving the UI/UX
- Adding new prediction features
- Optimizing performance

## 📄 License

This project is part of the Bank Customer Churn Prediction thesis work.
