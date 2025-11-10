# Technologies for AirTempTs Notebooks

This file lists unique and advanced technologies/libraries recommended for each notebook in the AirTempTs project, based on their focus areas. These are Python-based tools suitable for time series analysis and forecasting (e.g., air temperature data). Each includes a brief description of its advanced/unique aspects.

## 01_Data_Overview_EDA.ipynb (Exploratory Data Analysis)
- **Pandas Profiling**: Automated EDA reports with statistics and visualizations in one go.
- **Sweetviz**: Comparison-based EDA with HTML reports for quick insights.
- **Plotly**: Interactive, web-based visualizations for dynamic data exploration.
- **D-Tale**: Web-based UI for exploring dataframes with advanced filtering and correlations.
- **Ydata Profiling**: Formerly Pandas Profiling, with advanced statistical tests and data quality checks.
- **Missingno**: Visualizes missing data patterns in matrices and heatmaps for quick diagnosis.
- **PhiK**: Correlation analysis for mixed data types, better than Pearson for categorical/numerical combos.

## 02_Feature_Engineering.ipynb (Feature Engineering)
- **Featuretools**: Automated feature engineering for relational data, using deep feature synthesis.
- **Tslearn**: Time series-specific transformations like shapelets and DTW.
- **AutoFeat**: Automatic feature selection and generation using genetic algorithms.
- **Boruta**: Feature selection via random forest importance, wrapped around Scikit-learn.
- **Sktime**: Scikit-learn-like API for time series feature extraction, including Fourier transforms.
- **Category Encoders**: Advanced encoding techniques like target encoding and binary for categorical features.
- **Pytorch Forecasting**: Neural feature engineering with embeddings for time series variables.

## 03_Baseline_Models.ipynb (Baseline Models)
- **XGBoost**: Gradient boosting with optimized speed and accuracy.
- **LightGBM**: Microsoft's fast gradient boosting with categorical support.
- **CatBoost**: Yandex's boosting algorithm handling categorical variables natively.
- **Scikit-learn Pipelines**: Advanced stacking and grid search with hyperparameter tuning.
- **H2O.ai AutoML**: Automated machine learning with ensemble stacking and hyperparameter optimization.
- **TPOT**: Genetic programming to evolve optimal ML pipelines automatically.
- **InterpretML**: Explainable boosting models with feature importance and partial dependence plots.

## 04_LSTM_Sequence_Model.ipynb (LSTM Sequence Model)
- **TensorFlow/Keras**: With LSTM layers and attention mechanisms.
- **PyTorch**: With LSTMCell and TransformerEncoder for sequence modeling.
- **Temporal Fusion Transformer (TFT)**: Via Darts library, combining LSTM with attention for forecasting.
- **GluonTS**: AWS's deep learning toolkit for time series with pre-built LSTM models.
- **Hugging Face Transformers**: Pre-trained models adapted for time series, like Informer or Autoformer.
- **Skorch**: Scikit-learn wrapper for PyTorch, enabling grid search on neural nets.
- **FastAI**: High-level deep learning library with time series-specific callbacks and learning rate finders.

## 05_Uncertainty_and_Intervals.ipynb (Uncertainty and Intervals)
- **MAPIE**: Conformal prediction for uncertainty quantification on any model.
- **TensorFlow Probability**: Probabilistic programming with Bayesian layers.
- **PyMC3**: Bayesian modeling for posterior distributions and credible intervals.
- **Uncertainty Toolbox**: Quantifies uncertainty in neural network predictions.
- **Pyro**: Probabilistic programming with variational inference for Bayesian uncertainty.
- **Conformal Prediction**: Via libraries like crepes, for distribution-free prediction intervals.
- **Uncertainty Quantification Toolbox (UQ-Box)**: With metrics for aleatoric/epistemic uncertainty.

## 06_Report_and_Forecast.ipynb (Report and Forecast)
- **Prophet**: Facebook's additive model for forecasting with seasonality.
- **NeuralProphet**: Neural network extension of Prophet with deep learning.
- **Darts**: Unified framework for forecasting with multiple models like N-BEATS.
- **PyCaret**: Low-code ML for time series forecasting with auto-tuning.
- **Kats**: Meta's toolkit for forecasting with anomaly detection and multivariate support.
- **GluonTS**: Deep learning forecasting with pre-built estimators and evaluation metrics.
- **Pmdarima**: Auto-ARIMA with seasonal decomposition and exogenous variables.

These technologies emphasize automation, scalability, and advanced modeling for time series tasks. For installation, refer to `requirements.txt` in the project root. If you need code examples or implementation help, let me know!