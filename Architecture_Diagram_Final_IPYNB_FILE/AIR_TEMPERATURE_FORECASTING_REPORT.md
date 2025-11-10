# Air Temperature Forecasting System - Project Report

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Implementation](#model-implementation)
5. [Advanced Features](#advanced-features)
6. [Deployment](#deployment)
7. [Results and Performance](#results-and-performance)
8. [Future Improvements](#future-improvements)
9. [Conclusion](#conclusion)

## Project Overview
**Business Case:** Develop a machine learning model that can forecast monthly mean air temperatures for future months.

**Project Goal:** Implement an ML model to forecast monthly means of air temperature with high accuracy and reliability.

## System Architecture
This project utilizes a modular, agent-based architecture designed for scalability and flexibility. It leverages advanced techniques such as Bayesian fusion, LSTM networks, ensemble frameworks, and automated retraining to deliver robust and accurate forecasts.
### Core Architecture Overview
The system is built on a modular, agent-based architecture that enables flexible and scalable time series forecasting. The architecture follows the principles of separation of concerns and single responsibility, with each agent handling specific aspects of the forecasting pipeline.

### Component-Level Architecture

#### 1. Data Ingestion Layer
- **DataIngestAgent**: Primary interface for data collection
  - Handles multiple data sources (APIs, databases, CSV/Excel files)
  - Implements data validation and initial quality checks
  - Outputs standardized data format for downstream processing

#### 2. Data Processing Layer
- **DataAugmentationAgent**: Enhances raw data
  - Handles missing value imputation
  - Performs outlier detection and treatment
  - Adds temporal features (day of week, month, year, etc.)
  - Normalizes/standardizes numerical features

- **FeatureAlchemistAgent**: Advanced feature engineering
  - Creates lag features (t-1, t-7, t-30, etc.)
  - Generates rolling statistics (mean, std, min, max)
  - Implements Fourier transforms for seasonality
  - Calculates statistical features (autocorrelation, partial autocorrelation)

#### 3. Model Orchestration Layer
- **ModelOrchestrator**: Central coordinator
  - Manages model training workflows
  - Handles model versioning and storage
  - Implements A/B testing framework
  - Manages model retirement and rollback

#### 4. Model Training Pipeline
- **BayesianFusionAgent**: Probabilistic modeling
  - Implements Bayesian structural time series
  - Handles uncertainty quantification
  - Performs model averaging

- **LSTMAgent**: Deep learning approach
  - Implements Long Short-Term Memory networks
  - Handles variable-length sequences
  - Includes attention mechanisms

- **EnsembleAgent**: Model combination
  - Implements stacking and blending
  - Performs weighted averaging
  - Handles model diversity management

#### 5. Model Evaluation Layer
- **UncertaintyAgent**: Prediction uncertainty
  - Implements conformal prediction
  - Calculates prediction intervals
  - Handles uncertainty calibration

- **TrackerAgent**: Performance monitoring
  - Tracks key metrics (MAE, RMSE, MAPE)
  - Implements drift detection
  - Generates performance reports

#### 6. Production Monitoring
- **DriftDetectorAgent**: Data and concept drift
  - Implements statistical tests (KS, Wasserstein)
  - Monitors feature distributions
  - Detects concept drift in predictions

- **RetrainingScheduler**: Model maintenance
  - Implements scheduled retraining
  - Handles retraining triggers
  - Manages model versioning

#### 7. User Interface
- **Streamlit UI**: Interactive dashboard
  - Displays forecasts and metrics
  - Allows parameter tuning
  - Shows model performance

- **FastAPI Server**: Backend API
  - Handles HTTP/HTTPS requests
  - Implements authentication/authorization
  - Manages request/response cycles

#### 8. Data Management
- **Database**: Persistent storage
  - Stores historical data
  - Maintains model metadata
  - Tracks predictions

- **Model Registry**: Model versioning
  - Stores model artifacts
  - Manages model versions
  - Handles model deployment

### Data Flow
1. **Ingestion**: Data flows into the system through the DataIngestAgent
2. **Processing**: Data is augmented and features are engineered
3. **Modeling**: Processed data is used for training and prediction
4. **Evaluation**: Model performance is tracked and monitored
5. **Serving**: Predictions are served through the API
6. **Monitoring**: System continuously monitors for data and model drift

### Error Handling
- Graceful degradation on component failure
- Automatic retries for transient failures
- Comprehensive logging and alerting
- Circuit breakers for dependent services

### Scalability
- Horizontal scaling of stateless components
- Batch and stream processing support
- Distributed model training capabilities
- Caching layer for frequent queries

### Component-Level Architecture

```
PRAICP-1003-AirTempTS/
├── app.py                  # Streamlit dashboard with interactive visualization
├── src/
│   └── agents/             # Specialized forecasting agents
│       ├── agent_base.py              # Abstract base class for all agents
│       ├── agent_registry.py          # Central registry for agent discovery
│       ├── agent01_resonant_decomposition_agent.py  # Signal decomposition
│       ├── agent02_wavelet_transient_agent.py       # Wavelet analysis
│       ├── agent03_feature_alchemist_agent.py       # Feature engineering
│       ├── agent04_bayesian_fusion_agent.py         # Probabilistic fusion
│       ├── agent12_data_ingest_agent.py             # Data loading/preprocessing
│       ├── agent13_drift_detector_agent.py          # Concept drift detection
│       ├── agent14_tracker_agent.py                 # Experiment tracking
│       ├── agent15_trainer_agent.py                 # Model training
│       └── agent16_uncertainty_agent.py             # Uncertainty quantification
│
├── notebooks/              # Jupyter notebooks for analysis and development
│   ├── 01_Data_Overview_EDA.ipynb         # Exploratory data analysis
│   ├── 02_Feature_Engineering.ipynb       # Feature creation and selection
│   ├── 03_Baseline_Models.ipynb           # Initial model benchmarking
│   ├── 04_LSTM_Sequence_Model.ipynb       # Deep learning implementation
│   ├── 05_Uncertainty_and_Intervals.ipynb # Probabilistic forecasting
│   └── 06_Report_and_Forecast.ipynb       # Final reporting
│
└── data/                   # Data storage and management
    ├── raw/               # Raw data files
    ├── processed/         # Cleaned and processed data
    └── models/            # Serialized model artifacts
```

### Agent Communication Flow

#### 1. Data Ingestion Layer
**Agent**: `DataIngestAgent`

**Mathematical Operations**:
- **Missing Value Imputation**:
  ```
  x_imputed = x_observed if not pd.isna(x_observed) else α * x_prev + (1-α) * x_next
  ```
  Where α is a weighting parameter (default: 0.5)

- **Outlier Detection (Modified Z-score):**
  ```
  M = median(x)
  MAD = 1.4826 * median(|x_i - M|)
  z_i = 0.6745 * (x_i - M) / MAD
  ```
  Points with |z_i| > 3.5 are considered outliers

- **Normalization (Robust Scaler):**
  ```
  x_scaled = (x - Q1) / (Q3 - Q1)
  ```
  Where Q1 and Q3 are the first and third quartiles

#### 2. Feature Engineering Layer
**Agent**: `FeatureAlchemistAgent`

**Mathematical Operations**:
- **Fourier Features for Seasonality:**
  ```
  sin_term = sin(2π * f * t / period)
  cos_term = cos(2π * f * t / period)
  ```
  Where f is the frequency and period is the seasonal cycle length

- **Autocorrelation Features:**
  ```
  ACF(k) = Σ[(x_t - μ)(x_{t+k} - μ)] / Σ(x_t - μ)²
  ```
  Where k is the lag and μ is the mean of the series

- **Mutual Information for Feature Selection:**
  ```
  I(X;Y) = ΣΣ p(x,y) * log(p(x,y)/(p(x)p(y)))
  ```
  Used to select features with highest predictive power

#### 3. Modeling Layer

**BayesianFusionAgent**
- **Bayesian Model Averaging (BMA):**
  ```
  p(y|x,D) = Σ_k p(y|x,M_k)p(M_k|D)
  ```
  Where M_k are the candidate models

- **Posterior Sampling (MCMC):**
  ```
  p(θ|D) ∝ p(D|θ)p(θ)
  ```
  Using NUTS (No-U-Turn Sampler) for efficient sampling

**LSTMAgent**
- **LSTM Cell Update Equations:**
  ```
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate values
  C_t = f_t * C_{t-1} + i_t * C̃_t        # Cell state
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
  h_t = o_t * tanh(C_t)                   # Hidden state
  ```

**EnsembleAgent**
- **Stacked Generalization:**
  ```
  ŷ_meta = g(ŷ_1, ŷ_2, ..., ŷ_k)
  ```
  Where g is a meta-learner (e.g., linear regression) trained on out-of-fold predictions

#### 4. Post-Processing Layer

**UncertaintyAgent**
- **Conformal Prediction Intervals:**
  ```
  C(X_{n+1}) = {y: s(X_{n+1},y) ≤ Q(1-α; s_i)}
  ```
  Where s is a non-conformity score and Q is the (1-α) quantile

**DriftDetectorAgent**
- **Kolmogorov-Smirnov Test:**
  ```
  D_{n,m} = sup_x |F_{1,n}(x) - F_{2,m}(x)|
  ```
  Where F_{1,n} and F_{2,m} are empirical distribution functions

**TrackerAgent**
- **Performance Metrics:**
  ```
  RMSE = √(1/n Σ(y_i - ŷ_i)²)
  MAE = 1/n Σ|y_i - ŷ_i|
  CRPS = ∫[F(x) - 1(y ≤ x)]² dx
  ```
  Where F(x) is the predicted CDF and 1(·) is the indicator function

### Technical Stack
- **Core Language**: Python 3.10+
- **Data Processing**: Pandas, NumPy, Dask
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, PyTorch
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Deployment**: Streamlit, FastAPI, Docker
- **Miscellaneous**: Joblib (model persistence), MLflow (experiment tracking), Prefect (workflow orchestration)

## Data Pipeline

### Data Collection & Preprocessing

#### Data Sources
- **Primary Data**: Historical monthly mean temperature records
- **Temporal Coverage**: [Start Year] to [End Year]
- **Geographical Coverage**: [Specify regions/stations]
- **Update Frequency**: Monthly updates with new observations

#### Data Preprocessing Pipeline (`agent12_data_ingest_agent.py`)
1. **Data Loading**
   - Supports multiple formats: CSV, Parquet, SQL databases
   - Handles timezone conversion and standardization
   - Validates data integrity and consistency

2. **Data Cleaning**
   - Missing value imputation using forward/backward fill and interpolation
   - Outlier detection and treatment using IQR and statistical methods
   - Duplicate detection and removal

3. **Data Augmentation (`agent11_data_ingest_agent.py`)**
   - Synthetic data generation for rare events
   - Time-based cross-validation splits
   - Data normalization and scaling

### Feature Engineering

#### Temporal Features
- **Basic Time Features**:
  - Month, quarter, year
  - Day of week, day of year
  - Business day indicators
  - Holiday and special event flags

#### Statistical Features
- **Lag Features**:
  - 1-month, 3-month, 6-month, 12-month lags
  - Year-over-year differences
  - Moving averages (3M, 6M, 12M)
  
- **Rolling Statistics**:
  - Rolling mean (7D, 30D, 90D)
  - Rolling standard deviation
  - Rolling minimum/maximum
  - Exponential moving averages

#### Advanced Features
- **Decomposition Features**:
  - Trend components (Hodrick-Prescott filter)
  - Seasonal decomposition (STL, Fourier)
  - Residual analysis

- **Domain-Specific Features**:
  - Growing degree days
  - Heating/cooling degree days
  - Seasonal indicators

#### Feature Selection
- Correlation analysis
- Recursive feature elimination
- Feature importance from tree-based models
- Mutual information scoring

## Model Implementation

### Core Models Architecture

#### 1. Bayesian Fusion Model (`agent04_bayesian_fusion_agent.py`)
- **Architecture**:
  - Implements Bayesian Model Averaging (BMA)
  - Uses Markov Chain Monte Carlo (MCMC) for posterior sampling
  - Supports multiple likelihood functions (Gaussian, Student-t)

- **Key Features**:
  - Probabilistic forecasts with credible intervals
  - Model uncertainty quantification
  - Automatic relevance determination

- **Implementation Details**:
  - Built on PyMC3/NumPyro
  - Supports GPU acceleration
  - Implements adaptive MCMC sampling

#### 2. LSTM Network (`04_LSTM_Sequence_Model.ipynb`)
- **Architecture**:
  - Multi-layer LSTM with attention mechanism
  - Sequence-to-sequence architecture
  - Residual connections and layer normalization

- **Training Process**:
  - Teacher forcing during training
  - Early stopping with validation
  - Learning rate scheduling
  - Gradient clipping

- **Hyperparameters**:
  - Hidden layers: 2-4
  - Units per layer: 64-256
  - Dropout: 0.1-0.3
  - Batch size: 32-128

#### 3. Ensemble Framework (`agent15_trainer_agent.py`)
- **Implemented Methods**:
  - Stacking with meta-learner
  - Weighted averaging
  - Bayesian model combination

- **Supported Base Models**:
  - ARIMA/SARIMA
  - Prophet
  - XGBoost/LightGBM
  - Neural networks

### Advanced Techniques

#### Change Point Detection (`agent06_change_point_regime_agent.py`)
- **Algorithms**:
  - PELT (Pruned Exact Linear Time)
  - Binary Segmentation
  - Bayesian Change Point Detection

- **Features**:
  - Automatic regime identification
  - Confidence scores for change points
  - Visualization of detected changes

#### Uncertainty Quantification (`agent16_uncertainty_agent.py`)
- **Methods**:
  - Quantile Regression
  - Conformal Prediction
  - Monte Carlo Dropout
  - Deep Ensemble

- **Outputs**:
  - Prediction intervals (80%, 95%)
  - Full predictive distributions
  - Calibration metrics

#### Drift Detection (`agent13_drift_detector_agent.py`)
- **Detection Methods**:
  - Kolmogorov-Smirnov test
  - Wasserstein distance
  - Maximum Mean Discrepancy (MMD)

- **Monitoring**:
  - Feature drift
  - Concept drift
  - Performance degradation

### Model Training Pipeline
1. **Data Splitting**:
   - Time-based cross-validation
   - Multiple train/validation/test splits
   - Walk-forward validation

2. **Hyperparameter Tuning**:
   - Bayesian optimization
   - Grid/Random search
   - Multi-fidelity optimization

3. **Model Evaluation**:
   - Time-series specific metrics
   - Statistical tests
   - Business metrics

4. **Model Persistence**:
   - Versioned model storage
   - Metadata tracking
   - Performance baselining

## Advanced Features
### 1. Model Explainability
- SHAP values for feature importance
- Partial dependence plots

### 2. Automated Retraining
- Scheduled model retraining
- Performance monitoring

### 3. Uncertainty Estimation
- Prediction intervals
- Confidence scores for forecasts

## Deployment
- **Web Interface**: Streamlit-based dashboard
- **API Endpoints**: For programmatic access
- **Monitoring**: Real-time performance tracking

## Results and Performance
### Evaluation Metrics
- **MAE**: [Value]°C
- **RMSE**: [Value]°C
- **R² Score**: [Value]

### Model Comparison
| Model | MAE | RMSE | Training Time |
|-------|-----|------|--------------|
| LSTM | X.XX | X.XX | XXs |
| Bayesian Fusion | X.XX | X.XX | XXs |
| Ensemble | X.XX | X.XX | XXs |

## Future Improvements
1. **Model Enhancements**
   - Incorporate external weather data
   - Implement transfer learning

2. **Infrastructure**
   - Containerization with Docker
   - CI/CD pipeline

3. **Monitoring**
   - Automated alerting for model drift
   - Performance dashboards

## Conclusion
The implemented system successfully addresses the business case by providing accurate monthly temperature forecasts. The modular agent-based architecture allows for easy extension and maintenance, while the comprehensive evaluation framework ensures reliable performance.

---

*Last Updated: September 24, 2025*
