# 🌡️ PRAICP-1003-AirTempTS

## Air Temperature Forecasting System with Advanced ML & Agent-Based Architecture

A sophisticated machine learning forecasting system designed to predict monthly mean air temperatures with high accuracy and uncertainty quantification. Built for agriculture, energy planning, and urban development sectors to enable data-driven decision-making.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Technologies & Libraries](#-technologies--libraries)
- [Agent-Based System](#-agent-based-system)
- [Notebooks](#-notebooks)
- [Data](#-data)
- [Results & Performance](#-results--performance)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

**Business Case:** Develop a production-ready machine learning model that forecasts monthly mean air temperatures for future periods with quantified uncertainty intervals.

**Project Goal:** Implement an advanced ML forecasting system leveraging:
- Multi-agent architecture for modular, scalable design
- Deep learning (LSTM) and ensemble methods
- Bayesian uncertainty quantification
- Automated feature engineering and model retraining
- Interactive visualization dashboard

**Target Users:** Data scientists, climate researchers, agricultural planners, energy sector analysts, and urban developers.

---

## ✨ Key Features

### Core Capabilities
- **Multi-Model Forecasting**: LSTM, SARIMAX, XGBoost, LightGBM, CatBoost ensemble
- **Agent-Based Architecture**: 16 specialized agents for different forecasting tasks
- **Uncertainty Quantification**: Bayesian methods, conformal prediction, MAPIE
- **Advanced Feature Engineering**: Wavelet transforms, Fourier analysis, lag features
- **Automated Pipeline**: Data ingestion, validation, training, and deployment
- **Interactive Dashboard**: Streamlit-based UI for real-time forecasting and monitoring
- **Drift Detection**: Monitors data and concept drift for model retraining triggers
- **Production-Ready**: Includes model versioning, tracking, and orchestration

### Technical Highlights
- Physics-informed thermal modeling
- Koopman mode decomposition for system dynamics
- Temporal graph resonance analysis
- Change-point and regime detection
- Data augmentation strategies
- Comprehensive EDA with profiling tools

---

## 🏗️ System Architecture

The system employs a **modular, agent-based architecture** following separation of concerns principles:

### Architecture Layers

1. **Data Ingestion Layer**
   - `DataIngestAgent`: Multi-source data collection (APIs, CSV, databases)
   - Data validation and standardization

2. **Data Processing Layer**
   - `DataAugmentationAgent`: Missing value imputation, outlier treatment
   - `FeatureAlchemistAgent`: Advanced feature engineering (lags, rolling stats, Fourier)

3. **Model Training Pipeline**
   - `BayesianFusionAgent`: Probabilistic modeling with uncertainty
   - `LSTMAgent`: Deep learning sequence models
   - `EnsembleAgent`: Model stacking and blending

4. **Model Evaluation Layer**
   - `UncertaintyAgent`: Prediction intervals via conformal prediction
   - `TrackerAgent`: Performance monitoring (MAE, RMSE, MAPE)

5. **Production Monitoring**
   - `DriftDetectorAgent`: Statistical drift detection
   - `RetrainingScheduler`: Automated model maintenance

6. **User Interface**
   - **Streamlit Dashboard**: Interactive forecasting UI
   - **FastAPI Server**: Backend REST API

---

## 📁 Project Structure

```
PRAICP-1003-AirTempTS/
│
├── src/                              # Source code
│   ├── agents/                       # 16 specialized agents
│   │   ├── agent_base.py            # Base agent class
│   │   ├── agent_registry.py        # Agent registry
│   │   ├── autogen_orchestrator.py  # Main orchestrator
│   │   ├── agent01_resonant_decomposition_agent.py
│   │   ├── agent02_wavelet_transient_agent.py
│   │   ├── agent03_feature_alchemist_agent.py
│   │   ├── agent04_bayesian_fusion_agent.py
│   │   ├── agent05_uncertainty_synthesis_agent.py
│   │   ├── agent06_change_point_regime_agent.py
│   │   ├── agent07_tesla_oscillation_agent.py
│   │   ├── agent08_koopman_mode_agent.py
│   │   ├── agent09_physics_informed_thermal_agent.py
│   │   ├── agent10_temporal_graph_resonance_agent.py
│   │   ├── agent11_data_augmentation_agent.py
│   │   ├── agent12_data_ingest_agent.py
│   │   ├── agent13_drift_detector_agent.py
│   │   ├── agent14_tracker_agent.py
│   │   ├── agent15_trainer_agent.py
│   │   ├── agent16_uncertainty_agent.py
│   │   ├── constants.py             # Configuration constants
│   │   └── utils.py                 # Utility functions
│   │
│   ├── data_loader.py               # Data loading utilities
│   ├── features.py                  # Feature engineering (23KB)
│   └── __init__.py
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_Data_Overview_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_Models.ipynb
│   ├── 04_LSTM_Sequence_Model.ipynb
│   ├── 05_Uncertainty_and_Intervals.ipynb
│   ├── 06_Report_and_Forecast.ipynb
│   ├── PRAICP-1003-AirTempTS-Final-Files.ipynb
│   └── PRAICP-1003-AirTempTS-Final-File_Update.ipynb
│
├── data/                             # Datasets
│   ├── surface-air-temperature-monthly-mean.csv
│   ├── surface-air-temperature-monthly-mean-extended.csv
│   └── metadata-surface-air-temperature-monthly-mean.txt
│
├── results/                          # Model outputs
│   ├── csv_files/                   # Exported CSV results
│   ├── figures/                     # Visualization plots
│   ├── models/                      # Trained model artifacts
│   ├── logs/                        # Training logs
│   ├── model_mapping.json           # Model registry
│   ├── trend_summary.json           # Trend analysis
│   └── forecasting_model_comparison.csv
│
├── static/                           # Static assets (if needed)
│
├── streamlit_app.py                  # Streamlit dashboard (27KB)
├── index.html                        # Web interface
├── setup.py                          # Package setup
├── setup_and_run.ps1                 # PowerShell automation script
├── requirements.txt                  # Core dependencies
├── requirements-advanced.txt         # Advanced libraries
├── .gitignore                        # Git ignore rules
├── airtempts.db                      # SQLite database
│
├── AIR_TEMPERATURE_FORECASTING_REPORT.md     # Detailed project report
├── AirTemp_Forecasting_Analysis_Report_20240925.md
├── technologies.md                   # Technology documentation
├── LICENSE                           # Project license
└── README.md                         # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ (Python 3.10 recommended)
- pip package manager
- Virtual environment (recommended)

### Quick Start

#### Option 1: Automated Setup (Windows PowerShell)
```powershell
# Run the automated setup script
.\setup_and_run.ps1
```

#### Option 2: Manual Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd PRAICP-1003-AirTempTS
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv .Airtemp
   .Airtemp\Scripts\activate

   # Linux/Mac
   python3 -m venv .Airtemp
   source .Airtemp/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Core requirements
   pip install -r requirements.txt

   # Advanced features (optional)
   pip install -r requirements-advanced.txt
   ```

4. **Install Package in Development Mode**
   ```bash
   pip install -e .
   ```

---

## 💻 Usage

### 1. Run Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Access the dashboard at `http://localhost:8501`

### 2. Run Agent Orchestrator
```bash
python src/agents/autogen_orchestrator.py --root . --run-agents --run-csv "data/surface-air-temperature-monthly-mean.csv"
```

### 3. Execute Jupyter Notebooks
```bash
jupyter notebook
# Navigate to notebooks/ directory and run sequentially:
# 01_Data_Overview_EDA.ipynb → ... → 06_Report_and_Forecast.ipynb
```

### 4. Run Individual Agents
```bash
python list_agents.py          # List all registered agents
python run_agents.py           # Run specific agents
python run_orchestrator.py     # Run full orchestration
```

### 5. Test Imports & Debugging
```bash
python test_imports.py         # Verify all imports work
python import_debug.py         # Debug import issues
```

---

## 🛠️ Technologies & Libraries

### Core Stack
- **Python 3.10**: Primary language
- **Pandas, NumPy**: Data manipulation
- **Scikit-learn**: Classical ML algorithms
- **Statsmodels**: Statistical modeling
- **PyTorch/TensorFlow**: Deep learning frameworks

### Time Series Specific
- **LSTM Networks**: Sequential deep learning
- **SARIMAX**: Seasonal AutoRegressive models
- **Prophet / NeuralProphet**: Facebook's forecasting
- **pmdarima**: Auto-ARIMA implementation
- **Darts**: Unified forecasting framework
- **Kats**: Meta's forecasting toolkit

### Advanced Analytics
- **XGBoost, LightGBM, CatBoost**: Gradient boosting
- **MAPIE**: Conformal prediction for uncertainty
- **PyWavelets (pywt)**: Wavelet transforms
- **Ruptures**: Change-point detection
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model explainability
- **Boruta**: Feature selection

### Visualization & UI
- **Streamlit**: Interactive dashboard
- **Plotly**: Interactive plots
- **Matplotlib, Seaborn**: Static visualizations
- **Sweetviz**: Automated EDA reports

### Development Tools
- **Jupyter**: Interactive notebooks
- **tqdm**: Progress bars
- **joblib**: Model serialization

See `technologies.md` for detailed documentation on each library.

---

## 🤖 Agent-Based System

The project implements **16 specialized agents**, each handling specific forecasting tasks:

### Data & Feature Agents
1. **ResonantDecompositionAgent** - Time series decomposition
2. **WaveletTransientAgent** - Wavelet-based feature extraction
3. **FeatureAlchemistAgent** - Advanced feature engineering
4. **DataAugmentationAgent** - Data preprocessing and augmentation
5. **DataIngestAgent** - Multi-source data ingestion

### Modeling Agents
6. **BayesianFusionAgent** - Bayesian probabilistic modeling
7. **TeslaOscillationAgent** - Oscillation pattern detection
8. **KoopmanModeAgent** - Koopman operator analysis
9. **PhysicsInformedThermalAgent** - Physics-based constraints

### Analysis & Detection Agents
10. **ChangePointRegimeAgent** - Regime shift detection
11. **TemporalGraphResonanceAgent** - Graph-based temporal analysis
12. **DriftDetectorAgent** - Data/concept drift monitoring

### Production Agents
13. **UncertaintySynthesisAgent** - Uncertainty quantification
14. **TrackerAgent** - Performance tracking
15. **TrainerAgent** - Model training orchestration
16. **UncertaintyAgent** - Prediction interval generation

Each agent extends `BaseAgent` and registers with the `AgentRegistry` for orchestration.

---

## 📓 Notebooks

The project includes a sequential workflow across 6 main notebooks:

1. **01_Data_Overview_EDA.ipynb** (11.4 MB)
   - Exploratory Data Analysis
   - Tools: Pandas Profiling, Sweetviz, Missingno

2. **02_Feature_Engineering.ipynb** (1.8 MB)
   - Lag features, rolling statistics
   - Fourier transforms, wavelet decomposition

3. **03_Baseline_Models.ipynb** (611 KB)
   - Classical ML: XGBoost, LightGBM, CatBoost
   - Model comparison and hyperparameter tuning

4. **04_LSTM_Sequence_Model.ipynb** (2.1 MB)
   - Deep learning with LSTM networks
   - Attention mechanisms

5. **05_Uncertainty_and_Intervals.ipynb** (1.8 MB)
   - Conformal prediction
   - Bayesian uncertainty quantification

6. **06_Report_and_Forecast.ipynb** (1.2 MB)
   - Final forecasts and reporting
   - Model deployment preparation

**Main Integration Notebooks:**
- `PRAICP-1003-AirTempTS-Final-Files.ipynb` (18.9 MB)
- `PRAICP-1003-AirTempTS-Final-File_Update.ipynb` (18.9 MB)

---

## 📊 Data

### Primary Dataset
- **File**: `surface-air-temperature-monthly-mean.csv` (6.4 KB)
- **Extended**: `surface-air-temperature-monthly-mean-extended.csv` (150 KB)
- **Metadata**: `metadata-surface-air-temperature-monthly-mean.txt`

### Data Characteristics
- **Frequency**: Monthly mean temperatures
- **Features**: Date, temperature, derived temporal features
- **Target**: Monthly mean air temperature

### Database
- **airtempts.db** (16 KB SQLite): Stores agent results, model metadata, and tracking information

---

## 📈 Results & Performance

Results are stored in the `results/` directory:

- **Model Artifacts**: Trained models in `models/`
- **CSV Exports**: Predictions and metrics in `csv_files/`
- **Visualizations**: Plots and charts in `figures/`
- **Logs**: Training and evaluation logs in `logs/`

### Key Outputs
- `model_mapping.json`: Registry of trained models
- `trend_summary.json`: Trend analysis results
- `forecasting_model_comparison.csv`: Performance comparison
- `TempoAgent.json`: Temporal agent results

### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Prediction Intervals (50%, 95%)

---

## 📚 Documentation

### Detailed Reports
1. **AIR_TEMPERATURE_FORECASTING_REPORT.md** (16 KB)
   - Comprehensive project documentation
   - Architecture details
   - Model implementation guide

2. **AirTemp_Forecasting_Analysis_Report_20240925.md** (8.2 KB)
   - Analysis report with findings
   - Model performance evaluation

3. **technologies.md** (4.7 KB)
   - Technology stack documentation
   - Library-specific guidance

### Additional Resources
- Solution files: `LSTM SOLUTION.txt`, `advance algorithm solution.txt`
- Baseline comparisons: `Baseline technologeis.txt`
- Analysis outputs: `temp_md.md` (12.3 MB)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all imports work correctly

---

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.

---

## 🔧 Troubleshooting

### Import Issues
```bash
python fix_imports.py          # Fix import paths
python fix_all_imports.py      # Comprehensive import fix
python import_debug.py         # Debug import errors
```

### Common Issues
1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Path issues**: Ensure `src/` is in PYTHONPATH
3. **Data not found**: Verify data files are in `data/` directory
4. **Agent errors**: Check `src/agents/__init__.py` for registry

---

## 📞 Contact & Support

For questions, issues, or contributions:
- Open an issue on the repository
- Check documentation in `AIR_TEMPERATURE_FORECASTING_REPORT.md`
- Review notebook outputs for detailed analysis

---

## 🎓 Project Context

**Project ID**: PRAICP-1003  
**Domain**: AI Engineering - Time Series Forecasting  
**Application**: Climate Analytics, Agriculture Planning, Energy Forecasting  
**Status**: Production-Ready with Continuous Improvement

---

**Built with ❤️ for accurate and reliable temperature forecasting**
