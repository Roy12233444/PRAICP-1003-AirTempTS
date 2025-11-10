# Setup and Run Script for AirTempTS Project

# Function to check if a command exists
function Test-CommandExists {
    param($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# Check if Python is installed
if (-not (Test-CommandExists "python")) {
    Write-Host "Python is not installed or not in PATH. Please install Python 3.8+ and try again." -ForegroundColor Red
    exit 1
}

# Get Python version
$pythonVersion = python --version
Write-Host "Using Python: $pythonVersion" -ForegroundColor Green

# Remove existing virtual environment if it exists
$venvPath = ".\.Airtemp"
if (Test-Path $venvPath) {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Path $venvPath -Recurse -Force
}

# Create new virtual environment
Write-Host "Creating new virtual environment..." -ForegroundColor Cyan
python -m venv .Airtemp

# Activate the virtual environment
$activatePath = ".\.Airtemp\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Host "Failed to create virtual environment. Please check Python installation." -ForegroundColor Red
    exit 1
}

# Activate the environment and install requirements
Write-Host "Activating virtual environment and installing requirements..." -ForegroundColor Cyan
. $activatePath

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
$packages = @(
    "pandas",
    "numpy",
    "statsmodels",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "seaborn",
    "tqdm"
)

foreach ($pkg in $packages) {
    Write-Host "Installing $pkg..." -ForegroundColor Cyan
    pip install $pkg
}

# Run the orchestrator
Write-Host "\nStarting the orchestrator..." -ForegroundColor Green
$runCmd = "python src/agents/autogen_orchestrator.py --root . --run-agents --run-csv \"data/surface-air-temperature-monthly-mean.csv\" --load-results --mapping-file \"results/model_mapping.json\""
Write-Host "Running: $runCmd" -ForegroundColor Yellow
Invoke-Expression $runCmd

# Keep the window open to see the output
Write-Host "\nPress any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
