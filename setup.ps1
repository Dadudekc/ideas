# Portfolio_Optimization_System setup script in PowerShell
# Creates the directory structure with essential files and initial content

# Root project directory
$root = "Portfolio_Optimization_System"

# Define directory structure
$structure = @{
    "$root/config" = @(
        "optimization_params.yaml",
        "risk_constraints.yaml",
        "optimizer_config.yaml",
        "constraints.yaml",
        "paths.yaml"
    )
    "$root/data" = @()
    "$root/data/raw_data" = @()
    "$root/data/processed_data" = @()
    "$root/data/results" = @()
    "$root/data/logs" = @()
    "$root/data/optimization_results" = @()
    "$root/data/simulations" = @()
    "$root/data/risk_reports" = @()
    "$root/notebooks" = @(
        "Portfolio_Optimization_Analysis.ipynb",
        "Optimization_Experiments.ipynb",
        "Backtest_Analysis.ipynb"
    )
    "$root/scripts" = @(
        "run_optimization.py",
        "generate_risk_report.py",
        "run_backtest.py"
    )
    "$root/src" = @()
    "$root/src/data_management" = @(
        "data_cleaner.py",
        "data_manager.py",
        "market_data_loader.py"
    )
    "$root/src/optimization_algorithms" = @(
        "mean_variance_optimizer.py",
        "black_litterman_model.py",
        "risk_parity.py",
        "optimizer.py"
    )
    "$root/src/strategies" = @(
        "mean_variance.py",
        "risk_parity.py",
        "custom_strategy.py"
    )
    "$root/src/risk_analysis" = @(
        "risk_assessor.py",
        "volatility_calculator.py"
    )
    "$root/src/utils" = @(
        "logger.py",
        "performance_tracker.py",
        "report_generator.py",
        "risk_management.py"
    )
    "$root/src/tests" = @(
        "test_optimization_algorithms.py",
        "test_portfolio_optimizer.py",
        "test_risk_assessment.py",
        "test_utils.py"
    )
}

# Create directories and files
foreach ($path in $structure.Keys) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    foreach ($file in $structure[$path]) {
        New-Item -ItemType File -Path "$path/$file" -Force | Out-Null
    }
}

# Create additional essential files at root level
$rootFiles = @("requirements.txt", "README.md", "run_system.py")
foreach ($file in $rootFiles) {
    New-Item -ItemType File -Path "$root/$file" -Force | Out-Null
}

# Initial content for some key files
Set-Content -Path "$root/README.md" -Value "# Portfolio Optimization System`nThis project provides tools for optimizing and backtesting portfolios."
Set-Content -Path "$root/run_system.py" -Value "# Entry point to run Portfolio Optimization System`nif __name__ == '__main__':`n    print('Starting Portfolio Optimization System...')"
Set-Content -Path "$root/config/optimization_params.yaml" -Value "# Optimization Parameters`nmax_iter: 1000`nloss_tolerance: 0.01"
Set-Content -Path "$root/config/risk_constraints.yaml" -Value "# Risk Constraints Configuration`nmax_drawdown: 0.2`nrisk_tolerance: moderate"
Set-Content -Path "$root/config/paths.yaml" -Value "# Path Configuration`ndata_directory: './data/raw_data/'`nlogs_directory: './data/logs/'"

Write-Host "Portfolio Optimization System project structure created successfully."
