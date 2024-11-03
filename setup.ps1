# PowerShell Script to Set Up Project Structures for Quantitative_Backtester and Portfolio_Optimization_Tool

# Define root project directories
$projects = @("Quantitative_Backtester", "Portfolio_Optimization_Tool")

# Define folder structures for each project
$structures = @{
    "Quantitative_Backtester" = @(
        "config",
        "data/historical_data",
        "data/results",
        "data/logs",
        "src/data_ingestion",
        "src/strategy_backtesting",
        "src/analysis_tools",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
    "Portfolio_Optimization_Tool" = @(
        "config",
        "data/portfolio_data",
        "data/market_data",
        "data/outputs",
        "src/optimization_algorithms",
        "src/data_management",
        "src/risk_analysis",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
}

# Define files with initial content for each project
$files = @{
    "Quantitative_Backtester" = @{
        "config/backtest_settings.yaml" = "# Settings for backtesting configurations"
        "README.md" = "# Quantitative Backtester Project"
        "requirements.txt" = "# List of dependencies"
        "src/data_ingestion/price_data_loader.py" = "# Loads historical price data"
        "src/strategy_backtesting/backtest_engine.py" = "# Core backtesting logic for strategies"
        "src/analysis_tools/performance_metrics.py" = "# Calculates metrics like Sharpe, drawdown, etc."
        "src/utils/logger.py" = "# Custom logging setup"
        "src/utils/config_loader.py" = "# Utility to load configurations"
        "notebooks/Backtest_Analysis.ipynb" = "# Jupyter Notebook for analyzing backtest results"
        "scripts/run_backtest.py" = "# Script to execute backtest on strategies"
        "tests/test_backtest_engine.py" = "# Unit tests for backtest engine module"
    }
    "Portfolio_Optimization_Tool" = @{
        "config/optimization_params.yaml" = "# Configuration for optimization parameters"
        "config/risk_constraints.yaml" = "# User-defined risk constraints for optimization"
        "README.md" = "# Portfolio Optimization Tool Project"
        "requirements.txt" = "# List of dependencies"
        "src/optimization_algorithms/mean_variance_optimizer.py" = "# Mean-variance optimization algorithm"
        "src/optimization_algorithms/black_litterman_model.py" = "# Black-Litterman model for allocation"
        "src/data_management/data_cleaner.py" = "# Cleans and organizes market data"
        "src/risk_analysis/risk_assessor.py" = "# Evaluates portfolio risk metrics"
        "src/utils/logger.py" = "# Custom logging setup"
        "src/utils/report_generator.py" = "# Generates portfolio optimization reports"
        "notebooks/Optimization_Strategy_Analysis.ipynb" = "# Jupyter Notebook for optimization strategy exploration"
        "scripts/run_optimization.py" = "# Script to start portfolio optimization process"
        "tests/test_optimization_algorithms.py" = "# Unit tests for optimization algorithms"
    }
}

# Create folders and files for each project
foreach ($project in $projects) {
    $rootDir = $project

    # Create folders for the current project
    foreach ($folder in $structures[$project]) {
        $path = "$rootDir/$folder"
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }

    # Create files with initial content for the current project
    foreach ($filePath in $files[$project].Keys) {
        $content = $files[$project][$filePath]
        $fullPath = "$rootDir/$filePath"
        New-Item -ItemType File -Force -Path $fullPath -Value $content | Out-Null
    }
}

Write-Output "Project structures for Quantitative_Backtester and Portfolio_Optimization_Tool created successfully."
