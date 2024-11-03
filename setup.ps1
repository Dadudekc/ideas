# PowerShell Script to Set Up Project Structures for Stock_Analysis_Tool and Portfolio_Optimizer_Bot

# Define root project directories
$projects = @("Stock_Analysis_Tool", "Portfolio_Optimizer_Bot")

# Define folder structures for each project
$structures = @{
    "Stock_Analysis_Tool" = @(
        "config",
        "data/raw",
        "data/processed",
        "data/reports",
        "src/data_collection",
        "src/analysis",
        "src/visualization",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
    "Portfolio_Optimizer_Bot" = @(
        "config",
        "data/historical",
        "data/simulations",
        "data/results",
        "src/optimization",
        "src/strategies",
        "src/data_processing",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
}

# Define files with initial content for each project
$files = @{
    "Stock_Analysis_Tool" = @{
        "config/settings.yaml" = "# General settings for stock analysis tool"
        "data/logs/.gitkeep" = ""
        "README.md" = "# Stock Analysis Tool Project"
        "requirements.txt" = "# List of dependencies"
        "src/data_collection/data_fetcher.py" = "# Script for data retrieval from financial APIs"
        "src/analysis/financial_ratios.py" = "# Script for calculating financial ratios"
        "src/analysis/technical_indicators.py" = "# Script for technical indicator calculations"
        "src/visualization/chart_generator.py" = "# Script for generating stock charts"
        "src/utils/config_loader.py" = "# Utility to load configurations"
        "src/utils/logger.py" = "# Custom logging setup"
        "notebooks/Stock_Analysis_Overview.ipynb" = "# Jupyter Notebook for stock analysis"
        "scripts/generate_report.py" = "# Generates analysis report"
        "tests/test_financial_ratios.py" = "# Unit tests for financial ratios module"
    }
    "Portfolio_Optimizer_Bot" = @{
        "config/optimizer_config.yaml" = "# Settings for portfolio optimization parameters"
        "data/logs/.gitkeep" = ""
        "README.md" = "# Portfolio Optimizer Bot Project"
        "requirements.txt" = "# List of dependencies"
        "src/optimization/portfolio_optimizer.py" = "# Main portfolio optimization algorithm"
        "src/strategies/risk_parity.py" = "# Implementation of risk parity strategy"
        "src/strategies/mean_variance.py" = "# Implementation of mean-variance optimization"
        "src/data_processing/data_cleaner.py" = "# Cleans and processes raw data for optimization"
        "src/utils/risk_management.py" = "# Risk management utilities"
        "src/utils/logger.py" = "# Custom logging setup"
        "notebooks/Optimization_Experiments.ipynb" = "# Jupyter Notebook for optimization testing"
        "scripts/run_optimization.py" = "# Script to run portfolio optimization with configurations"
        "tests/test_portfolio_optimizer.py" = "# Unit tests for portfolio optimizer"
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

Write-Output "Project structures for Stock_Analysis_Tool and Portfolio_Optimizer_Bot created successfully."
