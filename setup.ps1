# PowerShell Script to Set Up Project Structures for Market_News_Analyzer and Portfolio_Optimizer_Bot

# Define root project directories
$projects = @("Market_News_Analyzer", "Portfolio_Optimizer_Bot")

# Define folder structures for each project
$structures = @{
    "Market_News_Analyzer" = @(
        "config",
        "data/raw",
        "data/processed",
        "data/reports",
        "src/data_collection",
        "src/text_analysis",
        "src/report_generation",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
    "Portfolio_Optimizer_Bot" = @(
        "config",
        "data/portfolio_data",
        "data/optimization_results",
        "data/risk_reports",
        "src/portfolio_management",
        "src/optimization",
        "src/risk_assessment",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
}

# Define files with initial content for each project
$files = @{
    "Market_News_Analyzer" = @{
        "config/api_keys.yaml" = "# API keys and access tokens for news sources"
        "README.md" = "# Market News Analyzer Project"
        "requirements.txt" = "# List of dependencies"
        "src/data_collection/news_fetcher.py" = "# Script to fetch news articles via API"
        "src/text_analysis/sentiment_analysis.py" = "# Sentiment analysis on news articles"
        "src/text_analysis/topic_modeling.py" = "# Topic modeling to identify trending topics"
        "src/report_generation/generate_report.py" = "# Generate reports on analyzed news data"
        "src/utils/logger.py" = "# Custom logging setup"
        "src/utils/config_loader.py" = "# Utility to load configurations"
        "notebooks/News_Sentiment_Analysis.ipynb" = "# Jupyter Notebook for sentiment analysis exploration"
        "scripts/run_news_analysis.py" = "# Script to run full news analysis pipeline"
        "tests/test_sentiment_analysis.py" = "# Unit tests for sentiment analysis module"
    }
    "Portfolio_Optimizer_Bot" = @{
        "config/optimization_settings.yaml" = "# Configuration for optimization parameters"
        "config/risk_tolerance.yaml" = "# Global risk tolerance settings"
        "README.md" = "# Portfolio Optimizer Bot Project"
        "requirements.txt" = "# List of dependencies"
        "src/portfolio_management/data_manager.py" = "# Manages portfolio data"
        "src/optimization/optimizer.py" = "# Core portfolio optimization engine"
        "src/optimization/constraints.py" = "# Define constraints for portfolio optimization"
        "src/risk_assessment/volatility_calculator.py" = "# Calculates portfolio volatility"
        "src/utils/logger.py" = "# Custom logging setup"
        "src/utils/performance_tracker.py" = "# Track and log optimization performance"
        "notebooks/Portfolio_Optimization_Analysis.ipynb" = "# Jupyter Notebook for portfolio optimization"
        "scripts/run_optimizer.py" = "# Script to start optimization"
        "scripts/generate_risk_report.py" = "# Script to generate risk reports"
        "tests/test_optimizer.py" = "# Unit tests for optimization module"
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

Write-Output "Project structures for Market_News_Analyzer and Portfolio_Optimizer_Bot created successfully."
