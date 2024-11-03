# PowerShell Script to Set Up Project Structures for Sentiment_Trading_Engine and Smart_Asset_Allocator

# Define root project directories
$projects = @("Sentiment_Trading_Engine", "Smart_Asset_Allocator")

# Define folder structures for each project
$structures = @{
    "Sentiment_Trading_Engine" = @(
        "config",
        "data/sentiment_data",
        "data/trade_logs",
        "data/reports",
        "src/data_collection",
        "src/sentiment_analysis",
        "src/trading_logic",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
    "Smart_Asset_Allocator" = @(
        "config",
        "data/market_data",
        "data/allocations",
        "data/analysis",
        "src/allocation_strategies",
        "src/data_processing",
        "src/risk_assessment",
        "src/utils",
        "notebooks",
        "scripts",
        "tests"
    )
}

# Define files with initial content for each project
$files = @{
    "Sentiment_Trading_Engine" = @{
        "config/api_keys.yaml" = "# API keys and access tokens for sentiment sources"
        "README.md" = "# Sentiment Trading Engine Project"
        "requirements.txt" = "# List of dependencies"
        "src/data_collection/social_media_fetcher.py" = "# Fetch social media sentiment data"
        "src/sentiment_analysis/sentiment_classifier.py" = "# Classify sentiment for trading decisions"
        "src/trading_logic/strategy_engine.py" = "# Executes trades based on sentiment scores"
        "src/utils/logger.py" = "# Custom logging setup"
        "src/utils/config_loader.py" = "# Utility to load configurations"
        "notebooks/Sentiment_Impact_Analysis.ipynb" = "# Jupyter Notebook for sentiment impact analysis"
        "scripts/run_sentiment_trade.py" = "# Script to run the sentiment-based trading engine"
        "tests/test_sentiment_classifier.py" = "# Unit tests for sentiment classifier module"
    }
    "Smart_Asset_Allocator" = @{
        "config/allocation_settings.yaml" = "# Configuration for allocation strategies"
        "config/risk_profile.yaml" = "# User risk tolerance settings"
        "README.md" = "# Smart Asset Allocator Project"
        "requirements.txt" = "# List of dependencies"
        "src/allocation_strategies/mean_variance_allocator.py" = "# Mean-variance optimization allocator"
        "src/allocation_strategies/risk_parity_allocator.py" = "# Risk parity allocation strategy"
        "src/data_processing/market_data_cleaner.py" = "# Cleans and preprocesses market data"
        "src/risk_assessment/volatility_calculator.py" = "# Calculates asset volatility"
        "src/utils/logger.py" = "# Custom logging setup"
        "src/utils/report_generator.py" = "# Generates reports for asset allocations"
        "notebooks/Allocation_Strategy_Analysis.ipynb" = "# Jupyter Notebook for allocation strategy exploration"
        "scripts/run_allocator.py" = "# Script to start asset allocation process"
        "tests/test_allocation_strategies.py" = "# Unit tests for allocation strategies"
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

Write-Output "Project structures for Sentiment_Trading_Engine and Smart_Asset_Allocator created successfully."
