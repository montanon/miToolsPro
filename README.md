# miToolsPro

<img src="assets/mitoolspro-banner.png" width="1280" alt="miToolsPro">

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-77%25-green.svg)](./coverage_html/index.html)

A comprehensive Python toolkit for data analysis, visualization, and research workflows. Features 17 specialized modules covering plotting, econometric modeling, clustering analysis, economic complexity, document processing, and modern API integrations.

## Installation

```bash
pip install mitoolspro
```

**Requirements:** Python 3.12+

## Quick Start

```python
from mitoolspro.plotting import LinePlotter
from mitoolspro.clustering import kmeans_clustering
import numpy as np

# Create sample data
data = np.random.rand(100, 2)

# Create a line plot
plotter = LinePlotter(x_data=data[:, 0], y_data=data[:, 1])
plotter.plot()

# Perform clustering
model, labels = kmeans_clustering(data, n_clusters=3)
```

## Core Modules

### 📊 Plotting (`mitoolspro.plotting`)

Professional-grade plotting with matplotlib compatibility, type validation, and composable architecture.

**Available Plotters:**
- `BarPlotter`, `BoxPlotter`, `ScatterPlotter`, `LinePlotter`
- `HistogramPlotter`, `PiePlotter`, `SankeyPlotter`
- `DistributionPlotter`, `ErrorPlotter`

**Features:**
- Type-safe parameter validation with Pydantic
- Plot composition with `PlotComposer`
- Mixin architecture for shared functionality
- matplotlib API compatibility

```python
from mitoolspro.plotting import PlotComposer, BarPlotter, LinePlotter

# Create individual plots
bar = BarPlotter(x_data=['A', 'B', 'C'], y_data=[1, 2, 3])
line = LinePlotter(x_data=[1, 2, 3], y_data=[1, 4, 2])

# Compose multiple plots
composer = PlotComposer()
composer.add_plot(bar, position=(0, 0))
composer.add_plot(line, position=(0, 1))
composer.compose(figsize=(12, 6))
```

### 📈 Econometric Analysis (`mitoolspro.regressions`)

Professional econometric modeling with comprehensive diagnostic tools.

**Model Types:**
- **OLS**: Ordinary Least Squares with robust standard errors
- **Panel**: Fixed/Random effects, time series cross-sectional data
- **IV**: Instrumental Variables, Two-Stage Least Squares
- **Quantile**: Quantile regression across multiple quantiles
- **Regime**: Regime switching and structural break models
- **Factor**: Multi-factor models and principal components

```python
from mitoolspro.regressions.linear_models import OLSModel
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Fit OLS model
model = OLSModel(
    data=data,
    dependent_variable='price',
    independent_variables=['size', 'location'],
    control_variables=['year']
)
results = model.fit()
print(results.results.summary())
```

**Advanced Quantile Regression:**
```python
from mitoolspro.regressions.managers import QuantilesRegression
from mitoolspro.regressions.wrappers.linear_models import QuantilesRegressionSpecs

# Define regression specifications
specs = QuantilesRegressionSpecs(
    dataframe=data,
    dependent_variable='price',
    independent_variables=['size', 'location']
)

# Run quantile regression across multiple quantiles
qr = QuantilesRegression(specs)
results = qr.fit(quantiles=[0.25, 0.5, 0.75])
```

### 🎯 Clustering Analysis (`mitoolspro.clustering`)

Complete clustering pipeline with evaluation metrics and visualization.

**Algorithms:**
- K-means clustering with automatic optimization
- Agglomerative hierarchical clustering
- Automatic cluster number detection

**Evaluation & Visualization:**
- Silhouette analysis and scoring
- Centroid calculation and visualization
- Distance metrics and similarity measures
- Growth analysis and cluster size tracking

```python
from mitoolspro.clustering import clustering_ncluster_search, plot_clusters_growth
import numpy as np

# Generate sample data
data = np.random.rand(200, 4)

# Find optimal number of clusters
best_n, results = clustering_ncluster_search(data, n_range=(2, 10))
print(f"Optimal clusters: {best_n}")

# Visualize cluster analysis
plot_clusters_growth(results)
```

### 🌍 Economic Complexity Analysis (`mitoolspro.economic_complexity`)

Advanced tools for trade analysis and economic complexity calculations.

**Core Functions:**
- **ECI/PCI Calculation**: Economic and Product Complexity Indices
- **RCA Analysis**: Revealed Comparative Advantage matrices
- **Proximity Networks**: Product and country similarity analysis
- **GPU Acceleration**: PyTorch integration for large datasets

```python
from mitoolspro.economic_complexity import (
    calculate_economic_complexity,
    calculate_proximity_matrix,
    exports_data_to_matrix
)
import pandas as pd

# Process trade data
trade_data = pd.read_csv('trade_data.csv')
rca_matrix = calculate_exports_matrix_rca(trade_data, 'country', 'product', 'value')

# Calculate complexity indices
eci, pci = calculate_economic_complexity(rca_matrix.values)

# Build proximity networks
proximity = calculate_proximity_matrix(rca_matrix.values)
```

### 🤖 LLM Integration (`mitoolspro.llms`)

Production-ready LLM clients with usage tracking and cost management.

**Supported Providers:**
- **OpenAI**: GPT models with structured output support
- **Ollama**: Local LLM deployment and management

**Features:**
- Token usage tracking and cost calculation
- Persistent usage history across sessions
- Model registry with pricing information
- Beta features support (structured outputs)

```python
from mitoolspro.llms import OpenAIClient, PersistentTokensCounter

# Set up usage tracking
counter = PersistentTokensCounter(
    file_path="usage.json", 
    source="openai", 
    model="gpt-4o-mini"
)

# Create client with usage tracking
client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o-mini",
    counter=counter
)

# Make requests with automatic cost tracking
response = client.request("Analyze this data trend...")
print(f"Total cost: ${counter.calculate_total_cost():.4f}")
```

### 📄 Document Processing (`mitoolspro.document`)

Extract and analyze content from PDF and DOCX documents.

**PDF Processing:**
- Text extraction with layout preservation
- Document structure analysis (pages, blocks, lines)
- Font and formatting detection
- Metadata extraction

**Document Generation:**
- DOCX file creation and manipulation
- Text styling and formatting
- Table and image insertion

```python
from mitoolspro.document import pdf_to_document
from mitoolspro.document.write_document import create_docx_document

# Extract structured content from PDF
document = pdf_to_document("report.pdf")
for page in document.pages:
    print(f"Page {page.page_number}: {len(page.lines)} lines")

# Create new DOCX document
create_docx_document(
    filename="output.docx",
    title="Analysis Report",
    content_blocks=[("heading", "Results"), ("paragraph", "Analysis complete.")]
)
```

## Specialized Modules

### 🌐 Google API Integration (`mitoolspro.google_utils`)

**Places API:**
- Location search and analysis
- Geospatial data processing
- Business saturation studies

**YouTube API:**
- Video download and conversion
- Metadata extraction
- Batch processing workflows

### 🕸️ Networks (`mitoolspro.networks`)
Interactive network visualization with pyvis integration.

### 🗄️ Databases (`mitoolspro.databases`)
SQLAlchemy and SQLite utilities for data persistence.

### 📁 Files (`mitoolspro.files`)
Multi-format file handlers: Excel, PDF, ICS, and document conversion.

### 🔤 NLP (`mitoolspro.nlp`)
Text processing with spaCy and Transformers integration.

### 🕷️ Scraping (`mitoolspro.scraping`)
Web scraping tools with Selenium automation.

### 🛠️ Utilities (`mitoolspro.utils`)
Development tools, decorators, and context managers.

## Example Notebooks

Comprehensive examples in the `examples/` directory:

**Plotting:** [`bar_plotter.ipynb`](examples/plotting/bar_plotter.ipynb), [`composer.ipynb`](examples/plotting/composer.ipynb)

**Analysis:** [`clustering.ipynb`](examples/clustering.ipynb), [`networks.ipynb`](examples/networks.ipynb)

**Regression:** [`ols.ipynb`](examples/regressions/ols.ipynb), [`ivars.ipynb`](examples/regressions/ivars.ipynb)

## Development

```bash
# Clone and install for development
git clone https://github.com/montanon/miToolsPro.git
cd miToolsPro
uv sync --group dev

# Run tests with coverage
uv run pytest tests/ --cov=mitoolspro

# Generate coverage report
uv run coverage html
```

## Technical Details

- **77% Test Coverage** across 94 test files
- **Type Safety** with comprehensive annotations
- **Exception Handling** with 83 custom exception classes
- **Modern Architecture** with lazy loading and abstract base classes
- **Performance** optimized with parallel processing support

## Dependencies

**Core Stack:**
- **Data:** pandas, numpy, scipy
- **Visualization:** matplotlib, seaborn, plotly
- **ML/Stats:** scikit-learn, statsmodels, torch
- **Documents:** pymupdf, python-docx, pdfminer
- **Web:** seleniumbase, requests

**Full dependency list:** See [`pyproject.toml`](pyproject.toml)

## License

MIT License - see [LICENSE](LICENSE) file.

Copyright (c) 2025 Sebastián Montagna

## Support

- **Documentation:** Coming soon
- **Issues:** [GitHub Issues](https://github.com/montanon/miToolsPro/issues)
