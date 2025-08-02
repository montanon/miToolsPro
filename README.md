# miToolsPro

<img src="assets/mitoolspro-banner.png" width="1280" alt="miToolsPro">


[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-77%25-green.svg)](./coverage_html/index.html)

A Python package providing tools for data analysis, visualization, and research workflows. Includes 17 modules covering plotting, clustering, regression analysis, document processing, and API integrations.

## Installation

```bash
pip install mitoolspro
```

**Requirements:** Python 3.12+

## Quick Example

```python
from mitoolspro.plotting import LinePlotter
from mitoolspro.clustering import kmeans_clustering

# Create a line plot
plotter = LinePlotter(x_data=[1, 2, 3, 4], y_data=[2, 4, 6, 8])
plotter.plot()

# Perform clustering
model, labels = kmeans_clustering(data, n_clusters=3)
```

## Modules

### Plotting (`mitoolspro.plotting`)

Create and compose various plot types with type validation and parameter management.

**Available plotters:**
- `BarPlotter`: Bar charts with grouping and stacking
- `BoxPlotter`: Box plots with outlier detection
- `ScatterPlotter`: Scatter plots with color mapping
- `LinePlotter`: Line plots for time series and continuous data
- `HistogramPlotter`: Histograms with density estimation
- `PiePlotter`: Pie charts with percentage labels
- `SankeyPlotter`: Flow diagrams for process visualization
- `DistributionPlotter`: Statistical distribution plots
- `ErrorPlotter`: Error bars and confidence intervals

**Plot composition:**
```python
from mitoolspro.plotting import PlotComposer, BarPlotter, LinePlotter

bar = BarPlotter(x_data=['A', 'B', 'C'], y_data=[1, 2, 3])
line = LinePlotter(x_data=[1, 2, 3], y_data=[1, 4, 2])

composer = PlotComposer()
composer.add_plot(bar, position=(0, 0))
composer.add_plot(line, position=(0, 1))
composer.compose(figsize=(12, 6))
```

### Clustering (`mitoolspro.clustering`)

Clustering algorithms with evaluation metrics and visualization tools.

**Functions:**
- `kmeans_clustering(data, n_clusters)`: K-means with model and labels output
- `agglomerative_clustering(data, n_clusters)`: Hierarchical clustering
- `clustering_ncluster_search(data, n_range)`: Find optimal cluster count using silhouette scores

**Evaluation tools:**
- Silhouette analysis
- Centroid calculations
- Distance metrics
- Cluster size analysis

```python
from mitoolspro.clustering import clustering_ncluster_search, plot_cluster_growth

# Find optimal number of clusters
best_n, results = clustering_ncluster_search(data, n_range=(2, 8))
plot_cluster_growth(results)
```

### Regression Analysis (`mitoolspro.regressions`)

Statistical and econometric models with diagnostic tools.

**Model classes:**
- `OLSModel`: Ordinary least squares regression
- `PanelOLSModel`: Panel data analysis with fixed/random effects
- `IVModel`: Instrumental variables regression
- `RegimeModel`: Regime switching models
- `SeasonalityModel`: Seasonal decomposition and modeling

```python
from mitoolspro.regressions import OLSModel

model = OLSModel(data, dependent='price', independent=['size', 'location'])
results = model.fit()
print(results.summary())
```

### Economic Complexity (`mitoolspro.economic_complexity`)

Tools for analyzing trade data and calculating economic complexity metrics.

**Main class: `EconomicComplexity`**
- Calculate Economic Complexity Index (ECI) and Product Complexity Index (PCI)
- Generate proximity matrices for products/countries
- Analyze trade relationships and export similarities

```python
from mitoolspro.economic_complexity import EconomicComplexity

ec = EconomicComplexity(trade_matrix)
eci_scores = ec.calculate_eci()
proximity_matrix = ec.calculate_proximity_matrix()
```

### Document Processing (`mitoolspro.document`)

Extract, analyze, and generate documents in PDF and DOCX formats.

**PDF processing:**
- `PDFProcessor`: Extract text and analyze document structure
- Layout analysis and section identification
- Font and formatting detection

**Document generation:**
- `DocumentWriter`: Create DOCX files programmatically
- Text styling and formatting
- Table and image insertion

```python
from mitoolspro.document import PDFProcessor, DocumentWriter

# Extract text from PDF
processor = PDFProcessor("report.pdf")
text = processor.extract_text()
structure = processor.analyze_structure()

# Generate new document
writer = DocumentWriter()
writer.add_heading("Analysis Results")
writer.add_paragraph(text)
writer.save("output.docx")
```

### Google API Integration (`mitoolspro.google_utils`)

**Places API (`mitoolspro.google_utils.places`):**
- `GooglePlacesClient`: Search places, get details, find nearby locations
- `PlacesWorkflow`: Batch operations and analysis workflows
- Geospatial analysis and saturation studies

```python
from mitoolspro.google_utils.places import GooglePlacesClient

client = GooglePlacesClient(api_key="your_api_key")
places = client.search_nearby(
    location=(37.7749, -122.4194), 
    radius=1000, 
    place_type="restaurant"
)
```

**YouTube API (`mitoolspro.google_utils.youtube`):**
- `YouTubeDownloader`: Download videos and extract metadata
- `VideoConverter`: Convert video formats and extract audio
- Batch processing capabilities

### LLM Integration (`mitoolspro.llms`)

Clients for language model APIs with usage tracking and cost management.

**Clients:**
- `OpenAIClient`: OpenAI API integration with token counting
- `OllamaClient`: Local LLM support via Ollama

**Usage tracking:**
- `PersistentTokensCounter`: Track token usage and costs across sessions
- Automatic cost calculation for different model types
- Usage history and statistics

```python
from mitoolspro.llms import OpenAIClient, PersistentTokensCounter

counter = PersistentTokensCounter("usage.json", model="gpt-4")
client = OpenAIClient(model="gpt-4", counter=counter)

response = client.request("Analyze this data trend...")
print(f"Total cost: ${counter.calculate_total_cost():.4f}")
```

### Network Analysis (`mitoolspro.networks`)

Create and visualize interactive network graphs.

- `NetworkGraph`: Build graphs from nodes and edges
- Interactive visualization with pyvis
- Community detection and layout algorithms
- Export to HTML for web display

### Database Utilities (`mitoolspro.databases`)

Simplified database operations for SQLAlchemy and SQLite.

- `SQLAlchemyHelper`: Connection management and query execution
- `SQLiteHelper`: Local database operations
- Schema management and data import/export

### File Processing (`mitoolspro.files`)

Handle various file formats with specialized processors.

- **Excel**: Read/write with `ExcelHandler`, column optimization
- **PDF**: Text extraction and metadata parsing
- **ICS**: Calendar file processing
- **General**: File conversion and format detection

### Natural Language Processing (`mitoolspro.nlp`)

Text processing tools using spaCy and transformers.

- `HuggingFaceEmbeddings`: Generate text embeddings
- spaCy pipeline components and utilities
- Keyword extraction and text analysis

### Web Scraping (`mitoolspro.scraping`)

Web scraping tools with Selenium integration.

- `WebScraper`: Basic scraping functionality
- `MultiScraper`: Parallel scraping for multiple targets
- Action-based scraping with `ScraperActions`

### Utilities (`mitoolspro.utils`)

General-purpose utilities and helper functions.

**Decorators:**
- `@parallel`: Multiprocessing decorator for batch operations
- `@timed`: Execution time measurement
- Context managers for resource handling

**Data structures:**
- String manipulation functions
- Iterable chunking and processing
- Development and debugging tools

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating each module:

**Plotting examples:**
- [`bar_plotter.ipynb`](examples/plotting/bar_plotter.ipynb)
- [`scatter_plotter.ipynb`](examples/plotting/scatter_plotter.ipynb) 
- [`line_plotter.ipynb`](examples/plotting/line_plotter.ipynb)
- [`composer.ipynb`](examples/plotting/composer.ipynb)

**Analysis examples:**
- [`clustering.ipynb`](examples/clustering.ipynb)
- [`networks.ipynb`](examples/networks.ipynb)
- [`google_places.ipynb`](examples/google_places.ipynb)

**Regression examples:**
- [`ols.ipynb`](examples/regressions/ols.ipynb)
- [`panelols.ipynb`](examples/regressions/panelols.ipynb)
- [`ivars.ipynb`](examples/regressions/ivars.ipynb)

## Command Line Interface

Basic project management through the CLI:

```bash
# Initialize a new project
mitoolspro init my_project --root ./projects --version v1
```

## Development

```bash
# Clone and install
git clone https://github.com/montanon/miToolsPro.git
cd miToolsPro
uv pip install -e .

# Run tests
pytest tests/ --cov=mitoolspro

# Generate coverage report
coverage html
```

**Code quality:**
- 77% test coverage with 84 test files
- Type annotations throughout
- 83 custom exception classes
- Comprehensive error handling

## Technical Details

**Architecture:**
- Modular design with lazy loading
- Abstract base classes for extensibility
- Mixin pattern for shared functionality
- Parallel processing support

**Dependencies:**
- Core: pandas, numpy, matplotlib, scikit-learn
- Visualization: plotly, seaborn
- Document processing: pymupdf, python-docx
- Web: requests, selenium
- ML: torch, transformers, spacy

## License

MIT License. See [LICENSE](LICENSE) file.

Copyright (c) 2025 Sebastián Montagna

## Support

- **Issues**: [GitHub Issues](https://github.com/montanon/miToolsPro/issues)
- **Contact**: sebastian@montagnainc.com