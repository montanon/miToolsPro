# miToolsPro

<img src="assets/mitoolspro-banner.png" width="1280" alt="Banner Image">

---

miToolsPro is a powerful Python package that provides a comprehensive suite of tools for data analysis, visualization, and project management. It is designed to streamline workflows and enhance productivity in data science and research projects with integrated functionalities that allow for complex analyses of varying data types.

## Features

### Data Visualization
- Advanced plotting capabilities with customizable parameters
- Support for various plot types:
  - Bar plots
  - Box plots
  - Distribution plots
  - Error plots
  - Histograms
  - Line plots
  - Pie charts
  - Scatter plots
- Flexible plot composition, customization, and integrated store/load functionality
- High-quality output with configurable DPI and formats

### Clustering Analysis
- Implementation of popular clustering algorithms:
  - K-means clustering
  - Agglomerative clustering
- Comprehensive clustering evaluation tools:
  - Centroid analysis
  - Distance calculations
  - Similarity metrics
  - Cluster size analysis
- Visualization tools for cluster analysis:
  - Cluster growth plots
  - Silhouette score analysis
  - Distribution plots
  - Cluster groupings visualization

### Economic Complexity Analysis
- Tools for calculating economic complexity indices
- Export matrix analysis
- Proximity matrix calculations
- Relatedness matrix analysis
- Distribution visualization for economic indicators

### Project Management
- Structured project organization
- Version control for project files
- Automated backup system
- Project metadata management
- File organization and tracking

### Google Places Integration
- Comprehensive Google Places API integration
- Place search and analysis
- Geospatial visualization
- Customizable place type filtering
- Detailed place information retrieval

### Document Processing
- PDF document analysis
- Text extraction and processing
- Document structure analysis
- Layout analysis capabilities

## Requirements
- Python >= 3.12
- Comprehensive set of dependencies including:
  - Data processing: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Machine learning: scikit-learn, torch
  - Geospatial: geopandas, folium
  - Document processing: pdfminer, pymupdf
  - And many more (see pyproject.toml for complete list)

## Installation
```bash
uv pip install mitoolspro
```

## Quick Start
Check out our comprehensive examples in the `examples/` directory:

- [Clustering Analysis](examples/clustering.ipynb): Examples of clustering analysis and visualization
- [Google Places Integration](examples/google_places.ipynb): Google Places API integration examples
- [Network Analysis](examples/networks.ipynb): Network analysis and visualization examples
- [Plotting Examples](examples/plotting/): Various plotting examples and use cases
- [Regression Analysis](examples/regressions/): Regression analysis examples

Each example notebook provides detailed code snippets and explanations to help you get started with miToolsPro.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Sebastián Montagna
