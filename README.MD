# World Happiness Analysis

This project provides an in-depth analysis of the World Happiness Report dataset, visualizing key insights and applying clustering techniques to group countries based on their happiness metrics. The project is implemented in Python and uses libraries such as Pandas, Matplotlib, Seaborn, and Plotly for data manipulation and visualization.

## Project Structure


world_happiness_analysis/
│
├── data/
│   └── happiness_report.csv         # Input dataset
│
├── plots/                           # Generated visualizations
│   ├── bubbleplot_happiness.png
│   ├── correlation_matrix.png
│   ├── Freedom to make life choices_cluster_histograms.png
│   ├── GDP per capita_cluster_histograms.png
│   ├── Generosity_cluster_histograms.png
│   ├── happiness_score_vs_gdp_per_capita.png
│   ├── Healthy life expectancy_cluster_histograms.png
│   ├── histograms.png
│   ├── optimal_number_of_clusters.png
│   ├── pairplot_with_median_and_correlation.png
│   ├── Perceptions of corruption_cluster_histograms.png
│   └── Social support_cluster_histograms.png
│
├── LICENSE                          # License for the project
├── main.py                          # Python code for data analysis
└── README.md                        # Project documentation


## Features

- **Exploratory Data Analysis (EDA):**
  - Pair plots with median lines and correlation annotations.
  - Histograms for all key metrics in the dataset.
  - Correlation matrix heatmap for visualizing relationships between variables.

- **Clustering Analysis:**
  - Elbow method to determine the optimal number of clusters.
  - KMeans clustering to group countries into three clusters.
  - Cluster-wise histograms for key metrics.

- **Scatter Plots:**
  - Happiness Score vs GDP Per Capita.
  - Bubble plot visualizing the relationship between GDP, corruption, and life expectancy.

## Dependencies

Install the required Python libraries before running the project:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

## Usage

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd world_happiness_analysis
   ```

2. Place the `happiness_report.csv` file in the `data/` directory.

3. Run the main script:

   ```bash
   python main.py


4. Check the generated visualizations in the `plots/` directory.

## Dataset

The dataset used in this analysis is the [World Happiness Report](https://worldhappiness.report/), which contains metrics for happiness scores, GDP, social support, life expectancy, and more.

## Results

The analysis provides:
- Insightful visualizations of happiness metrics.
- Clustered groupings of countries based on happiness-related features.
- Correlations and trends between variables such as GDP and happiness scores.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

