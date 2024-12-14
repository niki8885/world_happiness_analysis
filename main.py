import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Set pandas display options
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

# Load data
data_frame = pd.read_csv('data/happiness_report.csv')

# Basic data overview
print(data_frame.head())
print(data_frame.describe())
print(data_frame.shape)
print(data_frame[data_frame['Country or region'] == "Hungary"])
print(data_frame.info())
print(data_frame.isnull().sum())
print(data_frame.duplicated().sum())

# Define utility functions
def add_median_line(x, y, **kwargs):
    plt.scatter(x, y, alpha=0.6)
    plt.axhline(np.median(y), color='red', linestyle='--', alpha=0.8, label="Median Y")
    plt.axvline(np.median(x), color='blue', linestyle='--', alpha=0.8, label="Median X")
    plt.legend()

def annotate_correlation(x, y, **kwargs):
    corr = np.corrcoef(x, y)[0, 1]
    plt.gca().annotate(f'Corr: {corr:.2f}', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=10)

# Columns for pairplot
columns_to_plot = [
    'Score', 'GDP per capita', 'Social support',
    'Healthy life expectancy', 'Freedom to make life choices',
    'Generosity', 'Perceptions of corruption'
]

# Pairplot with additional annotations
g = sns.PairGrid(data_frame[columns_to_plot])
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_diag(sns.histplot, kde=True)
g.map_lower(add_median_line)
g.map_lower(annotate_correlation)
g.fig.suptitle("Pairplot with Median Lines and Correlation", y=1.02, fontsize=20)

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
pairplot_path = os.path.join(output_dir, "pairplot_with_median_and_correlation.png")
g.savefig(pairplot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Pairplot saved to: {pairplot_path}")

# Histograms for selected columns
plt.figure(figsize=(20, 50))
for i, column in enumerate(columns_to_plot):
    plt.subplot(len(columns_to_plot), 1, i + 1)
    sns.histplot(data_frame[column], kde=True, color='r')
    plt.title(column)
plt.tight_layout()

histogram_path = os.path.join(output_dir, "histograms.png")
plt.savefig(histogram_path, dpi=300)
plt.show()
print(f"Histograms saved to: {histogram_path}")

# Correlation matrix heatmap
correlation_matrix = data_frame[columns_to_plot].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Happiness Data", fontsize=16)

correlation_matrix_path = os.path.join(output_dir, "correlation_matrix.png")
plt.savefig(correlation_matrix_path, dpi=300)
plt.show()
print(f"Correlation matrix heatmap saved to: {correlation_matrix_path}")

# Scatter Plot: Happiness Score vs GDP Per Capita
fig = px.scatter(
    data_frame,
    x='GDP per capita', y='Score', size='Overall rank', hover_name="Country or region",
    trendline="ols", text='Country or region',
    labels={'GDP per capita': 'GDP Per Capita', 'Score': 'Happiness Score'}
)
fig.update_layout(
    title='Happiness Score vs GDP Per Capita', title_x=0.5,
    xaxis_title='GDP Per Capita', yaxis_title='Happiness Score',
    template='plotly', font=dict(size=10)
)
scatter_path = os.path.join(output_dir, "happiness_score_vs_gdp_per_capita.png")
fig.write_image(scatter_path, scale=3)
fig.show()
print(f"Scatter plot saved to: {scatter_path}")

# KMeans clustering
features = data_frame.drop(columns=['Country or region', 'Score', 'Overall rank'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Optimal number of clusters (Elbow Method)
scores = []
range_values = range(1, 20)
for i in range_values:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    scores.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range_values, scores, 'bx-')
plt.title('Finding the Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
optimal_clusters_path = os.path.join(output_dir, "optimal_number_of_clusters.png")
plt.savefig(optimal_clusters_path, dpi=300)
plt.show()
print(f"Elbow plot saved to: {optimal_clusters_path}")

# Perform KMeans with optimal clusters
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled_data)

data_frame['cluster'] = labels

# Save cluster histograms
for column in features.columns:
    plt.figure(figsize=(20, 10))
    for cluster in range(3):
        plt.subplot(1, 3, cluster + 1)
        cluster_data = data_frame[data_frame['cluster'] == cluster]
        cluster_data[column].hist(bins=20)
        plt.title(f'{column} - Cluster {cluster}')
    cluster_hist_path = os.path.join(output_dir, f"{column}_cluster_histograms.png")
    plt.savefig(cluster_hist_path, dpi=300)
    plt.close()
    print(f"Histogram for {column} saved to: {cluster_hist_path}")

# Bubble Plot: Impact of Economy, Corruption, and Life Expectancy
fig = px.scatter(
    data_frame,
    x='GDP per capita', y='Perceptions of corruption',
    size='Score', color='cluster',
    hover_name='Country or region',
    labels={
        'GDP per capita': 'GDP Per Capita',
        'Perceptions of corruption': 'Perceptions of Corruption',
        'Score': 'Happiness Score'
    },
    title='Impact of Economy, Corruption, and Life Expectancy on Happiness Scores',
    template='plotly'
)

bubble_plot_path = os.path.join(output_dir, "bubbleplot_happiness.png")
fig.write_image(bubble_plot_path, scale=3)
fig.show()
print(f"Bubble plot saved to: {bubble_plot_path}")
