import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.preprocessing import StandardScaler
from variables import QUALITY_DIMENSIONS
import utils


def calculate_clustering_on_raw_quality_dimensions(selected_columns, n_clusters=7):
    """
    Performs PCA and KMeans clustering on raw quality dimension scores
    from multiple CSV files and generates both PCA-only and PCA+KMeans plots.
    """
    csv_files = glob.glob('../data/quality_analysis_results/*.csv')
    data_frames = []

    available_ids = utils.get_always_sparql_up(['2024-01-07','2024-04-07','2024-08-04','2024-12-01','2025-04-06'])
    for file in csv_files:
        df = pd.read_csv(file)

        valid_cols = [col for col in selected_columns if col in df.columns]
        if len(valid_cols) < len(selected_columns):
            continue
        
        df = df[df['KG id'].isin(available_ids)]
        df_selected = df[valid_cols].dropna()

        # Ensure numeric before transpose
        df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

        # Drop rows with non-numeric values
        if df_selected.isnull().values.any():
            df_selected = df_selected.dropna()

        # Transpose (rows = quality dimensions)
        df_selected = df_selected.T
        df_selected.index = valid_cols

        data_frames.append(df_selected)
    
    if not data_frames:
        print("No valid data found.")
        return

    # Concatenate all data
    full_data = pd.concat(data_frames, axis=1).dropna()

    # Standardize
    scaler = StandardScaler()
    quality_scaled_T = scaler.fit_transform(full_data)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(quality_scaled_T)

    # Explained variance
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Labels = quality dimension names
    quality_dimension_labels = full_data.index.tolist()

    # PCA-only plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], s=80)
    for i, label in enumerate(quality_dimension_labels):
        offset_x = 0.1 if i % 2 == 0 else -0.1
        offset_y = 1.5 if i % 3 == 0 else -1.5
        plt.annotate(label, (X_pca[i, 0] + offset_x, X_pca[i, 1] + offset_y), fontsize=9)
    plt.title("PCA of Raw Quality Dimensions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../data/charts/pca_kmeans/raw_quality_data_pca_only.png")
    plt.show()

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X_pca)

    # PCA + KMeans plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', s=80)
    
    # Optional: Plot centroids
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', s=80)
    for i, label in enumerate(quality_dimension_labels):
        offset_x = 0.1 if i % 2 == 0 else -0.1
        offset_y = 1.5 if i % 3 == 0 else -1.5
        plt.annotate(label, (X_pca[i, 0] + offset_x, X_pca[i, 1] + offset_y), fontsize=9)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='black', s=150, alpha=0.6, marker='X', label='Centroids')

    plt.title("PCA + KMeans Clustering of Raw Quality Dimensions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../data/charts/pca_kmeans/raw_quality_data_pca_kmeans.png")

def calculate_clustering_on_pairwise_correlation(n_clusters=7):
    """
    This function performs PCA and KMeans clustering on the averaged pairwise
    correlation matrix of quality dimensions across multiple CSV files.
    It generates both PCA-only and PCA+KMeans plots.
    """
    csv_files = glob.glob('../data/inferential_statistics_results/*_Always_UP.csv')  
    corr_matrices = []

    for file in csv_files:
        df = pd.read_csv(file, index_col=0)

        # Keep only correlation columns (exclude p-value columns)
        corr_df = df.loc[:, ~df.columns.str.contains('p-value')]
        # Ensure symmetry
        corr_df = corr_df.loc[corr_df.columns]  
        corr_matrices.append(corr_df.astype(float))  

    if not corr_matrices:
        print("No valid correlation matrices found.")
        return

    # Average the matrices
    avg_corr_matrix = sum(corr_matrices) / len(corr_matrices)

    # Ignore self-correlation
    np.fill_diagonal(avg_corr_matrix.values, 0)

    X = avg_corr_matrix.values

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # PCA-only plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], s=100)
    for i, name in enumerate(avg_corr_matrix.index):
        plt.text(X_pca[i, 0]+0.01, X_pca[i, 1]+0.01, name, fontsize=9)
    plt.title("PCA of Averaged Correlation Matrix (Quality Dimensions)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../data/charts/pca_kmeans/pairwise_correlation_pca_only.png')
    plt.show()

    # KMeans on PCA-reduced data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X_pca)
    centers = kmeans.cluster_centers_

    # PCA + KMeans plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', s=100)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, alpha=0.6, marker='X', label='Centroids')

    for i, name in enumerate(avg_corr_matrix.index):
        plt.text(X_pca[i, 0]+0.01, X_pca[i, 1]+0.01, name, fontsize=9)

    plt.title("PCA + KMeans Clustering of Quality Dimensions (Correlation Matrix)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../data/charts/pca_kmeans/pairwise_correlation_pca_kmeans.png')

calculate_clustering_on_raw_quality_dimensions(QUALITY_DIMENSIONS,n_clusters=6)