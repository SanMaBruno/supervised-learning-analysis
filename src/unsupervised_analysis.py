import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple

def apply_kmeans(X: pd.DataFrame, n_clusters: int = 2, n_components: int = 2) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Aplica K-Means clustering a los datos después de reducir su dimensionalidad usando PCA.

    Args:
        X (pd.DataFrame): DataFrame con los datos de entrada.
        n_clusters (int): Número de clusters para K-Means.
        n_components (int): Número de componentes para PCA.

    Returns:
        Tuple[pd.Series, pd.DataFrame]: Serie con los clusters asignados y DataFrame con las componentes principales.
    """
    X_numeric = X.select_dtypes(include=[float, int])
    
    pca = PCA(n_components=n_components)
    X_pca = pd.DataFrame(pca.fit_transform(X_numeric), columns=[f'PCA{i+1}' for i in range(n_components)], index=X.index)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = pd.Series(kmeans.fit_predict(X_numeric), name='Cluster', index=X.index)

    return clusters, X_pca

def plot_clusters(X_pca: pd.DataFrame, clusters: pd.Series, title: str = 'Clusters identificados por K-Means') -> None:
    """
    Grafica los clusters en un gráfico de dispersión utilizando las componentes principales de PCA.

    Args:
        X_pca (pd.DataFrame): DataFrame con las componentes principales de PCA.
        clusters (pd.Series): Serie con los clusters asignados.
        title (str): Título del gráfico.

    Returns:
        None
    """
    X_clustered = X_pca.copy()
    X_clustered['Cluster'] = clusters

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X_clustered, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, alpha=0.7, edgecolor='k')
    plt.title(title, fontsize=15)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Cluster', loc='upper right')
    plt.show()