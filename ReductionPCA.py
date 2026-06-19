from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def ReducePCA(X):
    # IMPORTANTE: sempre padronizar antes do PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Escolher número de componentes que explicam X% da variância
    pca = PCA(n_components=0.95)  # mantém 95% da variância
    X_pca = pca.fit_transform(X_scaled)

    return X_pca