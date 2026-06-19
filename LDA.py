from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from basicConfig import X,y

# LDA considera as classes, então usa informação dos labels
# n_components máximo = min(n_features, n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=None)  # usa todos possíveis
X_lda = lda.fit_transform(X, y)