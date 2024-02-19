# %%
import numpy as np
from typing import Any
from sklearn.datasets import make_blobs

# TODO: implement the PCA with numpy
# Note that you are not allowed to use any existing PCA implementation from sklearn or other libraries.
class PrincipalComponentAnalysis:
    def __init__(self, n_components: int) -> None:
        """_summary_

        Parameters
        ----------
        n_components : int
            The number of principal components to be computed. This value should be less than or equal to the number of features in the dataset.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    # TODO: implement the fit method
    def fit(self, X: np.ndarray):
    
      self.mean = np.mean(X, axis=0)
        
        # Let´s center our data points
      centered_X = X - self.mean
        
        # Let´s compute the covariance matrix
      covariance_matrix = np.cov(centered_X, rowvar=False)
        
        # Let´s Perform eigen decomposition
      eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Let´s sort eigenvectors by decreasing eigenvalues and select top n_components
      idx = np.argsort(eigenvalues)[::-1]
      self.components = eigenvectors[:, idx[:self.n_components]]
        
      return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        # Let´s center the new data using the mean from the training set
        centered_X = X - self.mean
        
        # Let´s project the centered data onto the selected eigenvectors
        X_new = np.dot(centered_X, self.components)
        
        return X_new


# TODO: implement the LDA with numpy
# Note that you are not allowed to use any existing LDA implementation from sklearn or other libraries.
class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Hint:
        -----
        To implement LDA with numpy, follow these steps:
        1. Compute the mean vectors for each class.
        2. Compute the within-class scatter matrix.
        3. Compute the between-class scatter matrix.
        4. Compute the eigenvectors and corresponding eigenvalues for the scatter matrices.
        5. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix W.
        6. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
        """
        # Here we compute the mean vectors for each class
        self.mean = np.mean(X, axis=0)
        unique_classes = np.unique(y)
        n_features = X.shape[1]

        mean_vectors = []
        for cls in unique_classes:
            mean_vectors.append(np.mean(X[y == cls], axis=0))

        mean_vectors = np.array(mean_vectors)

        # Computing the within-class scatter matrix
        within_class_scatter_matrix = np.zeros((n_features, n_features))
        for cls, mean_vec in zip(unique_classes, mean_vectors):
            class_scatter_matrix = np.zeros((n_features, n_features))
            for row in X[y == cls]:
                row, mean_vec = row.reshape(n_features, 1), mean_vec.reshape(n_features, 1)
                class_scatter_matrix += (row - mean_vec).dot((row - mean_vec).T)
            within_class_scatter_matrix += class_scatter_matrix

        # Let´s compute the between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        between_class_scatter_matrix = np.zeros((n_features, n_features))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == unique_classes[i]].shape[0]
            mean_vec = mean_vec.reshape(n_features, 1)
            overall_mean = overall_mean.reshape(n_features, 1)
            between_class_scatter_matrix += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        # next we Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))

        # Then sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # and choose k eigenvectors with the largest eigenvalues
        self.components = eigenvectors[:, :self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        X_new = np.dot(X - self.mean, self.components)

        return X_new


# TODO: Generating adversarial examples for PCA.
# We will generate adversarial examples for PCA. The adversarial examples are generated by creating two well-separated clusters in a 2D space. Then, we will apply PCA to the data and check if the clusters are still well-separated in the transformed space.
# Your task is to generate adversarial examples for PCA, in which
# the clusters are well-separated in the original space, but not in the PCA space. The separabilit of the clusters will be measured by the K-means clustering algorithm in the test script.
#
# Hint:
# - You can place the two clusters wherever you want in a 2D space.
# - For example, you can use `np.random.multivariate_normal` to generate the samples in a cluster. Repeat this process for both clusters and concatenate the samples to create a single dataset.
# - You can set any covariance matrix, mean, and number of samples for the clusters.
class AdversarialExamples:
    def __init__(self) -> None:
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        """Generate adversarial examples for PCA

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_features : int
            The number of features.

        Returns
        -------
        X: ndarray of shape (n_samples, n_features)
            Transformed values.

        y: ndarray of shape (n_samples,)
            Cluster IDs. y[i] is the cluster ID of the i-th sample.

        """
       
        # Let´s generate two well-separated clusters in 2D space
        mean1 = np.array([5, 5]) 
        cov1 = np.identity(n_features)
        cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples//2)
        
        mean2 = np.array([-5, -5])
        cov2 = np.identity(n_features) 
        cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples//2)
        
        X = np.concatenate([cluster1, cluster2], axis=0)
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])

        return X, y
