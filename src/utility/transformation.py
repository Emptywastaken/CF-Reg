import torch

def pca( X, n_components: int =2):
        
        # 1. Standardize the data
        X_mean = torch.mean(X, 0)
        X_std = torch.std(X, 0)
        X = (X - X_mean) / X_std

        # 2. Calculate the covariance matrix
        X_t = X.t()
        covariance_matrix = X_t @ X / (X.size(0) - 1)

        # 3. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # 4. Sort eigenvalues and eigenvectors
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Select the top n_components eigenvectors (n_components <= number of features)
        components = eigenvectors[:, :n_components]

        # 6. Transform the original matrix
        X_pca = X @ components

        return X_pca
