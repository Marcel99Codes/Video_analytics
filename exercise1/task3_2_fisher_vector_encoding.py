import numpy as np
from sklearn.mixture import GaussianMixture
import os

def load_pca_descriptors(folder):
    descriptors = []
    video_names = []
    for file in os.listdir(folder):
        if file.endswith("_426d_descriptors.npy"):
            path = os.path.join(folder, file)
            desc = np.load(path)
            descriptors.append(desc)
            video_names.append(file.replace("_426d_descriptors.npy", ""))
    return descriptors, video_names

def compute_fisher_vector(x, gmm):
    N, D = x.shape
    K = gmm.n_components

    probs = gmm.predict_proba(x)  # N x K
    means = gmm.means_            # K x D
    covariances = gmm.covariances_  # K x D (diag cov)
    weights = gmm.weights_        # K

    fisher_vectors = []

    for i in range(K):
        q_i = probs[:, i][:, np.newaxis]  # N x 1
        x_centered = x - means[i]         # N x D
        sigma_inv = 1. / np.sqrt(covariances[i])

        grad_mu = np.sum(q_i * x_centered * sigma_inv, axis=0)
        grad_sigma = np.sum(q_i * ((x_centered ** 2 - covariances[i]) * sigma_inv ** 2), axis=0)
        grad_weight = (np.sum(q_i) - N * weights[i]) / np.sqrt(weights[i])

        fisher = np.concatenate([grad_mu, grad_sigma, [grad_weight]])
        fisher_vectors.append(fisher)

    return np.concatenate(fisher_vectors)

def main(output_path):
    descriptors, video_names = load_pca_descriptors(output_path)

    all_descriptors = np.vstack(descriptors)
    print(f"Fitting GMM on: {all_descriptors.shape}")

    gmm = GaussianMixture(n_components=min(8, all_descriptors.shape[0]), covariance_type='diag', random_state=0, max_iter=200)
    gmm.fit(all_descriptors)

    np.save(output_path + "/gmm_weights.npy", gmm.weights_)
    np.save(output_path + "/gmm_means.npy", gmm.means_)
    np.save(output_path + "/gmm_covariances.npy", gmm.covariances_)

    for x, name in zip(descriptors, video_names):
        fv = compute_fisher_vector(x, gmm)
        np.save(output_path + f"/{name}_fisher_vector.npy", fv)
