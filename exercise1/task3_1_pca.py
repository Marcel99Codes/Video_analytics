import numpy as np
from sklearn.decomposition import PCA
import os

def load_all_descriptors(folder):
    descriptors = []
    video_names = []
    for file in os.listdir(folder):
        if file.endswith("_426d_descriptors.npy"):
            path = os.path.join(folder, file)
            desc = np.load(path)
            descriptors.append(desc)
            video_names.append(file.replace("_426d_descriptors.npy", ""))
    return descriptors, video_names

def compute_video_representations(descriptors, pca):
    video_features = []
    for traj_desc in descriptors:
        reduced = pca.transform(traj_desc)
        mean_vec = np.mean(reduced, axis=0)
        video_features.append(mean_vec)
    return np.array(video_features)

def main(output_path):
    all_descriptors, video_names = load_all_descriptors(output_path)

    all_data = np.vstack(all_descriptors)
    print(f"Fitting PCA on shape: {all_data.shape}")

    pca = PCA(n_components=min(64, all_data.shape[0]))

    pca.fit(all_data)
    print("Explained variance ratio (first 64 components):", np.sum(pca.explained_variance_ratio_))

    np.save(output_path + "/pca_components.npy", pca.components_)
    np.save(output_path + "/pca_mean.npy", pca.mean_)

    for desc, name in zip(all_descriptors, video_names):
        reduced = pca.transform(desc)
        mean_feat = np.mean(reduced, axis=0)
        np.save(output_path + f"/{name}_64d_mean.npy", mean_feat)

