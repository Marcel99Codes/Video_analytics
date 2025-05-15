import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data_from_file(file_path, feature_dir):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            video, label = line.strip().split()
            feature_path = os.path.join(feature_dir, f"{video}_fisher_vector.npy")
            if os.path.exists(feature_path):
                feature = np.load(feature_path)
                X.append(feature)
                y.append(int(label))
            else:
                print(f"Missing feature for {video}")

    return np.array(X), np.array(y)

def main(output_path, data_path):
    train_file = data_path + "/train.txt"
    test_file = data_path + "/test.txt"

    print("Loading training data")
    X_train, y_train = load_data_from_file(train_file, output_path)

    print("Loading test data")
    X_test, y_test = load_data_from_file(test_file, output_path)

    print(f"Training SVM on {len(X_train)} samples")
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)

    print("Evaluating on test set")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

