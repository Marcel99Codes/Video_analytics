import os
import argparse

import task1_dense_trajectories as task1
import task2_descriptors as task2
import task3_1_pca as task3_1
import task3_2_fisher_vector_encoding as task3_2
import task4_svm as task4


MAX_DATA_PER_CLASS = 5 #Limit the number of videos per class for faster training

VIDEO_PATH = "data"
OUTPUT_PATH = "output"

def get_video_files(video_path):
    """Get all video files in the given path"""
    return [f for f in os.listdir(video_path) if f.endswith(".avi")]#[:MAX_DATA_PER_CLASS]

def feature_extraction_pipeline(video_path, output_ptah):
    """Pipeline: 1. Dense Trajectories -> 2. Descriptors -> 3. PCA -> 4. Fisher Vector Encoding"""
    # Class1 - TrampolineJumping
    for f in get_video_files(VIDEO_PATH + "/TrampolineJumping"):

        video_file = os.path.join(VIDEO_PATH, "TrampolineJumping", f)
        trajectory_file = os.path.join(OUTPUT_PATH, "TrampolineJumping", f + "_trajectories.npy")
        output_file = os.path.join(OUTPUT_PATH, "TrampolineJumping", f + "_426d_descriptors.npy")

        task1.main(video_file, trajectory_file)
        task2.main(video_file, trajectory_file, output_file)
    
    task3_1.main(OUTPUT_PATH + "/TrampolineJumping") 
    task3_2.main(OUTPUT_PATH + "/TrampolineJumping")

    # Class2 - UnevenBars
    for f in get_video_files(VIDEO_PATH + "/UnevenBars"):

        video_file = os.path.join(VIDEO_PATH, "UnevenBars", f)
        trajectory_file = os.path.join(OUTPUT_PATH, "UnevenBars", f + "_trajectories.npy")
        output_file = os.path.join(OUTPUT_PATH, "UnevenBars", f + "_426d_descriptors.npy")

        task1.main(video_file, trajectory_file)
        task2.main(video_file, trajectory_file, output_file)
    
    task3_1.main(OUTPUT_PATH + "/UnevenBars")
    task3_2.main(OUTPUT_PATH + "/UnevenBars")

    # Class3 - UnevenBars
    for f in get_video_files(VIDEO_PATH + "/VolleyballSpiking"):

        video_file = os.path.join(VIDEO_PATH, "VolleyballSpiking", f)
        trajectory_file = os.path.join(OUTPUT_PATH, "VolleyballSpiking", f + "_trajectories.npy")
        output_file = os.path.join(OUTPUT_PATH, "VolleyballSpiking", f + "_426d_descriptors.npy")

        task1.main(video_file, trajectory_file)
        task2.main(video_file, trajectory_file, output_file)
    
    task3_1.main(OUTPUT_PATH + "/VolleyballSpiking")
    task3_2.main(OUTPUT_PATH + "/VolleyballSpiking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=int, choices=range(5), help="0=pipeline, 1=trainSvm, 2=testDummy")

    args = parser.parse_args()
    state = args.state
        
    #Feature extraction pipeline
    if state == 0:
        feature_extraction_pipeline(VIDEO_PATH, OUTPUT_PATH)
    #Train the SVM
    elif state == 1:
        task4.main(OUTPUT_PATH, VIDEO_PATH)

    #Test on dummy data
    if state == 2: 
        video_file = VIDEO_PATH + "/dummy.avi"
        trajectory_file = OUTPUT_PATH + "/dummy_trajectories.npy"
        output_file = OUTPUT_PATH + "/dummy_426d_descriptors.npy"

        task1.main(video_file, output_file)
        