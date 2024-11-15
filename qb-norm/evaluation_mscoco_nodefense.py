import subprocess
import numpy as np
import os
import pickle as pkl
import re
import ast
import json
import math
from tqdm.notebook import tqdm

def parse_metrics(command):
    # Run the command using subprocess in Jupyter
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:\n", e.stderr)

    # Split the output into lines
    output_lines = result.stdout.strip().split('\n')
    # print(output_lines)

    metrics_before, metrics_after = None, None
    # Iterate over the lines to find metrics
    for i, line in enumerate(output_lines):
        if 'Metrics before applying QB-Norm' in line:
            # The next line contains the metrics dictionary
            if i + 1 < len(output_lines):
                metrics_line = output_lines[i + 1].strip()
                try:
                    metrics_before = ast.literal_eval(metrics_line)
                except (SyntaxError, ValueError):
                    print(f"Could not parse metrics before at line {i + 1}: {metrics_line}")
        elif 'Metrics after QB-Norm' in line:
            # The next line contains the metrics dictionary
            if i + 1 < len(output_lines):
                metrics_line = output_lines[i + 1].strip()
                try:
                    metrics_after = ast.literal_eval(metrics_line)
                except (SyntaxError, ValueError):
                    print(f"Could not parse metrics after at line {i + 1}: {metrics_line}")

    # Save the metrics into a variable
    metrics = {
        'before': metrics_before,
        'after': metrics_after
    }
    return metrics

def execute_dynamic_inverted_softmax(test_file_path, adv_test_similarity_path, gt_idx_path, out_dir):
    temp_out_dir=out_dir+"/output_cache"
    # Load data from file
    data = np.load(test_file_path, allow_pickle=True)  # Load the data as a NumPy array
    # Load adversarial similarity matrices
    adv_test_similarity_matrices = np.load(adv_test_similarity_path, allow_pickle=True)
    gt_idx=np.load(gt_idx_path, allow_pickle=True)
    # Create temp directory if it doesn't exist
    os.makedirs(temp_out_dir, exist_ok=True)
    
    gt_metrics = {}
    adv_metrics = {}
    # Iterate over adversarial matrices, modify data, save with different names, and execute command
    for i, adv in enumerate(tqdm(adv_test_similarity_matrices[:100])):
        # print("Adv shape:", adv.shape)
        test_data_temp = data.copy()
        # print("Test data shape:", test_data_temp.shape)
        # print("Adv shape:", adv.shape)
        test_data_temp = np.column_stack((test_data_temp, adv.flatten()))        
        # Save the modified data to a new file with different names in temp directory
        modified_file_path = f'{temp_out_dir}/modified-test-images-texts-seed0-{i}.pkl'
        with open(modified_file_path, 'wb') as f:
            pkl.dump(test_data_temp, f)
        # print(f"Modified data saved to {modified_file_path}")


        num_queries, num_vids = test_data_temp.shape
        adv_idx = np.array([
            num_vids * (i+1) - 1 for i in range(num_queries)
        ])

        adv_idx_path = f'{temp_out_dir}/adv_idx.pkl'
        with open(adv_idx_path, 'wb') as f:
            pkl.dump(adv_idx, f)
        # print(f"adv_idx saved to {adv_idx_path}")

        modified_gt_index = gt_idx.copy()
        temp_shape = modified_gt_index.shape
        modified_gt_index = modified_gt_index.flatten() + np.arange(modified_gt_index.size)
        modified_gt_index = modified_gt_index.reshape(temp_shape)

        modified_gt_index_path=f'{temp_out_dir}/modified_gt_idx-{i}.pkl'
        with open(modified_gt_index_path, 'wb') as f:
            pkl.dump(modified_gt_index, f)
        # print(f"modified_gt_idx saved to {modified_gt_index_path}")

        # Define the command and arguments for each modified data file
        gt_command = [
            'python', 'dynamic_inverted_softmax.py',
            '--sims_test_path', modified_file_path,
            '--gt_idx_path', modified_gt_index_path
        ]

        adv_command = [
            'python', 'dynamic_inverted_softmax.py',
            '--sims_test_path', modified_file_path,
            '--gt_idx_path', adv_idx_path
        ]
        
        # Run the command
        gt_metrics[i]=parse_metrics(gt_command)
        # print(gt_metrics[i])

        adv_metrics[i]=parse_metrics(adv_command)
        # print(adv_metrics[i])
        
        # save the metrics to a file
        with open(f'{out_dir}/transfer_openclip_gt_metrics.pkl', 'wb') as f:
            pkl.dump(gt_metrics, f)
        with open(f'{out_dir}/transfer_openclip_adv_metrics.pkl', 'wb') as f:
            pkl.dump(adv_metrics, f)

        # Remove the modified data files
        # os.remove(modified_file_path)
        
    return gt_metrics, adv_metrics

out_dir = '../outputs/mscoco/imagebind/naive'
test_file_path = f'{out_dir}/test_similarity_matrix.pkl'
adv_test_similarity_path = f'{out_dir}/transfer_openclip_adv_test_similarity_matrix.pkl'
gt_idx_path='../outputs/mscoco/gt_idx.pkl'

os.makedirs(out_dir, exist_ok=True)

# print the shape of the similarity matrices
test_data = np.load(test_file_path, allow_pickle=True)
adv_test_similarity_matrices = np.load(adv_test_similarity_path, allow_pickle=True)
gt_idx=np.load(gt_idx_path, allow_pickle=True)
print("Test data shape:", test_data.shape)
print("Adv test similarity matrices shape:", adv_test_similarity_matrices.shape)
print("gt_idx shape:", gt_idx.shape)

# Define the command and arguments
command = [
    'python', 'dynamic_inverted_softmax.py',
    '--sims_test_path', test_file_path,
    '--gt_idx_path', gt_idx_path
]

# Parse the metrics
metrics = parse_metrics(command)
print("Mertrics for original data:")
print(metrics)

gt_metrics, adv_metrics = execute_dynamic_inverted_softmax(test_file_path, adv_test_similarity_path, gt_idx_path, out_dir)