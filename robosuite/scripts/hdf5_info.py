import h5py

def read_actions_states(file_path):
    with h5py.File(file_path, 'r') as file:
        data_group = file['data']

        for demo_name, demo_group in data_group.items():
            print(f"Reading demonstration: {demo_name}")

           

            # Access and print the 'states' dataset
            if 'states' in demo_group:
                states = demo_group['states'][:]
                print(f"    States: {states.shape}")
                import pdb; pdb.set_trace()
                # Uncomment the next line to print the states data
                # print(states)

            # Access and print the 'actions' dataset
            if 'actions' in demo_group:
                actions = demo_group['actions'][:]
                print(f"    Actions: {actions.shape}")
                # Uncomment the next line to print the actions data
                # print(actions)



# read_actions_states('/home/dpenmets/LIRA_work/robosuite_vd/robosuite/models/assets/demonstrations/1705188586_7761292/low_dim.hdf5')

#--------------------------------------


import h5py
from sklearn.cluster import KMeans
import numpy as np

def read_and_cluster(file_path):
    data_to_cluster = []
    demo_names = []

    with h5py.File(file_path, 'r') as file:
        data_group = file['data']

        # Extracting the relevant part of the 'states' for each demonstration
        for demo_name, demo_group in data_group.items():
            if 'states' in demo_group:
                states = demo_group['states'][:]
                last_state_slice = states[-1][23:26]  # Extract indices 23 to 26 of the last state
                data_to_cluster.append(last_state_slice)
                demo_names.append(demo_name)

    # Convert list to numpy array for clustering
    data_to_cluster = np.array(data_to_cluster)

    # Applying K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data_to_cluster)
    labels = kmeans.labels_

    # Separating demonstrations into two clusters
    cluster_1 = [demo_names[i] for i in range(len(labels)) if labels[i] == 0]
    cluster_2 = [demo_names[i] for i in range(len(labels)) if labels[i] == 1]

    return cluster_1, cluster_2

# Example usage
file_path = '/home/dpenmets/LIRA_work/robosuite_vd/robosuite/models/assets/demonstrations/1705188586_7761292/low_dim.hdf5'
cluster_1, cluster_2 = read_and_cluster(file_path)
print("Cluster 1 Demonstrations:", len(cluster_1))
print("Cluster 2 Demonstrations:", len(cluster_2))
