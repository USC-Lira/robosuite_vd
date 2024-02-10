# import h5py
# import pdb

# def read_actions_states(file_path):
#     with h5py.File(file_path, 'r') as file:
#         data_group = file['data']
#         pdb.set_trace()
        

#         for demo_name, demo_group in data_group.items():

#             print(f"Reading demonstration: {demo_name}")
#             # pdb.set_trace()
           

#             # Access and print the 'states' dataset
#             if 'states' in demo_group:
#                 states = demo_group['states'][:]
#                 print(f"    States: {states.shape}")
#                 # pdb.set_trace()
#                 # Uncomment the next line to print the states data
#                 # print(states)

#             # Access and print the 'actions' dataset
#             if 'actions' in demo_group:
#                 actions = demo_group['actions'][:]
#                 print(f"    Actions: {actions.shape}")
#                 # Uncomment the next line to print the actions data
#                 # print(actions)



# # read_actions_states('/home/dpenmets/LIRA_work/robosuite_vd/robosuite/models/assets/demonstrations/1705188586_7761292/low_dim.hdf5')
# # read_actions_states('/home/dpenmets/LIRA_work/robosuite_vd/robosuite/models/assets/demonstrations/1705269065_8573515/demo.hdf5')

# #--------------------------------------


# import h5py
# from sklearn.cluster import KMeans
# import numpy as np

# def read_and_cluster(file_path):
#     data_to_cluster = []
#     demo_names = []

#     with h5py.File(file_path, 'r') as file:
#         data_group = file['data']

#         # Extracting the relevant part of the 'states' for each demonstration
#         for demo_name, demo_group in data_group.items():
#             if 'states' in demo_group:
#                 states = demo_group['states'][:]
#                 last_state_slice = states[-1][23:26]  # Extract indices 23 to 26 of the last state
#                 data_to_cluster.append(last_state_slice)
#                 demo_names.append(demo_name)

#     # Convert list to numpy array for clustering
#     data_to_cluster = np.array(data_to_cluster)

#     # Applying K-Means Clustering
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(data_to_cluster)
#     labels = kmeans.labels_

#     # Separating demonstrations into two clusters
#     cluster_1 = [demo_names[i] for i in range(len(labels)) if labels[i] == 0]
#     cluster_2 = [demo_names[i] for i in range(len(labels)) if labels[i] == 1]

#     return cluster_1, cluster_2

# # # Example usage
# file_path = '/home/dpenmets/LIRA_work/robosuite_vd/robosuite/scripts/28_27_combined.hdf5'
# cluster_1, cluster_2 = read_and_cluster(file_path)
# print("Cluster 1 Demonstrations:", len(cluster_1), "\n")
# print("Cluster 2 Demonstrations:", len(cluster_2))

# # import pprint

# # message = f"Group 1 demos count: {len(cluster_1)}  Group 2 demos count: {len(cluster_2)}"
# # pprint.pprint(message)


import h5py
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import norm

def read_and_cluster(file_path):
    data_to_cluster = []
    demo_names = []

    with h5py.File(file_path, 'r') as file:
        data_group = file['data']

        for demo_name, demo_group in data_group.items():
            import pdb; pdb.set_trace()
            if 'states' in demo_group:
                states = demo_group['states'][:]
                last_state_slice = states[-1][23:26]
                # print(last_state_slice)
                # import pdb; pdb.set_trace()
                data_to_cluster.append(last_state_slice)
                demo_names.append(demo_name)

    data_to_cluster = np.array(data_to_cluster)

    # Applying K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data_to_cluster)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    cluster_1 = [demo_names[i] for i in range(len(labels)) if labels[i] == 0]
    cluster_2 = [demo_names[i] for i in range(len(labels)) if labels[i] == 1]

    # Cluster statistics
    print("Cluster Statistics:")

    for cluster_index, cluster in enumerate([cluster_1, cluster_2]):
        print(f"\nCluster {cluster_index + 1}:")
        print(f"Number of Demonstrations: {len(cluster)}")

        # Calculating distances to centroid
        cluster_points = data_to_cluster[labels == cluster_index]
        distances = norm(cluster_points - centroids[cluster_index], axis=1)
        
        print(f"Centroid: {centroids[cluster_index]}")
        print(f"Average Distance to Centroid: {np.mean(distances)}")
        print(f"Min Distance to Centroid: {np.min(distances)}")
        print(f"Max Distance to Centroid: {np.max(distances)}")

    return cluster_1, cluster_2

# Usage example
cluster_1, cluster_2 = read_and_cluster('/home/dhanush/robosuite_vd/robosuite/models/assets/demonstrations/demo_DATA_BP_DUAL_SHOCK/demo.hdf5')