import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, HDBSCAN, BisectingKMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



N_KMEANS_CLUSTERS = 800 # 'k' in k-means or number of clusters

#index of each numeric, and categoric column
numeric_columns = [4,15,88,89,114,115,116]
categoric_columns = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
                     38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,
                     71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,
                     105,106,107,108,109,110,111,112,113]

#pull vehicle data excluding Make, Model, and MSRP
def extract_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            corrected_row = {'Year': int(row['Year']),
                             'Engine Fuel Type': row['Engine Fuel Type'],
                             'Engine HP': int(row['Engine HP']),
                             'Engine Cylinders': int(row['Engine Cylinders']),
                             'Transmission Type': row['Transmission Type'],
                             'Driven_Wheels': row['Driven_Wheels'],
                             'Number of Doors': int(row['Number of Doors']),
                             'Market Category': row['Market Category'],
                             'Vehicle Size': row['Vehicle Size'],
                             'Vehicle Style': row['Vehicle Style'],
                             'highway MPG': int(row['highway MPG']),
                             'city mpg': int(row['city mpg']),
                             'Popularity': int(row['Popularity'])}
            data.append(corrected_row)
    return data

#pull complete vehicle data
def extract_all_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            corrected_row = {'Make': row['Make'],
                             'Model': row['Model'],
                             'Year': int(row['Year']),
                             'Engine Fuel Type': row['Engine Fuel Type'],
                             'Engine HP': int(row['Engine HP']),
                             'Engine Cylinders': int(row['Engine Cylinders']),
                             'Transmission Type': row['Transmission Type'],
                             'Driven_Wheels': row['Driven_Wheels'],
                             'Number of Doors': int(row['Number of Doors']),
                             'Market Category': row['Market Category'],
                             'Vehicle Size': row['Vehicle Size'],
                             'Vehicle Style': row['Vehicle Style'],
                             'highway MPG': int(row['highway MPG']),
                             'city mpg': int(row['city mpg']),
                             'Popularity': int(row['Popularity']),
                             'MSRP': int(row['MSRP'])}
            data.append(corrected_row)
    return data
    
def normalize_and_fit(data):

    #prepare dict_vectorizer, which applies one-hot-encoding to catergoric features
    dict_vectorizer = DictVectorizer(sparse=False)
    X_temp = dict_vectorizer.fit_transform(data)
    all_feature_names = dict_vectorizer.get_feature_names_out()

    #prepare standard scaler for numeric columns
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", "passthrough", categoric_columns)
        ],
        remainder="drop"
    )

    #assemble pipeline using transformer and dict_vectorizer in order to apply changes to data
    pipeline = Pipeline([
        ("dict_vect", dict_vectorizer),
        ("preprocessing", transformer)
    ])

    #appy changes to data via the pipeline
    prepped_data = pipeline.fit_transform(data)
    numeric_feature_names = [all_feature_names[i] for i in numeric_columns]
    categorical_feature_names = [all_feature_names[i] for i in categoric_columns]
    final_feature_names = numeric_feature_names + categorical_feature_names

    return prepped_data, final_feature_names

def k_means_cluster(data):
    kmeans = BisectingKMeans(n_clusters=N_KMEANS_CLUSTERS, random_state=0, n_init=12)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, kmeans.labels_)
    print(f"Silhouette Score (k=800): {score:.4f}")
    return data, kmeans, labels

def k_means_meta_cluster(data):
    data_prepped = np.array(list(data.values()))
    ids = list(data.keys())
    kmeans = BisectingKMeans(n_clusters=20, random_state=0, n_init=12)
    labels = kmeans.fit_predict(data_prepped)
    score = silhouette_score(data_prepped, kmeans.labels_)
    print(f"Silhouette Score *Meta-clusters* (k=20): {score:.4f}")
    clustered_data = dict(zip(ids, labels))
    return clustered_data, kmeans, labels

def add_meta_cluster_to_car(clustered_data, full_data):
    for car in full_data:
        car['meta_cluster'] = clustered_data[car['cluster']]
    return full_data

def hdbscan_cluster(data):
    hdb = HDBSCAN(min_cluster_size=20, n_jobs=8, cluster_selection_method='leaf')
    labels = hdb.fit_predict(data)
    score = silhouette_score(data, hdb.labels_)
    print(f"Silhouette Score (HDBSCAN): {score:.4f}")
    return data, hdb, labels

def PCA_reduction(clusters, kmeans, final_feature_names, full_data):
    labels = kmeans.labels_
    pca = PCA(n_components=3, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(clusters)

    #print weight of 10 most influential features for each PCA axis
    for i, pc in enumerate(pca.components_):
        loadings = list(zip(final_feature_names, pc))
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n=== Top features in PC{i+1} ===")
        for feat, weight in loadings[:10]:
            print(f"{feat}: {weight:.4f}")

    #attach PCA coordinate to cars
    for i, row in enumerate(full_data):
        row["P1"] = float(X_pca[i, 0])
        row["P2"] = float(X_pca[i, 1])
        row["P3"] = float(X_pca[i, 2])


    unique_labels = set(labels)
    num_clusters = len(unique_labels)
    groups = {label: [] for label in unique_labels}
    groups_avg_PCA = {label: [] for label in unique_labels}
    return groups, groups_avg_PCA, X_pca

def attach_data_to_cars(full_data, groups, groups_avg_PCA):
    #attach vehicle to cluster it belongs too
    for row in full_data:
        cluster_label = row['cluster']
        groups[cluster_label].append(row)

    for i in range(N_KMEANS_CLUSTERS):
        groups_avg_PCA[i].append(0)
        groups_avg_PCA[i].append(0)
        groups_avg_PCA[i].append(0)

    #find centroid of each cluster by finding avg P1, P2, and P3 for each cluster
    for i in range(N_KMEANS_CLUSTERS):
        count = 0
        for car in groups[i]:
            groups_avg_PCA[i][0] += car['P1']
            groups_avg_PCA[i][1] += car['P2']
            groups_avg_PCA[i][2] += car['P3']
            count += 1
        groups_avg_PCA[i][0] /= count
        groups_avg_PCA[i][1] /= count
        groups_avg_PCA[i][2] /= count
        
    
    #output full list of cluster members and key values
    for lbl in sorted(groups.keys(), key=lambda x: int(x)):  
        print('cluster', lbl)
        for row in groups[lbl]:
            print(row['Year'], row['Make'], row['Model'], row['MSRP'], row['P1'], row['P2'], row['P3'])
    
    return full_data, groups_avg_PCA

def visualize_clusters(X_pca, labels):
    # use matplotlib for basic visualization of PCA reduced clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        X_pca[:, 0], 
        X_pca[:, 1], 
        X_pca[:, 2], 
        c=labels, 
        cmap="rainbow", 
        alpha=0.7
    )

    ax.set_title("K-Means Clusters (k=800) in 3D PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()

def attach_clusters_to_data(full_data, kmeans):
    labels = kmeans.labels_
    for i, row in enumerate(full_data):
        row["cluster"] = int(labels[i])

def find_and_assign_value_scores(full_data):
    df = pd.DataFrame(full_data)
    df['cluster_avg'] = df.groupby('cluster')['MSRP'].transform('mean')
    df['value_score'] = df['cluster_avg'] / df['MSRP']
    new_full_data = df.to_dict(orient='records')
    return new_full_data

def find_and_assign_meta_values(data):
    df = pd.DataFrame(data)
    df['meta_cluster_avg'] = df.groupby('meta_cluster')['value_score'].transform('mean')
    df['meta_value_score'] = df['value_score'] / df['meta_cluster_avg']
    df.drop(columns=['meta_cluster_avg'], inplace=True)
    updated_vehicles = df.to_dict(orient='records')
    return updated_vehicles

def write_data_to_csv(data):
    df = pd.DataFrame(data)
    df.to_csv('final_vehicle_data.csv', index=False)

def main():
    data = extract_data('cars.csv')
    full_data = extract_all_data('cars.csv')

    prepped_data, final_feature_names = normalize_and_fit(data)

    clusters, kmeans, labels = k_means_cluster(prepped_data)
    #clusters, hdb, labels = hdbscan_cluster(prepped_data)

    attach_clusters_to_data(full_data, kmeans)

    groups, groups_avg_PCA, X_pca = PCA_reduction(clusters, kmeans, final_feature_names, full_data)

    full_data, groups_avg_PCA = attach_data_to_cars(full_data, groups, groups_avg_PCA)

    clustered_data, meta_kmeans, meta_labels = k_means_meta_cluster(groups_avg_PCA)

    add_meta_cluster_to_car(clustered_data, full_data)

    value_full_data = find_and_assign_value_scores(full_data)

    final_full_data = find_and_assign_meta_values(value_full_data)

    write_data_to_csv(final_full_data)
       
    #visualize_clusters(X_pca, labels)


main()