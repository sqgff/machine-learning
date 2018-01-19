import numpy as np


def Kmeans_plus_initiate(data, K):
    '''Use the kmeans++ to initiate K centers'''
    print('Begin kmeans++ for initialization of centers')
    data_ini = np.array(data)
    N = data_ini.shape[0]
    random_indice = np.random.choice(N)
    centers = [data_ini[random_indice]]
    distance = np.linalg.norm(data-centers[-1], axis=1)
    probs = distance**2
    probs = probs / np.sum(probs)
    while len(centers) < K:
        new_c = np.random.choice(N, p=probs)
        centers.append(data_ini[new_c])
        new_dis = np.linalg.norm(data-centers[-1], axis=1)
        distance = np.amin([distance, new_dis], axis=0)
        probs = distance**2
        probs = probs / np.sum(probs)
    print('End initialization of centers\n')
    return centers

def k_means(data_frame, K, epi=1e-4, init_k_plus=True, max_iters=2000, verbose=True):
    data = np.array(data_frame)
    centers = Kmeans_plus_initiate(data, K) if init_k_plus else data[np.random.choice(data.shape[0], size=K)]
    points = [[], [], [], []]
    J_old = np.inf
    Js = []
    k = 0
    print('Begin KMeans Process: ')
    while True:
        k+=1
        if verbose and k%100 == 0:
            print('iter: ', k,': sum of distance = ', J_old)
        #print(centers)
        distances = [np.linalg.norm(data-center, axis=1) for center in centers]
        indices = np.argmin(distances, axis=0)
        dis_min = np.array(distances)[indices, np.arange(indices.shape[0])]
        J_new = np.sum(dis_min**2)
        Js.append(J_new)
        if np.abs(J_new-J_old) < epi:
            break
        if len(Js) > max_iters:
            break
        # otherwise, update centers
        for i in range(indices.shape[0]):
            points[indices[i]].append(i)
        centers = [np.mean(data[l], axis=0) for l in points]
        J_old = J_new
    print('End of KMeans process!\n')
    return centers, points, Js

def predict_kmeans(data_frame, centers):
    data = np.array(data_frame)
    distances = [np.linalg.norm(data-center, axis=1) for center in centers]
    indices = np.argmin(distances, axis=0)
    dis_min = np.array(distances)[indices, np.arange(indices.shape[0])]
    J = np.sum(dis_min**2)
    Points = [[], [], [], []]
    for i in range(indices.shape[0]):
            Points[indices[i]].append(i)
    return indices, J, Points