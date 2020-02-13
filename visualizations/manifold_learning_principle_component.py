import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go


def manifold_learning_principle_component(data):
    '''
    Note! The axis are not uniform in scale
    :return:
    '''
    import plotly.graph_objects as go
    fig = go.Figure()
    krng = 0.1
    xvec = 0.7
    gz = 0.1
    min_dist = 1E-3

    def get_hierarchical_principle_components(dataset, rng):
        dataset_cluster_labels = DBSCAN(eps=0.02, min_samples=5).fit_predict(dataset)
        dataset_clusters = {}
        for i, v in enumerate(dataset_cluster_labels):
            if not v in dataset_clusters:
                dataset_clusters[v] = [[i, dataset[i]]]
            else:
                dataset_clusters[v].append([i, dataset[i]])

        pc = [[] for i in range(len(dataset))]
        for c in dataset_clusters:
            clu = dataset_clusters[c]
            clu_emb = [x[1] for x in clu]
            pcc = get_principle_components(clu_emb, rng)
            for i, pos in enumerate([x[0] for x in clu]):
                pc[pos] = pcc[i]
        return np.vstack(pc)

    def get_principle_components(dataset, rng):
        axis_cnt = np.shape(dataset)[1]
        pc_bucket = np.zeros_like(dataset)

        point_bucket = [[] for x in dataset]

        for i in range(len(dataset) - 1):
            cpoint = dataset[i]
            for ii in range(i + 1, len(dataset)):
                npoint = dataset[ii]
                if np.linalg.norm(cpoint - npoint) < rng:  # fills 2 clusters simultaneusly
                    point_bucket[i].append(npoint)
                    point_bucket[ii].append(cpoint)
        # den = np.max([len(x) for x in point_bucket])
        for bi in range(len(point_bucket)):
            if len(point_bucket[bi]) > 0:
                points = np.asarray(point_bucket[bi])
                pc = PCA(n_components=1).fit(np.asarray(points)).components_  # alrddy len 1
                pc_bucket[bi] = pc  # * len(point_bucket[bi]) / den

        _debug = []
        for i in range(len(dataset)):
            _debug.append(dataset[i])
            _debug.append(dataset[i] + pc_bucket[i] / 10)
            _debug.append([None] * axis_cnt)
        plot_l(_debug)
        return pc_bucket

    def rrgb():
        return 'rgb(' + str(np.random.randint(0, 255)) + ',' + str(np.random.randint(0, 255)) + ',' + str(
            np.random.randint(0, 255)) + ')'

    def plot_d(dataset, name, htext):
        x = [x[0] for x in dataset]
        y = [x[1] for x in dataset]
        z = [x[2] for x in dataset]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            hovertext=htext, name=name, mode='markers',  # hoverinfo='text',
            marker=dict(size=6, color=rrgb(), opacity=1)  # ,line=dict(width=1, color='black'))
        ))
        return

    def plot_l(dataset):
        x = [x[0] for x in dataset]
        y = [x[1] for x in dataset]
        z = [x[2] for x in dataset]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines', line=dict(color='red', width=1),
        ))
        return

    def get_anomaly_score(point, data, vector_field, score_dist, vector_relative_weightage=0.5, d_angle=15):
        scores = []
        l2 = np.linalg.norm
        neighbours, vectors = [], []
        d_angle_rad = d_angle / 180 * np.pi
        offset = -np.cos(2 * d_angle_rad)
        kp = 1 / (1 + offset)
        kn = -1 / (offset - 1)

        def get_manifold_reducer(v, vec):
            angle = get_angle(v, vec)
            vlen = l2(vec)
            vlen_sc = np.power(vlen, 2)  # filters for strong directionality
            angle = np.pi / 2 if np.isnan(angle) else angle

            a = np.cos(2 * angle) + offset
            angle_factor = kp * a if a > 0 else kn * a
            reducer = angle_factor * vlen_sc
            return reducer

        def get_angle(a, b):
            xx = np.dot(a, b) / l2(a) / l2(b)  # due to round off errors, value can be >1
            xx = 1.0 if xx > 1 else xx
            return np.arccos(xx)

        def _h(point, nei, vec):
            v = nei - point
            l2d = l2(v)

            if l2d > min_dist:
                if vec.mean() == 0.0:
                    manifold_reducer = 0
                else:
                    manifold_reducer = get_manifold_reducer(v, vec)
                    # 1 means max reduc, -1 means max increasing score
                dist_rat = l2d / score_dist
                total_reducer = vector_relative_weightage * manifold_reducer + 1 - vector_relative_weightage
                dist_total_reducer = np.tanh(total_reducer / np.power(2.3 * dist_rat, 2))
                if dist_total_reducer < 0:
                    dist_total_reducer = 0
                score = 1 - dist_total_reducer
                return score
            else:
                return 0.0

        for i, d in enumerate(data):
            v = d - point
            l2d = l2(v)
            if l2d > min_dist:
                if l2d < score_dist * 2:
                    neighbours.append(d)
                    vectors.append(vector_field[i])
            else:
                return 0.0
        if len(neighbours) == 0:
            return 1.0
        dbsmodel = DBSCAN(eps=0.04, min_samples=5).fit(neighbours)
        _tpoints = {}
        for i, l in enumerate(dbsmodel.labels_):
            if l != -1:
                if not l in _tpoints:
                    _tpoints[l] = [[neighbours[i], vectors[i], l2(point - neighbours[i])]]
                else:
                    _tpoints[l].append([neighbours[i], vectors[i], l2(point - neighbours[i])])
            else:
                score = _h(point, neighbours[i], vectors[i])
                scores.append(score)
        for k in _tpoints:
            nei = _tpoints[k][np.argmin([x[2] for x in _tpoints[k]])][0]
            vnei = np.asarray([x[1] for x in _tpoints[k]])
            for i in range(len(vnei)):
                if np.dot(vnei[0], vnei[i]) < 0:
                    vnei[i] *= -1
            vnei = vnei.mean(axis=0)
            score = _h(point, nei, vnei)
            scores.append(score)
        tscore = np.min(scores) if len(scores) > 0 else 1.0
        tscore = 1 - 1.5 * np.tanh(len(neighbours) / 5) * (1 - tscore)
        if tscore < 0:
            tscore = 0
        return tscore

    data = np.asarray(data)

    plot_d(data, 'test', data)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    rand_points = []
    rand_scores = []
    nvectors = get_hierarchical_principle_components(data, krng)

    for i in [0.3]:  # np.arange(0.27, 0.32, 0.01):
        for j in np.arange(0, 1, 0.02):
            for k in np.arange(0, 1, 0.02):
                new_point = [i, j, k]
                score = get_anomaly_score(new_point, data, nvectors, score_dist=gz, vector_relative_weightage=xvec)

                rand_scores.append(score)
                rand_points.append(new_point)

    # pts = [
    #     [0.9, 0.58, 0.76]
    #     # [0.3, 0.22, 0.32]
    # ]
    # for new_point in pts:
    #     score = get_anomaly_score(new_point, data, nvectors, score_dist=gz, vector_relative_weightage=xname)
    #     rand_scores.append(score)
    #     rand_points.append(new_point)

    # for i in range(3000):
    #     new_point = data[np.random.randint(len(data))] + 0.05 * np.random.normal(scale=1, size=3)
    #     # new_point = np.random.rand(3)
    #     score = get_anomaly_score(new_point, data, nvectors, grid_size=gz, x=xname)
    #     if score < 0.9:
    #         rand_scores.append(score)
    #         rand_points.append(new_point)

    x = [x[0] for x in rand_points]
    y = [x[1] for x in rand_points]
    z = [x[2] for x in rand_points]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        hovertext=[str(x) for x in rand_scores], name=str(krng) + '-' + str(xvec) + '-' + str(gz), mode='markers',
        marker=dict(size=4, color=np.asarray(rand_scores), opacity=0.7, colorscale='RdYlGn', reversescale=True)
    ))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

    fig.show()
    return


data = [[0.3, 0.3 + i, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]

data.append([0.3, 0.36, 0.68])
data.append([0.3, 0.5, 0.8])
data.append([0.3, 0.5, 0.6])
data.append([0.3, 0.6, 0.34])
data.append([0.3, 0.74, 0.82])
data.append([0.3, 0.96, 0.06])
data.append([0.3, 0.96, 0.96])
data.append([0.3, 0.98, 0.98])
data.append([0.3, 0.92, 0.92])
data.append([0.3, 0.88, 0.88])


def data_cross():
    data_cross = [[0.3, 0.3 + i, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    data_cross += [[0.3, 0.3 + i, 0.6 - i] for i in np.arange(0, 0.3, 0.01)]
    return data_cross


def data_a():
    data = [[0.3, 0.3 + i, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    data += [[0.3, 0.3 + i, 0.7 - i] for i in np.arange(0, 0.07, 0.005)]
    data += [[0.3, 0.33 + i, 0.7 - i] for i in np.arange(0, 0.07, 0.005)]
    data += [[0.3, 0.36 + i, 0.7 - i] for i in np.arange(0, 0.07, 0.005)]
    return data


# square
data_square = []
for ii in np.arange(0, 0.3, 0.02):
    for jj in np.arange(0, 0.3, 0.02):
        data_square += [[0.9, 0.4 + ii, 0.4 + jj]]

manifold_learning_principle_component(data_a())
