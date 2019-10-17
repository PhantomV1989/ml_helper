import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def manifold_learning_principle_component():
    '''
    Note! The axis are not uniform in scale
    :return:
    '''
    import plotly.graph_objects as go
    fig = go.Figure()
    krng = 0.04
    xname = 0.5
    gz = 0.2

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
        den = np.max([len(x) for x in point_bucket])
        for bi in range(len(point_bucket)):
            if len(point_bucket[bi]) > 0:
                points = np.asarray(point_bucket[bi])
                pc = PCA(n_components=1).fit(np.asarray(points)).components_  # alrddy len 1
                pc_bucket[bi] = pc * len(point_bucket[bi]) / den

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

    def get_anomaly_score(point, data, vector_field, score_dist, vector_relative_weightage=0.5, d_angle=10):
        scores = []
        l2 = np.linalg.norm
        neighbours, vectors = [], []
        d_angle_rad = d_angle / 180 * np.pi
        offset = -np.cos(2 * d_angle_rad)
        kp = 1 / (1 + offset)
        kn = -1 / (offset - 1)

        def get_manifold_reducer(angle):
            a = np.cos(2 * angle) + offset
            return kp * a if a > 0 else kn * a

        def get_angle(a, b):
            xx = np.dot(a, b) / l2(a) / l2(b)  # due to round off errors, value can be >1
            xx = 1.0 if xx > 1 else xx
            return np.arccos(xx)

        def _h(point, nei, vec):
            v = nei - point
            l2d = l2(v)
            if l2d != 0:
                if vec.mean() == 0.0:
                    manifold_reducer = 0
                else:
                    angle = get_angle(v, vec)
                    angle = np.pi / 2 if np.isnan(angle) else angle
                    manifold_reducer = get_manifold_reducer(angle)
                    # 1 means max reduc, -1 means max increasing score
                dist_rat = l2d / score_dist
                total_reducer = vector_relative_weightage * manifold_reducer + 1 - vector_relative_weightage
                dist_total_reducer = np.tanh(total_reducer / np.power(2.3 * dist_rat, 2))
                score = 1 - dist_total_reducer
                return score
            else:
                return 0.0

        for i, d in enumerate(data):
            v = d - point
            l2d = l2(v)
            if l2d != 0:
                if l2d < score_dist * 3:
                    neighbours.append(d)
                    vectors.append(vector_field[i])
            else:
                return 0.0
        if len(neighbours) == 0:
            return 1.0
        dbsmodel = DBSCAN(eps=0.03, min_samples=5).fit(neighbours)
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
        # tscore = np.power(tscore, 1.5 * np.power(len(scores), -0.92) + 0.5)
        return tscore

    data = [[0.3, 0.3 + i, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    # data += [[0.01, 0.2, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    # data += [[0.28, 0.32 + i, 0.42 + i] for i in np.arange(0, 0.3, 0.01)]

    # for i in np.linspace(0, 20, 100):
    #     data += [[np.cos(i) / 40, np.sin(i) / 40, i / 40]]
    # for i in range(70):
    #     data.append(data[np.random.randint(30)] + 0.01 * np.random.normal(scale=1, size=3))
    #     data.append([0.3, np.random.rand(), np.random.rand()])
    data.append([0.3, 0.36, 0.68])
    data.append([0.3, 0.5, 0.8])
    data.append([0.3, 0.5, 0.42])
    data.append([0.3, 0.6, 0.34])
    data.append([0.3, 0.74, 0.82])
    for ii in np.arange(0, 0.3, 0.02):
        for jj in np.arange(0, 0.3, 0.02):
            data += [[0.9, 0.4 + ii, 0.4 + jj]]
    #  += [[0.5 + i * 0.6, 0.8 - 0.7 * i, 0.45] for i in np.arange(0, 0.4, 0.02)]
    data = np.asarray(data)
    # data = np.vstack([data, 1 * np.random.rand(100, 3)])
    # data, scalar = normalize_coordinates(data)

    plot_d(data, 'test', data)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    rand_points = []
    rand_scores = []
    nvectors = get_principle_components(data, krng)

    for i in [0.3, 0.9]:  # np.arange(0.27, 0.32, 0.01):
        for j in np.arange(0, 1, 0.02):
            for k in np.arange(0, 1, 0.02):
                new_point = [i, j, k]
                score = get_anomaly_score(new_point, data, nvectors, score_dist=gz, vector_relative_weightage=xname)
                qwe = 1
                if score <= qwe:
                    rand_scores.append(score)
                    rand_points.append(new_point)

    # pts = [
    #     [0.3, 0.12, 0.24]
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
        hovertext=[str(x) for x in rand_scores], name=str(krng) + '-' + str(xname) + '-' + str(gz), mode='markers',
        marker=dict(size=4, color=np.asarray(rand_scores), opacity=0.7, colorscale='RdYlGn', reversescale=True)
    ))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

    fig.show()
    return


manifold_learning_principle_component()
