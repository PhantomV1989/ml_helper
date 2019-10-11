import numpy as np
from sklearn.decomposition import PCA

def manifold_learning_clustering_test():
    '''
    Note! The axis are not uniform in scale
    :return:
    '''
    import plotly.graph_objects as go
    fig = go.Figure()

    def get_density_partitions(dataset, density_interval=0.005, min_size=0):
        if min_size == 0:
            min_size = round(len(dataset) * 0.1)
        density = min_size / density_interval
        axis_cnt = dataset.shape[1]
        axis_partitions = {}
        mdata = [[x, [0] * axis_cnt] for x in dataset]
        tdataset = np.hstack((dataset, np.arange(len(dataset)).reshape(len(dataset), -1)))
        for i in range(axis_cnt):
            partition_id = 0
            _data = tdataset[tdataset[:, i].argsort()]

            ii = 0
            last_state = False
            while True:
                if ii + min_size >= len(_data):
                    break
                else:
                    cwindow = _data[ii:ii + min_size + 1][:, i]
                    cdensity = min_size / (cwindow.max() - cwindow.min())
                    state = cdensity > density
                    if state:
                        if not last_state:
                            partition_id += 1
                        for dd in _data[ii:ii + min_size + 1][:, axis_cnt]:
                            mdata[int(dd)][1][i] = partition_id
                    last_state = state
                ii += 1

        mdata = [[x[0], ''.join([str(y) for y in x[1]])] for x in mdata]
        dense = list(filter(lambda x: not '0' in x[1], mdata))
        kd = set([x[1] for x in dense])
        for d in kd:
            dset = list(filter(lambda x: x[1] == d, mdata))
            tr = []
            for i in range(axis_cnt):
                axi = [x[0][i] for x in dset]
                tr.append([np.min(axi), np.max(axi)])
            axis_partitions[d] = tr
        mdata = [x[1] for x in mdata]
        mdata = ['0' * axis_cnt if '0' in x else x for x in mdata]

        lkeys = {}
        for m in mdata:
            if not m in lkeys:
                lkeys[m] = 1
            else:
                lkeys[m] += 1
        if np.std([lkeys[k] for k in lkeys]) < len(data) / 20:
            return ['0' * axis_cnt] * len(data), {}
        return mdata, axis_partitions

    def get_principle_components(dataset, density_interval=0.005, min_size=0):
        axis_cnt = np.shape(dataset)[1]
        dense_labels, dense_partitions = get_density_partitions(dataset, density_interval=density_interval,
                                                                min_size=min_size)
        pc_bucket = np.zeros_like(dataset)
        dense_clusters = {}
        for i, dl in enumerate(dense_labels):
            if dl != '0' * axis_cnt:
                if not dl in dense_clusters:
                    dense_clusters[dl] = [i]
                else:
                    dense_clusters[dl].append(i)
        if dense_clusters == {}:
            dense_clusters['0' * axis_cnt] = [i for i in range(len(dataset))]

        for dclus in dense_clusters:
            cluster_pos = dense_clusters[dclus]
            cluster_points = dataset[cluster_pos, :]
            krng = 0.05
            max_axis_range = [[cluster_points[:, x].min(), cluster_points[:, x].max()] for x in range(axis_cnt)]
            max_range = np.linalg.norm(
                np.asarray([x[0] for x in max_axis_range]) - np.asarray([x[1] for x in max_axis_range]))
            rng = max_range * krng
            point_bucket = [[] for x in cluster_points]

            for i in range(len(cluster_points) - 1):
                cpoint = cluster_points[i]
                for ii in range(i + 1, len(cluster_points)):
                    npoint = cluster_points[ii]
                    if np.linalg.norm(cpoint - npoint) < rng:  # fills 2 clusters simultaneusly
                        point_bucket[i].append(npoint)
                        point_bucket[ii].append(cpoint)
            den = np.max([len(x) for x in point_bucket])
            for bi in range(len(point_bucket)):
                if len(point_bucket[bi]) > 0:
                    points = np.asarray(point_bucket[bi])
                    pc = PCA(n_components=1).fit(np.asarray(points)).components_  # alrddy len 1
                    pc_bucket[cluster_pos[bi]] = pc * len(point_bucket[bi]) / den / 10

        _debug = []
        # qq = np.max([np.linalg.norm(x) for x in nvectors]) *5
        for i in range(len(dataset)):
            _debug.append(dataset[i])
            _debug.append(dataset[i] + pc_bucket[i])
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
            marker=dict(size=8, color=rrgb(), opacity=0.9,
                        line=dict(width=1, color='black'))
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

    # data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data = [[0.3, 0.3 + i, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    data += [[0.01, 0.2, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    # data += [[0.28, 0.32 + i, 0.42 + i] for i in np.arange(0, 0.3, 0.01)]

    for i in np.linspace(0, 20, 100):
        data += [[np.cos(i) / 40, np.sin(i) / 40, i / 40]]
    for i in range(100):
        data.append(data[np.random.randint(30)] + 0.01 * np.random.normal(scale=1, size=3))
    for ii in np.arange(0, 0.3, 0.03):
        for jj in np.arange(0, 0.3, 0.03):
            data += [[0.9, 0.2 + ii, 0.2 + jj]]
    #  += [[0.5 + i * 0.6, 0.8 - 0.7 * i, 0.45] for i in np.arange(0, 0.4, 0.02)]
    data = np.asarray(data)
    data = np.vstack([data, 0.3 * np.random.rand(180, 3)])
    # data, scalar = normalize_coordinates(data)

    plot_d(data, 'test', data)

    fig.show()
    return
