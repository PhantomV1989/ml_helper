import numpy as np


def visualizing_vectors():
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

    def get_vectors_of_only_dense_clusters(dataset, density_interval=0.005, min_size=0):
        axis_cnt = np.shape(dataset)[1]
        dense_labels, dense_partitions = get_density_partitions(dataset, density_interval=density_interval,
                                                                min_size=min_size)
        nvectors = np.zeros_like(dataset)
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
            # krng = -0.0869 * np.log(len(cluster_points)) + 0.577 if len(cluster_points) < 400 else 0.05
            krng = 0.15
            max_axis_range = [[cluster_points[:, x].min(), cluster_points[:, x].max()] for x in range(axis_cnt)]
            max_range = np.linalg.norm(
                np.asarray([x[0] for x in max_axis_range]) - np.asarray([x[1] for x in max_axis_range]))
            rng = max_range * krng
            vector_buckets = [[] for x in cluster_points]

            for i in range(len(cluster_points) - 1):
                cpoint = cluster_points[i]
                for ii in range(i + 1, len(cluster_points)):
                    npoint = cluster_points[ii]
                    if np.linalg.norm(cpoint - npoint) < rng:  # fills 2 clusters simultaneusly
                        outdir = cpoint - npoint
                        vector_buckets[i].append(outdir)
                        vector_buckets[ii].append(-outdir)
            for bi in range(len(vector_buckets)):
                if len(vector_buckets[bi]) > 0:
                    divect = np.asarray(vector_buckets[bi]).sum(axis=0)
                    unit_vec = divect / np.linalg.norm(divect)
                    nvec = unit_vec * len(vector_buckets[bi])  # / np.linalg.norm(divect)
                    nvectors[cluster_pos[bi]] = nvec

        _debug = []
        qq = np.max([np.linalg.norm(x) for x in nvectors]) * 5
        for i in range(len(dataset)):
            _debug.append(dataset[i])
            _debug.append(dataset[i] + nvectors[i] /qq)
            _debug.append([None] * axis_cnt)
        plot_l(_debug)
        return nvectors

    # data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data = [[0.3, 0.3 + i, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    data += [[0.32, 0.3 + i, 0.4 + i] for i in np.arange(0, 0.3, 0.01)]
    data += [[0.28, 0.32 + i, 0.42 + i] for i in np.arange(0, 0.3, 0.01)]
    data += [[0.1, 1.3 - i, -0.4 + i] for i in np.arange(0, 0.3, 0.01)]

    for ii in np.arange(0, 0.3, 0.03):
        for jj in np.arange(0, 0.3, 0.03):
            data += [[0.9, 0.2 + ii, 0.2 + jj]]

    for ii in np.arange(0, 0.3, 0.05):
        for jj in np.arange(0, 0.3, 0.05):
            for kk in np.arange(0, 2, 0.05):
                data += [[0.4 + ii, 1 + jj, 0.2 + kk]]

    for i in range(100):
        data.append(np.random.normal(size=[3], scale=0.1))

    data = np.asarray(data)
    data = np.vstack([data, np.random.rand(30, 3)])

    plot_d(data, 'test', data)
    nvectors = get_vectors_of_only_dense_clusters(data)

    fig.show()
    return


visualizing_vectors()
