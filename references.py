import torch as t
import numpy as np
import time
import copy
import ml_helper
from sklearn import tree
from sklearn.metrics import accuracy_score
import datetime as dt

tdevice = t.device('cpu')


def torch_backward_list_tensor_test():
    # conclusion, do not backward a list, 30x slower
    a = []
    b = []
    sz = 1
    iter_count = 1000
    while True:
        x = t.rand(sz, device=tdevice)
        if x[0] > 0.5:
            if len(a) < 80: a.append(x)
        else:
            if len(b) < 80: b.append(x)
        if len(a) >= 80 and len(b) >= 80:
            break

    w1 = t.nn.Linear(sz, 2)

    params = [
        {'params': w1.parameters()}
        # {'params': w2},
    ]
    optim = t.optim.SGD(params, lr=0.01)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~ANN list backwards~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    time_a = dt.datetime.now()
    for i in range(iter_count):
        o1 = [w1(aa).softmax(dim=0) for aa in a]
        loss_a = [t.nn.BCELoss()(oo, t.tensor([1.0, 0], device=tdevice)) for oo in o1]

        o2 = [w1(aa).softmax(dim=0) for aa in b]
        loss_b = [t.nn.BCELoss()(oo, t.tensor([0.0, 1.0], device=tdevice)) for oo in o2]

        loss = loss_a + loss_b
        [l.backward(retain_graph=True) for l in loss]
        optim.step()
        optim.zero_grad()

        o1 = t.stack([x[0] for x in o1]).data.cpu().numpy()
        o2 = t.stack([x[0] for x in o2]).data.cpu().numpy()

        y1 = [0] * len(o1)
        y2 = [1] * len(o2)

        _y = np.hstack([o1, o2])
        y = y1 + y2
        r = ml_helper.evaluate(_y, y)
        print('     TP:', r['True positive'], '    TN:', r['True negative'], '   Loss:',
              t.stack(loss).mean().data.cpu().numpy())
    time_b = dt.datetime.now()
    time_diff = time_b - time_a
    print('     Finished in ', time_diff)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~ANN tensor backwards~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    time_a = dt.datetime.now()
    for i in range(iter_count):
        o1 = w1(t.stack(a)).softmax(dim=1)
        o2 = w1(t.stack(b)).softmax(dim=1)
        _y = t.cat([o1, o2])
        y = t.stack([t.tensor([1.0, 0], device=tdevice)] * len(o1) + [t.tensor([0, 1.0], device=tdevice)] * len(o2))
        loss = t.nn.BCELoss()(_y, y)

        loss.backward()
        optim.step()
        optim.zero_grad()

        o1 = [x[0].data.item() for x in o1]
        o2 = [x[0].data.item() for x in o2]
        y1 = [0] * len(o1)
        y2 = [1] * len(o2)

        _y = np.hstack([o1, o2])
        y = y1 + y2
        r = ml_helper.evaluate(_y, y)
        print('     TP:', r['True positive'], '    TN:', r['True negative'], '   Loss:', loss.data.cpu().numpy())
    time_b = dt.datetime.now()
    time_diff = time_b - time_a
    print('     Finished in ', time_diff)
    return


def dynamic_learning_rate():
    w = t.tensor([1, 1], dtype=t.float32, requires_grad=True)
    w._grad = t.tensor([1, 1], dtype=t.float32)

    optim = t.optim.SGD([w], lr=0.02)
    lr = np.arange(0, 1, 0.1)

    for i in lr:
        for j in optim.param_groups:
            j['lr'] = i
        a = copy.deepcopy(w.data.numpy())
        optim.step()
        b = copy.deepcopy(w.data.numpy())
        print('Grad @ ', w._grad, '.  LR @ ', i, '  Step diff @ ', b - a)
    return


def test_grad_accumulation():
    # not recommended, too slow compared to cat tensor then backward step
    x = t.rand([2, 100])
    w = t.rand([100, 2], requires_grad=True)
    y = t.tensor(np.asarray([[0, 1.0], [1.0, 0]]), dtype=t.float32)
    optim = t.optim.SGD([w], lr=0.01)
    _y = x.matmul(w).softmax(dim=1)

    loss = t.nn.BCELoss()(_y, y)
    loss.backward(retain_graph=True)
    print('Grad of 2 data points is ', w._grad[0].data)
    optim.zero_grad()

    loss = t.nn.BCELoss()(_y[0], y[0])
    loss.backward(retain_graph=True)
    print('Grad of first data point is ', w._grad[0].data)
    optim.zero_grad()

    loss = t.nn.BCELoss()(_y[1], y[1])
    loss.backward(retain_graph=True)
    print('Grad of second data point is ', w._grad[0].data)
    optim.zero_grad()

    loss = t.nn.BCELoss()(_y[0], y[0])
    loss.backward(retain_graph=True)
    loss = t.nn.BCELoss()(_y[1], y[1])
    loss.backward(retain_graph=True)
    print('Grad of 2 data point backwarded consecutively is ', w._grad[0].data)
    optim.zero_grad()

    _yf = _y.reshape(shape=[-1])
    yf = y.reshape(shape=[-1])
    loss = t.nn.BCELoss()(_yf, yf)
    loss.backward(retain_graph=True)
    print('Grad of 2 data points with flattened structure is ', w._grad[0].data)
    optim.zero_grad()

    return


def noiseless_timeseries_lstm_vs_ann():
    # conclusion: both 100% valid
    pattern_a = '123' * 100
    pattern_b = '789' * 100
    emb_size = 3
    emb = t.rand([10, emb_size], device=tdevice)

    data_count = 100
    time_len = 10

    label_a = []
    label_b = []

    ya = t.tensor([[1.0, 0]] * data_count, device=tdevice)
    yb = t.tensor([[0, 1.0]] * data_count, device=tdevice)

    y = t.cat([ya, yb])

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    for i in range(data_count):
        np.random.seed(dt.datetime.now().microsecond)
        ii = np.random.randint(0, len(pattern_a) - time_len)
        label_a.append(to_emb(pattern_a[ii:ii + time_len]))
        label_b.append(to_emb(pattern_b[ii:ii + time_len]))

    def _h(fp, op):
        aout = [fp(x) for x in label_a]
        bout = [fp(x) for x in label_b]

        aout = t.stack(aout)
        bout = t.stack(bout)
        _y = t.cat([aout, bout])

        loss = t.nn.BCELoss()(_y, y)

        loss.backward()  # slow
        op.step()
        op.zero_grad()

        _y1 = np.asarray([x[0].data.item() for x in _y])
        y1 = np.asarray([x[0].data.item() for x in y])
        scores = ml_helper.evaluate(_y1, y1)
        print(' F1', scores['F1'], '   TP:', scores['True positive'], '  TN:', scores['True negative'])
        return loss.data.cpu().item()

    def test(fp):
        a = []
        b = []

        for i in range(100):
            np.random.seed(dt.datetime.now().microsecond)
            ii = np.random.randint(0, len(pattern_a) - time_len)
            a.append(to_emb(pattern_a[ii:ii + time_len]))
            b.append(to_emb(pattern_b[ii:ii + time_len]))

        ra = [fp(x)[0].data.item() for x in a]
        rb = [fp(x)[0].data.item() for x in b]

        correct = len(list(filter(lambda x: x > 0.5, ra))) + len(list(filter(lambda x: x <= 0.5, rb)))
        wrong = len(list(filter(lambda x: x < 0.5, ra))) + len(list(filter(lambda x: x >= 0.5, rb)))

        return {'correct': correct, 'wrong': wrong}

    def lstm_test():
        print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')
        out_size = 10
        layers = 1
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=out_size, batch_size=1,
                                                       num_of_layers=layers,
                                                       device=tdevice)
        last_w = t.nn.Linear(out_size * layers, 2)
        last_w.to(tdevice)

        last_w2 = t.nn.Linear(100, 2)
        last_w2.to(tdevice)

        params = [
            {'params': list(lstm.parameters())},
            {'params': list(last_w.parameters())},
            {'params': list(last_w2.parameters())}
        ]
        lstm_optim = t.optim.SGD(params, lr=0.03)

        def lstm_fprop(x):  # upper 60~72%  on 100 iter
            out, h = lstm(x.unsqueeze(1), init)
            ii = out.reshape(-1)
            r = last_w2(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop1(x): # dont work at all
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[0].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop2(x):  # 100% valid accuracy
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[1].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        for i in range(100):
            print('Iter[2]:', i)
            loss = _h(lstm_fprop1, lstm_optim)
            print(i, ' Loss:', loss)
        r2 = [test(lstm_fprop1) for i in range(50)]

        # dynamic length test
        test_a = lstm_fprop2(to_emb(pattern_a[61:78]))
        test_b = lstm_fprop2(to_emb(pattern_b[18:87]))
        return

    def ann_test():
        # 100% valid results, but no dynamic len
        print('~~~~~~~~~~~~~~~~ANN test~~~~~~~~~~~~~~~~~~~~')
        w1 = t.nn.Linear(time_len * emb_size, 100)
        w2 = t.nn.Linear(100, 2)

        params = [
            {'params': w1.parameters()},
            {'params': w2.parameters()}
        ]

        optim = t.optim.SGD(params, lr=0.03)

        def ann_fprop(x):
            x = x.reshape(-1)
            o = w1(x)
            r = w2(o).softmax(dim=0)
            return r

        for i in range(100):
            loss = _h(ann_fprop, optim)
            print(i, ' Loss:', loss)
        r = [test(ann_fprop) for i in range(100)]
        return r

    # rl = lstm_test()
    rann = ann_test()
    return
