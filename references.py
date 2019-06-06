import torch as t
import numpy as np
import time
import copy
import ml_helper
import datetime as dt

'''
Conclusion: 
Ordered vs unordered context:
    Use LSTM cell state if you are not sure if the context is ordered or not
    If confirmed unordered, use ANN because it is faster to train compared to LSTM

LSTM vs ANN time series:
    For fixed length time series, ANN and LSTM are comparable. But LSTM is harder to train
'''

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


# you cannot just use a new optim with same params and different lr, lr wont change like that!
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

        def lstm_fprop1(x):  # dont work at all
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


# timeseries
def ordered_context_lstm_vs_ann():
    # setup: A is a series of repeated 123, B is repeated 789
    # Noise is any number from 0-9 injected into A and B, higher noise, A and B get more and more similar
    # result

    # LSTM cell
    # 100% valid accuracy at 0% noise
    # 99~100% valid accuracy at 10% noise, time_len=10, iter 300, lr 0.05, loss 0.008
    # 96~99% valid accuracy at 20% noise, time_len=10, iter 300, lr 0.05, loss 0.56
    # 98~100% valid accuracy at 30% noise, time_len=10, iter 300, lr 0.07, loss 0.09
    # 92~95% valid accuracy at 50% noise, time_len=10, iter 400, lr 0.07, loss 0.112
    # 80~84% valid accuracy at 70% noise, time_len=10, iter 1200, lr 0.07->0.02, loss 0.3609, still decreasing
    # 49% valid accuracy at 90% noise, time_len=10, iter 1500, lr 0.07->0.02, loss 0.6847, still decreasing

    # ANN
    # 100% valid accuracy at 0% noise
    # 92% valid accuracy at 30% noise, time_len=10, iter 800, lr 0.05
    # 95% valid accuracy at 50% noise, time_len=10, iter 800, lr 0.05
    # 77% valid accuracy at 70% noise, time_len=10, iter 800, lr 0.05
    # 87~89% valid accuracy at 70% noise, time_len=20, iter 800, lr 0.05

    # Conclusion: Lstm is slightly better than ANN, but ANN trains way faster.
    # Always use LSTM h[1] cell state for training of noisy timeseries context
    # Higher noise levels hence lower training/validation accuracy needs to be compensated for longer time periods
    # for each data (increase window size)

    pattern_a = '123' * 100
    pattern_b = '789' * 100
    s = '1234567890'
    emb_size = 3
    data_count = 100
    time_len = 30

    noise_percent = 0.7
    emb = t.rand([10, emb_size], device=tdevice)

    pattern_a = list(pattern_a)
    pattern_b = list(pattern_b)
    for i in range(len(pattern_a)):

        np.random.seed(dt.datetime.now().microsecond)
        if np.random.rand() < noise_percent:
            pattern_a[i] = np.random.choice(list(s))
        if np.random.rand() < noise_percent:
            pattern_b[i] = np.random.choice(list(s))
    pattern_a = ''.join(pattern_a)
    pattern_b = ''.join(pattern_b)

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

    def _h(fp, op, verbose=True):
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
        if verbose:
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
        lstm_optim = t.optim.SGD(params, lr=0.07)

        def lstm_fprop(x):  # output
            # 86~92% valid accuracy at 50% noise, iter 1200, lr 0.07, loss 0.233
            # 67% valid accuracy at 70% noise, iter 1200, lr 0.07, loss 0.5517
            out, h = lstm(x.unsqueeze(1), init)
            ii = out.reshape(-1)
            r = last_w2(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop1(x):  # h[0] hidden state
            # 80~88% valid accuracy at 20% noise, time_len=10, iter 300, lr 0.07, loss 0.56
            # 81~85% valid accuracy at 50% noise, time_len=10, iter 1200, lr 0.07, loss 0.4177
            # 55~60% valid accuracy at 70% noise, time_len=10, iter 1200, lr 0.07, loss 0.6147

            out, h = lstm(x.unsqueeze(1), init)
            ii = h[0].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop2(x):  # h[1] cell state
            # 100% valid accuracy at 0% noise
            # 99~100% valid accuracy at 10% noise, time_len=10, iter 300, lr 0.05, loss 0.008
            # 96~99% valid accuracy at 20% noise, time_len=10, iter 300, lr 0.05, loss 0.56
            # 98~100% valid accuracy at 30% noise, time_len=10, iter 300, lr 0.07, loss 0.09
            # 92~95% valid accuracy at 50% noise, time_len=10, iter 400, lr 0.07, loss 0.112
            # 80~84% valid accuracy at 70% noise, time_len=10, iter 1200, lr 0.07->0.02, loss 0.3609, still decreasing
            # 49% valid accuracy at 90% noise, time_len=10, iter 1500, lr 0.07->0.02, loss 0.6847, still decreasing
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[1].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def one(f, lstm_optim):
            for i in range(1200):
                if i > 400:
                    lstm_optim = t.optim.SGD(params, lr=0.04)
                elif i > 800:
                    lstm_optim = t.optim.SGD(params, lr=0.02)
                elif i > 1200:
                    lstm_optim = t.optim.SGD(params, lr=0.01)

                if i % 10 == 0:
                    loss = _h(f, lstm_optim, verbose=True)
                    print(i, '   Noise:', noise_percent, '  Loss:', loss)
                else:
                    loss = _h(f, lstm_optim, verbose=False)

            r = [test(f) for i in range(50)]
            return r

        r1 = one(lstm_fprop, lstm_optim)

        # dynamic length test, longer seq yields better confidence at higher noise levels
        test_a = lstm_fprop(to_emb(pattern_a[61:78]))
        test_b = lstm_fprop(to_emb(pattern_b[18:87]))
        return

    def ann_test():
        # 100% valid accuracy at 0% noise
        # 92% valid accuracy at 30% noise, time_len=10, iter 800, lr 0.05
        # 95% valid accuracy at 50% noise, time_len=10, iter 800, lr 0.05
        # 77% valid accuracy at 70% noise, time_len=10, iter 800, lr 0.05
        # 87~89% valid accuracy at 70% noise, time_len=20, iter 800, lr 0.05
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

        for i in range(800):
            loss = _h(ann_fprop, optim)
            print(i, '   Noise:', noise_percent, '  Loss:', loss)
        r = [test(ann_fprop) for i in range(100)]
        return r

    # rl = lstm_test()
    rann = ann_test()
    return


def unordered_context_test_same_unit_length():  # contains atomic patterns 1 letter in length
    # in this test, B contains a random order of (24680)s while A contains a random order of (13579)s.
    # Noise perturbs A such that A contains numbers normally not found in its range of (1,3,5)
    # Results:
    # 30% noise: LSTM cell state (96.2%), ANN flattened (90%), ANN mean aggregrated(95%)
    # 50% noise  LSTM cell state (86.19%), ANN flattened (83%), ANN mean aggregrated(87.2%)
    # 70% noise  LSTM cell state (74.7%), ANN flattened (72%), ANN mean aggregrated(75.185%)
    # Conclusion: LSTM(cell state) and ANN(input mean aggregated) are comparable, but ANN trains faster while LSTM is
    # flexible in data length

    s = list('0123456789')
    s1 = list('13579')  # identifiers for label A
    s2 = list('24680')
    context_length = 7
    emb_size = 3
    data_count = 100
    emb = t.rand([10, emb_size], device=tdevice)

    ref_a = ''.join(np.random.choice(s1, 100 * 3))
    ref_b = ''.join(np.random.choice(s2, 100 * 3))

    noise_percent = 0.7  # for polluting A, higher noise, A is less discerning to B, aka harder separation
    ref_a = list(ref_a)
    ref_b = list(ref_b)
    for i in range(len(ref_a)):
        np.random.seed(dt.datetime.now().microsecond)
        if np.random.rand() < noise_percent:
            ref_a[i] = np.random.choice(s)
        if np.random.rand() < noise_percent:
            ref_b[i] = np.random.choice(s)

    ref_a = ''.join(ref_a)
    ref_b = ''.join(ref_b)

    label_a = []
    label_b = []

    ya = t.tensor([[1.0, 0]] * data_count, device=tdevice)
    yb = t.tensor([[0, 1.0]] * data_count, device=tdevice)

    y = t.cat([ya, yb])

    for i in range(data_count):
        ii = np.random.randint(0, len(ref_a) - context_length)
        label_a.append(ref_a[ii:ii + context_length])
        label_b.append(ref_b[ii:ii + context_length])

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    def _h(fp, op, verbose=True):
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
        if verbose:
            print(' F1', scores['F1'], '   TP:', scores['True positive'], '  TN:', scores['True negative'])
        return loss.data.cpu().item()

    def test(fp):
        a = []
        b = []

        for i in range(data_count):
            ii = np.random.randint(0, len(ref_a) - context_length)
            a.append(to_emb(ref_a[ii:ii + context_length]))
            b.append(to_emb(ref_b[ii:ii + context_length]))

        ra = [fp(x)[0].data.item() for x in a]
        rb = [fp(x)[0].data.item() for x in b]

        correct = len(list(filter(lambda x: x > 0.5, ra))) + len(list(filter(lambda x: x <= 0.5, rb)))
        wrong = len(list(filter(lambda x: x < 0.5, ra))) + len(list(filter(lambda x: x >= 0.5, rb)))

        return {'correct': correct, 'wrong': wrong}

    label_a = [to_emb(x) for x in label_a]
    label_b = [to_emb(x) for x in label_b]

    def lstm_test():
        print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')
        out_size = 10
        layers = 1
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=out_size, batch_size=1,
                                                       num_of_layers=layers,
                                                       device=tdevice)
        last_w = t.nn.Linear(out_size * layers, 2)
        last_w.to(tdevice)

        last_w2 = t.nn.Linear(out_size, 2)
        last_w2.to(tdevice)

        params = [
            {'params': list(lstm.parameters())},
            {'params': list(last_w.parameters())},
            {'params': list(last_w2.parameters())}
        ]
        lstm_optim = t.optim.SGD(params, lr=0.08)

        def lstm_fprop(x):  # uses out

            out, h = lstm(x.unsqueeze(1), init)
            ii = out.reshape(-1)
            r = last_w2(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop1(x):  # uses h[0], hidden state
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[0].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop2(x):  # uses h[1], cell state
            # 95% iter 1200, noise 0%, context len 7
            # 96.2% iter 1200, noise 30%, context len 7
            # 86.19% iter 1200, noise 50%, context len 7
            # 74.7% iter 1200, noise 70%, context len 7
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[1].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def one(f, lstm_optim):
            for i in range(1200):
                if i > 400:
                    optim = t.optim.SGD(params, lr=0.06)
                elif i > 800:
                    optim = t.optim.SGD(params, lr=0.03)
                elif i > 1000:
                    optim = t.optim.SGD(params, lr=0.01)

                if i % 10 == 0:
                    loss = _h(f, lstm_optim, verbose=True)
                    print(i, '  Noise:', noise_percent, ' Loss:', loss)
                else:
                    loss = _h(f, lstm_optim, verbose=False)

            r = [test(f) for i in range(50)]
            return r

        r1 = one(lstm_fprop2, lstm_optim)
        return

    def ann_test_flattened():
        # 90% iter 50, noise 30%, context len 7
        # 83% iter 50, noise 50%, context len 7
        # 72% iter 50, noise 70%, context len 7
        print('~~~~~~~~~~~~~~~~ANN test~~~~~~~~~~~~~~~~~~~~')
        w1 = t.nn.Linear(context_length * emb_size, 100)
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

        for i in range(1200):
            loss = _h(ann_fprop, optim)
            print(i, ' Loss:', loss)
        r = [test(ann_fprop) for i in range(100)]
        return r

    def ann_test_aggregration():
        print('~~~~~~~~~~~~~~~~ANN test~~~~~~~~~~~~~~~~~~~~')
        w1 = t.nn.Linear(emb_size, 6000)
        w1.to(tdevice)

        w2 = t.nn.Linear(6000, 2)
        w2.to(tdevice)

        params = [
            {'params': w1.parameters()},
            {'params': w2.parameters()}
        ]

        optim = t.optim.SGD(params, lr=0.005)

        def fprop_v1(x):  # w1 first, then mean then w2
            # Abysmal results, highly not recommended
            # 60~65% iter 1200 loss  0.6557
            o = [w1(xx) for xx in x]
            o = t.stack(o)
            o = t.mean(o, dim=0)
            r = w2(o).softmax(dim=0)
            return r

        def fprop_v2(x):  # mean, then w1 then w2
            # 100% iter 1200, noise 0%, context len 7
            # 95% iter 1200, noise 30%, context len 7
            # 80.9% iter 1200, noise 50%, context len 7
            # 65.8% iter 1200, noise 70%, context len 7
            # 69.69% iter 1200, noise 70%, context len 21
            o = x.mean(dim=0)
            o = w1(o)
            r = w2(o).softmax(dim=0)
            return r

        for i in range(1200):
            if i > 400:
                optim = t.optim.SGD(params, lr=0.003)
            elif i > 800:
                optim = t.optim.SGD(params, lr=0.002)
            elif i > 1000:
                optim = t.optim.SGD(params, lr=0.001)
            loss = _h(fprop_v2, optim)
            print(i, '  Noise:', noise_percent, ' Loss:', loss)
        r = [test(fprop_v2) for i in range(100)]
        return r

    def conv_1d():
        print('~~~~~~~~~~~~~~~~CNN1D test~~~~~~~~~~~~~~~~~~~~')
        w1 = t.nn.Linear(emb_size, 100)
        w2 = t.nn.Linear(100, 2)
        return

    rl = lstm_test()
    rann = ann_test_aggregration()

    return


def continuous_timeseries_test():
    # in this example, both datasets contain the same pattern, only difference is the time difference between eventa
    # this tests LSTM ability to distinguish differences in relative time differences, hence able do away with modelling
    # discrete time steps traditionally, which has many cons such as time step optimization
    # Results: 100% accuracy
    base_events = '0123'
    pattern = base_events * 100
    time_init = t.tensor([0.0], device=tdevice)
    time_lag_a = t.rand([10, 10], device=tdevice)
    time_lag_b = t.rand([10, 10], device=tdevice)

    emb_size = 3
    data_count = 100
    time_len = 10

    noise_percent = 0
    emb = t.rand([10, emb_size], device=tdevice)

    label_a = []
    label_b = []

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    for i in range(data_count):
        np.random.seed(dt.datetime.now().microsecond)
        ii = np.random.randint(0, len(pattern) - time_len)
        pattern_fragment = pattern[ii:ii + time_len]
        frag_emb = to_emb(pattern_fragment)

        temp_a = []
        temp_b = []
        for i in range(len(pattern_fragment)):
            if i == 0:
                temp_a.append(t.cat([frag_emb[i], time_init]))
                temp_b.append(t.cat([frag_emb[i], time_init]))
            else:
                prev = int(pattern_fragment[i - 1])
                cur = int(pattern_fragment[i])

                # emb, then time info
                temp_a.append(t.cat([frag_emb[i], time_lag_a[prev, cur].unsqueeze(0)]))
                temp_b.append(t.cat([frag_emb[i], time_lag_b[prev, cur].unsqueeze(0)]))
        label_a.append(t.stack(temp_a))
        label_b.append(t.stack(temp_b))

    ya = t.tensor([[1.0, 0]] * data_count, device=tdevice)
    yb = t.tensor([[0, 1.0]] * data_count, device=tdevice)

    y = t.cat([ya, yb])

    def _h(fp, op, verbose=True):
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
        if verbose:
            print(' F1', scores['F1'], '   TP:', scores['True positive'], '  TN:', scores['True negative'])
        return loss.data.cpu().item()

    # def test(fp):
    #     a = []
    #     b = []
    #
    #     for i in range(100):
    #         np.random.seed(dt.datetime.now().microsecond)
    #         ii = np.random.randint(0, len(pattern_a) - time_len)
    #         a.append(to_emb(pattern_a[ii:ii + time_len]))
    #         b.append(to_emb(pattern_b[ii:ii + time_len]))
    #
    #     ra = [fp(x)[0].data.item() for x in a]
    #     rb = [fp(x)[0].data.item() for x in b]
    #
    #     correct = len(list(filter(lambda x: x > 0.5, ra))) + len(list(filter(lambda x: x <= 0.5, rb)))
    #     wrong = len(list(filter(lambda x: x < 0.5, ra))) + len(list(filter(lambda x: x >= 0.5, rb)))
    #
    #     return {'correct': correct, 'wrong': wrong}

    def lstm_test():
        print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')
        out_size = 10
        layers = 1
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size + 1, output_size=out_size, batch_size=1,
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
        lstm_optim = t.optim.SGD(params, lr=0.1)

        def lstm_fprop(x):
            out, h = lstm(x.unsqueeze(1), init)
            ii = out.reshape(-1)
            r = last_w2(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop1(x):
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[0].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def lstm_fprop2(x):
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[1].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def one(f, lstm_optim):
            for i in range(1200):
                if i > 400:
                    lstm_optim = t.optim.SGD(params, lr=0.04)
                elif i > 800:
                    lstm_optim = t.optim.SGD(params, lr=0.02)
                elif i > 1200:
                    lstm_optim = t.optim.SGD(params, lr=0.01)

                if i % 10 == 0:
                    loss = _h(f, lstm_optim, verbose=True)
                    print(i, '   Noise:', noise_percent, '  Loss:', loss)
                else:
                    loss = _h(f, lstm_optim, verbose=False)

            # r = [test(f) for i in range(50)]
            return

        r1 = one(lstm_fprop2, lstm_optim)
        #
        # # dynamic length test, longer seq yields better confidence at higher noise levels
        # test_a = lstm_fprop(to_emb(pattern_a[61:78]))
        # test_b = lstm_fprop(to_emb(pattern_b[18:87]))
        return

    lstm_test()
    return


def noise_training_noisy_positives():
    # Both datasets contain 100% noisy data points. But one of them has some data points containing true patterns.
    # Usecase: Manual labelling of positive labels, of which some of them are labelled wrongly (negatives in a positive
    # set)
    # Positive set may contain negatives, negative set only contains negatives
    # Goal: Model's ability to get high True positives when there are wrongly labelled positives

    # Results:
    # 95% valid, 0% noise, 1200 iter, still decreasing
    # 86% acc 99%TP 72% TN, 30% noise, 1200 iter, still decreasing
    # 72% acc 89%TP 56% TN, 50% noise, 1200 iter, still decreasing
    # 59% acc 54%TP 65% TN, 70% noise, 1200 iter
    key_patterns = '1234'
    s = '1234567890'

    data_count = 100
    time_len = 10

    noise_level = 0.7

    label_a = []
    label_b = []

    emb_size = 3

    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    def _h(fp, op, verbose=True):
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
        if verbose:
            print(' F1', scores['F1'], '   TP:', scores['True positive'], '  TN:', scores['True negative'])
        return loss.data.cpu().item()

    for i in range(data_count):
        l = ''.join(np.random.choice(list(s), time_len))
        if np.random.rand() > noise_level:
            # insert key patterns
            start = np.random.randint(0, len(l) - len(key_patterns))
            l = l[0:start] + key_patterns + l[start:-len(key_patterns)]
        label_a.append(l)
        label_b.append(''.join(np.random.choice(list(s), time_len)))  # neg only contains neg AKA all noises
    label_a = [to_emb(x) for x in label_a]
    label_b = [to_emb(x) for x in label_b]

    ya = t.tensor([[1.0, 0]] * data_count, device=tdevice)
    yb = t.tensor([[0, 1.0]] * data_count, device=tdevice)

    y = t.cat([ya, yb])

    def test(fp):  # validation set has no noise
        a = []
        b = []

        for i in range(50):
            l = ''.join(np.random.choice(list(s), time_len))
            # this time, there should be no noises
            start = np.random.randint(0, len(l) - len(key_patterns))
            l = l[0:start] + key_patterns + l[start:-len(key_patterns)]

            a.append(l)
            b.append(''.join(np.random.choice(list(s), time_len)))
        a = [to_emb(x) for x in a]
        b = [to_emb(x) for x in b]

        ra = [fp(x)[0].data.item() for x in a]
        rb = [fp(x)[0].data.item() for x in b]

        tp = len(list(filter(lambda x: x > 0.5, ra)))
        tn = len(list(filter(lambda x: x <= 0.5, rb)))
        fp = len(list(filter(lambda x: x < 0.5, ra)))
        fn = len(list(filter(lambda x: x >= 0.5, rb)))

        correct = tp + tn
        wrong = fp + fn

        return {'correct': correct, 'wrong': wrong, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    def lstm_test():
        print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')
        out_size = 10
        layers = 1
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=out_size, batch_size=1,
                                                       num_of_layers=layers,
                                                       device=tdevice)
        last_w = t.nn.Linear(out_size * layers, 2)
        last_w.to(tdevice)

        last_w2 = t.nn.Linear(out_size, 2)
        last_w2.to(tdevice)

        params = [
            {'params': list(lstm.parameters())},
            {'params': list(last_w.parameters())},
            {'params': list(last_w2.parameters())}
        ]
        lstm_optim = t.optim.SGD(params, lr=0.25)

        def lstm_fprop2(x):
            # 95% valid, 0% noise, 1200 iter, still decreasing
            # 86% acc 99%TP 72% TN, 30% noise, 1200 iter, still decreasing
            # 72% acc 89%TP 56% TN, 50% noise, 1200 iter, still decreasing
            # 59% acc 54%TP 65% TN, 70% noise, 1200 iter
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[1].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def one(f, lstm_optim):
            for i in range(1200):
                if i % 10 == 0:
                    loss = _h(f, lstm_optim, verbose=True)
                    print(i, '  Noise:', noise_level, ' Loss:', loss)
                else:
                    loss = _h(f, lstm_optim, verbose=False)

            r = [test(f) for i in range(50)]
            return r

        r1 = one(lstm_fprop2, lstm_optim)
        return

    lstm_test()
    return


# anomaly detection time series
def noise_training_noisy_negatives_autoencoder_ann_vs_lstm():
    # Anomaly detection against time series of random length
    # Conclusion: Mean aggregration ANN is still best with 1% noise detection rate, but require longer time length

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LSTM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Finished training in  2:04:46.741849 <-----fking 2 hours
    # Validating against noise at  0
    #      Good mean: 0.005826742388308048      Good max: 0.1185045912861824    Good min: 3.240163692908027e-09
    #      Bad mean: 0.0055285487323999405      Bad max: 0.12678655982017517    Bad min: 3.753997113165042e-10
    # Validating against noise at  0.005
    #      Good mean: 0.0058425371535122395      Good max: 0.12748640775680542    Good min: 1.5335253067405574e-09
    #      Bad mean: 0.005722632631659508      Bad max: 0.1381041705608368    Bad min: 9.967582315084655e-11
    # Validating against noise at  0.01
    #      Good mean: 0.005268234293907881      Good max: 0.11359921842813492    Good min: 2.7825741710785223e-11
    #      Bad mean: 0.005310694221407175      Bad max: 0.148960143327713    Bad min: 8.530065542800003e-12
    # Validating against noise at  0.1
    #      Good mean: 0.005168387666344643      Good max: 0.11446716636419296    Good min: 2.0784624021885634e-12
    #      Bad mean: 0.005754341837018728      Bad max: 0.12355780601501465    Bad min: 2.482977379258955e-09
    # Validating against noise at  0.3
    #      Good mean: 0.00501678604632616      Good max: 0.09210450202226639    Good min: 8.995743039363902e-11
    #      Bad mean: 0.005463255103677511      Bad max: 0.15041233599185944    Bad min: 1.1910152863947587e-09
    # Validating against noise at  0.5
    #      Good mean: 0.005854926537722349      Good max: 0.12954002618789673    Good min: 9.596149430635137e-10
    #      Bad mean: 0.005779569502919912      Bad max: 0.16743168234825134    Bad min: 3.525180147789797e-12
    # Validating against noise at  0.7
    #      Good mean: 0.005416967440396547      Good max: 0.12901702523231506    Good min: 2.0806112388527254e-10
    #      Bad mean: 0.005212957505136728      Bad max: 0.10033897310495377    Bad min: 4.26952362353461e-11
    # 0  Loss: 0.4398900270462036
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ANN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Finished training in  0:04:45.373816
    # Validating against noise at  0
    #      Good mean: 0.0003713587357196957      Good max: 0.008444118313491344    Good min: 1.991745657292654e-11
    #      Bad mean: 0.00036989181535318494      Bad max: 0.008291475474834442    Bad min: 1.7905765758996495e-10
    # Validating against noise at  0.005
    #      Good mean: 0.0003746516886167228      Good max: 0.00847811158746481    Good min: 2.877698079828406e-13
    #      Bad mean: 0.00032785814255476      Bad max: 0.009606100618839264    Bad min: 1.6961454463171322e-10
    # Validating against noise at  0.01
    #      Good mean: 0.00032482456299476326      Good max: 0.009713081642985344    Good min: 5.841735983835861e-10
    #      Bad mean: 0.00035888413549400866      Bad max: 0.007528867106884718    Bad min: 2.2920687570149312e-10
    # Validating against noise at  0.1
    #      Good mean: 0.0003452543169260025      Good max: 0.00901718158274889    Good min: 1.82262205328243e-10
    #      Bad mean: 0.00045542052248492837      Bad max: 0.01522256713360548    Bad min: 2.551701072661672e-10
    # Validating against noise at  0.3
    #      Good mean: 0.00032740726601332426      Good max: 0.004628137685358524    Good min: 2.3309354446610087e-11
    #      Bad mean: 0.0005330688436515629      Bad max: 0.010963350534439087    Bad min: 1.2423484463397472e-10
    # Validating against noise at  0.5
    #      Good mean: 0.000376830343157053      Good max: 0.013526303693652153    Good min: 2.3540138727184967e-09
    #      Bad mean: 0.0007937237969599664      Bad max: 0.01807171106338501    Bad min: 1.1201075622579992e-09
    # Validating against noise at  0.7
    #      Good mean: 0.00030086591141298413      Good max: 0.009397735819220543    Good min: 1.566746732351021e-10
    #      Bad mean: 0.0013094970490783453      Bad max: 0.025551117956638336    Bad min: 6.687895237611485e-10

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ANN only sensivity optimization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Conclusion: Sensivity is tested at best 1% noise detection, with longer time length.

    # data_count = 600
    # max_time_len = 200
    # min_time_len = 40
    # Finished training in  0:10:25.408214 (2000 iter)
    # bad:good
    # 0% 0.000497:000446
    # 0.5% 0.0004544:0.0004877
    # 1% 0004683895385824144:00049206503899768
    # 3% 0004677371180150658:0005293051362968981
    # 5% 00046205378021113575:0005201962776482105
    # 7% 00047103254473768175:0004024988447781652
    # 9% 0005560332210734487:000497371016535908 <------
    #
    # data_count = 600
    # max_time_len = 120
    # min_time_len = 20
    # Finished training in  0:10:25.408214 (2000 iter)
    # noise bad:good
    # 0% 0.0007058838964439929:0008016185602173209
    # 0.5% 0.0008811770239844918:0.0007632211199961603
    # 1% 0007124589756131172:0008798842900432646
    # 3% 0008144420571625233:0007756570121273398
    # 5% 0007736627594567835:0008676669676788151
    # 7% 000924094405490905:0007216933881863952 <------
    # 9% 0007887544925324619:0006597673054784536
    # 10% 0008224970079027116:000814529019407928
    # 30% 0010200951946899295:0008001350797712803 <------
    # 40% 0012596054002642632:0006964309141039848 <------
    # 50% 001827389351092279:000777921115513891 <------
    # 60% 001981183886528015:0007596184150315821 <------
    # 70% 002628359477967024:0007228725007735193 <------
    # 80% 0031278354581445456:0008283860515803099 <------
    # 90% 0035632161889225245:0007503657834604383 <------
    # 100% 00418577715754509:0008262812625616789 <------
    #
    # data_count = 600
    # max_time_len = 300
    # min_time_len = 100
    # Finished training in  ? (2000 iter)
    # noise bad:good
    # 0% 00027990268426947296:0002625519991852343
    # 0.5% 00029652329976670444:00027735097683034837   <---
    # 1% 0002917275414802134:0002783473173622042  <--
    # 3% 00026276710559614:00028564181411638856
    # 5% 0002880408719647676:0002940943813882768  <--
    # 7% 00032785756047815084:00027572002727538347 <------
    # 9% 0003384773153811693:00027624567155726254 <------
    # 10% 0003319961251690984:0002704804646782577 <------
    # 20% 00042327132541686296:0002581196022219956 <------
    # 30% 0005606891354545951:0002683283237274736 <------
    # 40% 0009246580884791911:00024527322966605425 <------
    # 50% 0011530306655913591:0002951421483885497 <------
    # 60% 0014943181304261088:00028733719955198467 <------
    # 70% 0020007132552564144:00029301270842552185 <------
    # 80% 0028081010095775127:0002490574843250215 <------
    # 90% 0034917471930384636:00027867924654856324 <------
    # 100% 004061240237206221:00027311075245961547 <------
    #
    # data_count = 600
    # max_time_len = 500
    # min_time_len = 200
    # Finished training in  0:11:17.686206 (2000 iter)
    # noise bad:good
    # 0% 000127850376884453:00013236295490060002
    # 0.5% 00013791536912322044:00012105504720238969 <-
    # 1% 00014659545558970422:00011824637476820499 <---- Stable detection rate at 2 of 200 events
    # 3% 0001504896063124761:0001374738203594461  <------
    # 5% 0001629124308237806:00013523899542633444 <------
    # 7% 00017010104784276336:00013665227743331343 <------
    # 9% 0001689684868324548:00012298367801122367 <------
    # 10% 00021823054703418165:00013824265624862164 <------
    # 20% 00037455977872014046:00014137427206151187 <------
    # 30% 0005791793810203671:00011004442058037966 <------
    # 40% 0009000598802231252:00011691282270476222 <------
    # 50% 0012664379319176078:00012960021558683366 <------
    # 60% 0015720038209110498:00016047849203459918 <------
    # 70% 0019082155777141452:00013320606376510113 <------
    # 80% 002324917819350958:00014366924006026238 <------
    # 90% 0026481079403311014:00013394902634900063 <------
    # 100% 003103169146925211:00013215825310908258 <------
    #
    # ... Higher also >~1-3% onwards for stable differentiation

    data_count = 600
    max_time_len = 800
    min_time_len = 300

    tanh_scale = 0.03

    data = []

    emb_size = 20

    ap = np.power(np.random.rand(10, 10), 3)
    at = 10 * np.random.rand(10, 10)

    bp = np.power(np.random.rand(10, 10), 3)
    bt = 10 * np.random.rand(10, 10)

    s = list('0123456789')

    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(x):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in x[0]]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        time = t.tensor(x[1], dtype=t.float32, device=tdevice)
        f = []
        for i in range(len(x[0])):
            f.append(t.cat([a[i], time[i].unsqueeze(0)]))
        f = t.stack(f)
        return f

    def _h(fp, op, verbose=True):
        out = [fp(x) for x in data]
        _y = t.stack([x[0] for x in out])
        y = t.stack([x[1] for x in out])

        loss = t.nn.MSELoss()(_y, y)

        loss.backward()
        op.step()
        op.zero_grad()

        return loss.data.cpu().item()

    def gen_data(prop, time):
        xs = [np.random.choice(s)]
        xt = [0]
        curr = 0
        while curr < max_time_len:
            np.random.seed(dt.datetime.now().microsecond)
            if curr > len(xt) - 1:
                return gen_data(prop, time)
            ct = xt[curr]
            cl = int(xs[curr])

            xpp = prop[cl]
            xpt = time[cl]
            ch = np.random.rand(10)

            for ii in range(10):
                if ch[ii] < xpp[ii]:
                    nextl = str(ii)
                    nextt = ct + xpt[ii]
                    for ttt in range(curr, len(xt)):
                        tt = xt[ttt]
                        if nextt < tt:
                            xt = xt[:ttt] + [nextt] + xt[ttt:]
                            xs = xs[:ttt] + [nextl] + xs[ttt:]
                            break
                        elif ttt == len(xt) - 1:
                            xt.append(nextt)
                            xs.append(nextl)
                            break

            curr += 1
        end = np.random.randint(min_time_len, max_time_len)
        last_time = xt[:end][-1]
        xt = np.tanh(tanh_scale * (np.asarray(xt[:end]) - last_time))
        return [''.join(xs[:end]), xt]

    def gen_data2(prop, time, prop2, time2, noise):
        xs = [np.random.choice(s)]
        xt = [0]
        curr = 0
        while curr < max_time_len:
            np.random.seed(dt.datetime.now().microsecond)
            if curr > len(xt) - 1:
                return gen_data(prop, time)
            ct = xt[curr]
            cl = int(xs[curr])

            if np.random.rand() < noise:
                xpp = prop2[cl]
                xpt = time2[cl]
            else:
                xpp = prop[cl]
                xpt = time[cl]
            ch = np.random.rand(10)

            for ii in range(10):
                if ch[ii] < xpp[ii]:
                    nextl = str(ii)
                    nextt = ct + xpt[ii]
                    for ttt in range(curr, len(xt)):
                        tt = xt[ttt]
                        if nextt < tt:
                            xt = xt[:ttt] + [nextt] + xt[ttt:]
                            xs = xs[:ttt] + [nextl] + xs[ttt:]
                            break
                        elif ttt == len(xt) - 1:
                            xt.append(nextt)
                            xs.append(nextl)
                            break

            curr += 1
        end = np.random.randint(min_time_len, max_time_len)
        last_time = xt[:end][-1]
        xt = np.tanh(tanh_scale * (np.asarray(xt[:end]) - last_time))
        return [''.join(xs[:end]), xt]

    for i in range(data_count):
        d = gen_data(ap, at)
        data.append(d)

    data = [to_emb(x) for x in data]

    def test(fp):  # validation set has no noise
        test_count = 100

        for noise in [0, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            print('Validating against noise at ', noise)
            a = [gen_data(ap, at) for x in range(test_count)]
            b = [gen_data2(ap, at, bp, bt, noise) for x in range(test_count)]

            def _f(fp, d):
                _y, y = fp(to_emb(d))
                return t.pow(t.sub(_y, y), 2)

            ra = t.stack([_f(fp, x) for x in a])
            rb = t.stack([_f(fp, x) for x in b])

            _ = lambda x: x.data.cpu().item()
            print('     Good mean:', _(t.mean(ra)), '     Good max:', _(t.max(ra)), '   Good min:', _(t.min(ra)))
            print('     Bad mean:', _(t.mean(rb)), '     Bad max:', _(t.max(rb)), '   Bad min:', _(t.min(rb)))
        return

    def ann_test(xx):
        neck = 3
        w1 = t.nn.Linear(emb_size + 1, neck)
        w1.to(tdevice)

        w2 = t.nn.Linear(neck, emb_size + 1)
        w2.to(tdevice)

        params = [
            {'params': list(w1.parameters())},
            {'params': list(w2.parameters())}
        ]
        optim = t.optim.SGD(params, lr=0.02)

        def fprop(x):
            o1 = x.mean(dim=0)
            o = w1(o1).tanh()
            o = w2(o)
            return o, o1

        time_a = dt.datetime.now()
        for i in range(xx):
            if i % 100 == 0:
                loss = _h(fprop, optim, verbose=True)
                print(i, ' Loss:', loss)
            else:
                loss = _h(fprop, optim, verbose=False)
        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))
        r = test(fprop)
        return

    def lstm_test(xx):
        print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')
        out_size = 10
        layers = 2
        neck = 3
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size + 1, output_size=out_size, batch_size=1,
                                                       num_of_layers=layers,
                                                       device=tdevice)
        neck_w = t.nn.Linear(out_size * layers, neck)
        neck_w.to(tdevice)

        last_w = t.nn.Linear(neck, out_size * layers)
        last_w.to(tdevice)

        params = [
            {'params': list(lstm.parameters())},  # <---cannot train this
            {'params': list(neck_w.parameters())},
            {'params': list(last_w.parameters())}
        ]
        lstm_optim = t.optim.SGD(params, lr=0.08)

        def fprop(x):
            out, h = lstm(x.unsqueeze(1), init)
            h1 = h[1].reshape(-1)
            ii = neck_w(h1)
            r = last_w(ii)
            return r, h1

        time_a = dt.datetime.now()
        for i in range(xx):
            if i % 100 == 0:
                loss = _h(fprop, lstm_optim, verbose=True)
                print(i, ' Loss:', loss)
            else:
                loss = _h(fprop, lstm_optim, verbose=False)
        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))
        r = test(fprop)
        return

    # lstm_test(2000)
    ann_test(2000)
    return
