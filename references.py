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
