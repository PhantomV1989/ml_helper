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

Autoencoder does not always have a folded distribution despite training every value for minimum! This ability is not controlled by AE layer count as well.
'''

tdevice = t.device('cpu')


def pytorch_lstm_and_lstmcell_test():
    emb_in = 5
    emb_out = 2
    dlen = 13

    test = t.rand(size=[dlen, emb_in])

    lstm, init1 = ml_helper.TorchHelper.create_lstm(input_size=emb_in, output_size=emb_out, batch_size=1,
                                                    num_of_layers=1, device=tdevice)
    lstm_seq = lstm(test.unsqueeze(1), init1)
    lstm_unit = lstm(test[0:1].unsqueeze(1), init1)

    if (lstm_seq[0][0] - lstm_unit[0][0]).mean().cpu().data.numpy() == 0:
        print('LSTM outputs for 1st of seq and only 1st seq is same')
    else:
        print('LSTM inconsistent')

    lstmc, init2 = ml_helper.TorchHelper.create_lstm_cell(input_size=emb_in, output_size=2, batch_size=1,
                                                          device=tdevice)
    lstmc_seq = lstmc(test)
    lstmc_unit = lstmc(test[0:1], init2)
    if abs((lstmc_seq[0][0] - lstmc_unit[0][0]).mean().cpu().data.numpy()) < 1E-7:  # for some reason, error is 1E-9
        print('LSTM cell output for 1st of seq and only 1st seq is same')
    else:
        print('LSTM inconsistent')
    return


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
    # Conclusion: Mean aggregration ANN is still best with 1% noise detection rate, but require longer time length and
    # tanh

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
    # Conclusion: Sensivity is tested at best 1% noise detection, with longer time length and tanh activation

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


# tests untrained lstm's sensivity to very old events
def untrained_lstm_sensivity_test():
    s = list('123456789')
    emb_size = 4

    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a.unsqueeze(1)

    out_size = 10
    layers = 2
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=out_size, batch_size=1,
                                                   num_of_layers=layers,
                                                   device=tdevice)

    def get_mse(s1, s2):
        s1 = to_emb(s1)
        s2 = to_emb(s2)
        o1, h1 = lstm(s1, init)
        o2, h2 = lstm(s2, init)

        l = t.nn.MSELoss()(h1[1], h2[1])
        return l.data.cpu().item()

    get_mse('123', '4564')
    l = get_mse('1239999999999999999999999999999999999999', '9999999999999999999999999999999999999999')  # 5E-16
    l = get_mse('846', '397')  # 0.0054
    return


# ann cannot detect sequences as expected
def sequence_detection_ann_aggregation_test():
    seq_a = '1234567890'
    seq_b = '0987654321'

    emb_size = 5
    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(x):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in x]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        time = t.linspace(0, 1, len(x), device=tdevice)
        f = []
        for i in range(len(x)):
            f.append(t.cat([a[i], time[i].unsqueeze(0)]))
        f = t.stack(f)
        return f

    def _tself(fp, op):
        y = to_emb(seq_a)
        _y = fp(y)
        loss = t.nn.MSELoss()(_y, y)

        loss.backward()
        op.step()
        op.zero_grad()

        return loss.data.cpu().item()

    def _tclass(fp, op):
        a = fp(to_emb(seq_a))
        b = fp(to_emb(seq_b))
        _y = t.stack([a, b])
        y = t.tensor(np.asarray([[1, 0.0], [0.0, 1.0]]), dtype=t.float32, device=tdevice)
        loss = t.nn.BCELoss()(_y, y)

        loss.backward()
        op.step()
        op.zero_grad()

        return loss.data.cpu().item()

    def ann_ac_test(xx):
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
            return o

        time_a = dt.datetime.now()
        for i in range(xx):
            loss = _tself(fprop, optim)
            if i % 10 == 0:
                print(i, ' Loss:', loss)

        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))

        def tt(x):
            y = to_emb(x)
            _y = fprop(y)
            loss = t.nn.MSELoss()(_y, y)
            return loss.data.cpu().item()

        print('Loss seq a:', str(tt(seq_a)))
        print('Loss seq b:', str(tt(seq_b)))

        return

    def ann_classifier_test(xx):
        w1 = t.nn.Linear(emb_size + 1, 5)
        w1.to(tdevice)

        w2 = t.nn.Linear(5, 2)
        w2.to(tdevice)

        params = [
            {'params': list(w1.parameters())},
            {'params': list(w2.parameters())}
        ]
        optim = t.optim.SGD(params, lr=0.02)

        def fprop(x):
            o1 = x.mean(dim=0)
            o = w1(o1).tanh()
            o = w2(o).softmax(dim=0)
            return o

        time_a = dt.datetime.now()
        for i in range(xx):
            loss = _tclass(fprop, optim)
            if i % 10 == 0:
                print(i, ' Loss:', loss)

        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))

        def tt(x):
            y = to_emb(x)
            _y = fprop(y)
            loss = t.nn.MSELoss()(_y, y)
            return loss.data.cpu().item()

        return

    # ann_ac_test(100)
    ann_classifier_test(1000)

    return


# this tests if different models converges if trained on same data,
# for categorially different models even if they see the same data
# conclusion: different models are DIFFERENT even if trained on same data
def models_similarity_test_on_same_data():
    ''' Results
    i: 0   loss: 0.2849670350551605
    i: 1000   loss: 0.06827231496572495
    i: 2000   loss: 0.06409560889005661
    i: 3000   loss: 0.06236245855689049
    i: 4000   loss: 0.06132930889725685
    i: 5000   loss: 0.060513466596603394
    i: 6000   loss: 0.0598052553832531
    i: 7000   loss: 0.05918245390057564
    i: 8000   loss: 0.05864206328988075
    i: 9000   loss: 0.05818251892924309
    i: 0   loss: 0.36589521169662476    |m1-m2|: 0.2643558979034424
    i: 1000   loss: 0.06521538645029068    |m1-m2|: 0.2612735629081726
    i: 2000   loss: 0.06165115535259247    |m1-m2|: 0.2723885178565979
    i: 3000   loss: 0.06037070229649544    |m1-m2|: 0.2795097827911377
    i: 4000   loss: 0.05963187292218208    |m1-m2|: 0.282612681388855
    i: 5000   loss: 0.05906539037823677    |m1-m2|: 0.2847711443901062
    i: 6000   loss: 0.05859127268195152    |m1-m2|: 0.2857397496700287
    i: 7000   loss: 0.0581863597035408    |m1-m2|: 0.2859709560871124
    i: 8000   loss: 0.057838816195726395    |m1-m2|: 0.2861637473106384
    i: 9000   loss: 0.05753947049379349    |m1-m2|: 0.2861420810222626
    '''
    feat = 15
    epoch = 10000
    y = t.rand([100, feat], device=tdevice)

    def get_ae():
        w1 = t.nn.Linear(feat, 3)
        w1.to(tdevice)
        w2 = t.nn.Linear(3, feat)
        w2.to(tdevice)
        params = [
            {'params': list(w1.parameters())},
            {'params': list(w2.parameters())}
        ]
        lstm_optim = t.optim.SGD(params, lr=0.06)
        return [w1, w2, lstm_optim]

    def _t(m, d):
        o = m[0](d)
        o = m[1](o).tanh()

        loss = t.nn.MSELoss()(o, d)
        loss.backward()
        m[2].step()
        m[2].zero_grad()
        return loss.data.cpu().item()

    m1 = get_ae()
    for i in range(epoch):
        l = _t(m1, y)
        if i % 1000 == 0:
            print('i:', i, '  loss:', l)

    m2 = get_ae()
    for i in range(epoch):
        l = _t(m2, y)
        d = m2[0].weight.sub(m1[0].weight).abs().mean().data.cpu().item()
        if i % 1000 == 0:
            print('i:', i, '  loss:', l, '   |m1-m2|:', d)
    return


# tests for current mean-aggregated with context vs current concat mean-aggreg context
def anomaly_detection_context_focus_concat_test():
    '''
    Conclusion, DO NOT MEAN AGGREGRATE EVERYTHING! Only the context can be aggregated, but the subject of interest
    should not be mean-aggregated together with the context!
    '''
    # focus concat mean-agg context
    # Finished training in  0:10:22.953199
    # [1227  201  127   97   82   71   65   58   47   25]
    # Validating against noise at  0
    #      Good mean: 0.3243067777586217      Good max: 7802.138167432821    Good min: 551.3168886407906
    #      Bad mean: 7.824828706017395      Bad max: 7813.743710886022    Bad min: 551.3169860512105
    # Validating against noise at  0.005
    #      Good mean: 5.365242796623158      Good max: 7607.618711838109    Good min: 551.3169891065706
    #      Bad mean: 11.528002904637805      Bad max: 7913.228445697616    Bad min: 551.3169581264842
    # Validating against noise at  0.01
    #      Good mean: 9.372809305382926      Good max: 7441.085158725415    Good min: 551.3169891439716
    #      Bad mean: 0.6182216186353824      Bad max: 7417.02423264549    Bad min: 551.316987235395
    # Validating against noise at  0.03
    #      Good mean: 0.899010518852587      Good max: 7802.138167432821    Good min: 551.3169065602049
    #      Bad mean: 18.460085745299182      Bad max: 7813.743710886022    Bad min: 551.3169865086263
    # Validating against noise at  0.05
    #      Good mean: 17.328082503719617      Good max: 7779.385178765844    Good min: 551.3169865086263
    #      Bad mean: 23.663095116819488      Bad max: 9904.13911044557    Bad min: 551.3169884566508
    # Validating against noise at  0.07
    #      Good mean: 3.587099846435576      Good max: 7465.661809125709    Good min: 551.3169194224749
    #      Bad mean: 35.377378693721866      Bad max: 9875.805705233011    Bad min: 551.3169858857685
    # Validating against noise at  0.09
    #      Good mean: 11.443964869927406      Good max: 7730.669021266156    Good min: 551.3169521499113
    #      Bad mean: 31.4900920768926      Bad max: 10072.095271051476    Bad min: 551.3169890520131
    # Validating against noise at  0.1
    #      Good mean: 12.309993188663796      Good max: 7802.138167432821    Good min: 551.3169892181317
    #      Bad mean: 14.52783332288461      Bad max: 9989.910002155679    Bad min: 551.3167429338652
    # Validating against noise at  0.2
    #      Good mean: 16.89912472568052      Good max: 7551.5409680181965    Good min: 551.3169766954886
    #      Bad mean: 62.506129601795024      Bad max: 10174.90834859882    Bad min: 551.3168380585937
    # Validating against noise at  0.3
    #      Good mean: 18.32264172143083      Good max: 7490.400860299574    Good min: 551.3169890413445
    #      Bad mean: 61.93139675661975      Bad max: 10216.507394061078    Bad min: 551.3169843560935
    # Validating against noise at  0.4
    #      Good mean: 34.58468538380335      Good max: 7607.566324491796    Good min: 551.3169888986102
    #      Bad mean: 108.25003280025379      Bad max: 10256.89687390513    Bad min: 551.3169625556358
    # Validating against noise at  0.5
    #      Good mean: 13.950954051714566      Good max: 7563.455596814644    Good min: 551.316806673151
    #      Bad mean: 193.8886464609211      Bad max: 10330.064534255587    Bad min: 551.3169866709978
    # Validating against noise at  0.6
    #      Good mean: 35.107401959702      Good max: 7670.849656756246    Good min: 551.3169762792209
    #      Bad mean: 197.86862957664306      Bad max: 10509.087230729296    Bad min: 551.3168591818096
    # Validating against noise at  0.7
    #      Good mean: 4.692974899041569      Good max: 7528.74723363742    Good min: 551.316967468415
    #      Bad mean: 239.72961177046528      Bad max: 10843.007668617962    Bad min: 551.3169874967334
    # Validating against noise at  0.8
    #      Good mean: 17.17670490093608      Good max: 7809.510813303934    Good min: 551.3169634514373
    #      Bad mean: 267.73777887466593      Bad max: 10488.476884526519    Bad min: 551.3169877901123
    # Validating against noise at  0.9
    #      Good mean: 22.55179215305387      Good max: 7867.45878538722    Good min: 551.3169681239899
    #      Bad mean: 305.6593781461588      Bad max: 10604.498558324243    Bad min: 551.3169829422123
    # Validating against noise at  1
    #      Good mean: 13.098568271080515      Good max: 7874.247603387756    Good min: 551.3169845020891
    #      Bad mean: 360.15851659717373      Bad max: 10565.437388750037    Bad min: 551.3168308042646
    #
    # focus being mean-agg with context
    # Finished training in  0:09:07.148683
    # [1619  108   65   49   38   33   28   24   20   16]
    # Validating against noise at  0
    #      Good mean: 0.45261300851675834      Good max: 3.233966311950346    Good min: 0.5095926808743774
    #      Bad mean: 0.4565816772720046      Bad max: 2.9986092847208656    Bad min: 0.5095926792483044
    # Validating against noise at  0.005
    #      Good mean: 0.4368865626502712      Good max: 5.314683158413496    Good min: 0.5095926790182436
    #      Bad mean: 0.41507603381643865      Bad max: 4.483771411423884    Bad min: 0.5095925840511724
    # Validating against noise at  0.01
    #      Good mean: 0.45881354100695804      Good max: 1.8344153028320291    Good min: 0.5095926697370133
    #      Bad mean: 0.4621457432786258      Bad max: 1.0452505069274853    Bad min: 0.5095926844867767
    # Validating against noise at  0.03
    #      Good mean: 0.4336032113078854      Good max: 5.855115803914694    Good min: 0.5095926887331488
    #      Bad mean: 0.43687897441611606      Bad max: 5.314683158413496    Bad min: 0.5095926843668906
    # Validating against noise at  0.05
    #      Good mean: 0.43155070542946955      Good max: 4.512600352420551    Good min: 0.5095926819454436
    #      Bad mean: 0.42790714034423405      Bad max: 5.855115803914694    Bad min: 0.5095926842802373
    # Validating against noise at  0.07
    #      Good mean: 0.39854189577146815      Good max: 5.855115803914694    Good min: 0.5095926817614859
    #      Bad mean: 0.4457864779177134      Bad max: 2.36096363770555    Bad min: 0.5095926774140933
    # Validating against noise at  0.09
    #      Good mean: 0.4534239910106392      Good max: 2.319882239848848    Good min: 0.5095926886834018
    #      Bad mean: 0.4271638768213182      Bad max: 2.9986092847208656    Bad min: 0.5095926876240537
    # Validating against noise at  0.1
    #      Good mean: 0.43412781598479394      Good max: 5.855115803914694    Good min: 0.5095926887026713
    #      Bad mean: 0.4405837625515386      Bad max: 4.09579910551878    Bad min: 0.5095926710440251
    # Validating against noise at  0.2
    #      Good mean: 0.43044145896570357      Good max: 5.314683158413496    Good min: 0.5095926612639979
    #      Bad mean: 0.44319089166183745      Bad max: 2.479944581207054    Bad min: 0.5095926860445985
    # Validating against noise at  0.3
    #      Good mean: 0.4401460391431266      Good max: 5.855115803914694    Good min: 0.5095924952417963
    #      Bad mean: 0.43244556421545416      Bad max: 6.6991679827468396    Bad min: 0.5095926247291193
    # Validating against noise at  0.4
    #      Good mean: 0.40673646639466043      Good max: 5.314683158413496    Good min: 0.5095926691938779
    #      Bad mean: 0.40456782754134857      Bad max: 5.855115803914694    Bad min: 0.509592667556529
    # Validating against noise at  0.5
    #      Good mean: 0.4568671634045874      Good max: 5.855115803914694    Good min: 0.5095926882756292
    #      Bad mean: 0.4363014482706873      Bad max: 1.8640193672348502    Bad min: 0.5095926855055959
    # Validating against noise at  0.6
    #      Good mean: 0.4237383107356418      Good max: 2.9986092847208656    Good min: 0.5095926860445985
    #      Bad mean: 0.4096642585867101      Bad max: 5.845047331796315    Bad min: 0.5095924540139108
    # Validating against noise at  0.7
    #      Good mean: 0.4232196357599238      Good max: 5.063177370215368    Good min: 0.5095926706252316
    #      Bad mean: 0.40771053350496944      Bad max: 4.278039706353073    Bad min: 0.5095922725166427
    # Validating against noise at  0.8
    #      Good mean: 0.45702973976091354      Good max: 2.9986092847208656    Good min: 0.5095925617974486
    #      Bad mean: 0.41587863893547966      Bad max: 2.1530282595017605    Bad min: 0.5095926844356011
    # Validating against noise at  0.9
    #      Good mean: 0.46134392284184766      Good max: 3.3938728582120086    Good min: 0.5095926880718961
    #      Bad mean: 0.38406579204812874      Bad max: 10.155725308397205    Bad min: 0.5095926844867767
    # Validating against noise at  1
    #      Good mean: 0.45299363291528494      Good max: 5.855115803914694    Good min: 0.5095926826159485
    #      Bad mean: 0.3964183137303607      Bad max: 5.795901112311945    Bad min: 0.5095926879079945

    data_count = 1000
    max_time_len = 80
    min_time_len = 1

    tanh_scale = 0.03

    data = []

    emb_size = 20

    # pattern A
    ap = np.power(np.random.rand(10, 10), 3)
    at = 10 * np.random.rand(10, 10)
    af = [0, 1, 2, 3, 4]

    bp = np.power(np.random.rand(10, 10), 3)
    bt = 10 * np.random.rand(10, 10)
    bf = [5, 6, 7, 8, 9]

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
        focus = str(np.random.choice(af))
        d = xs[:end - 1] + [focus]
        return [''.join(d), xt]

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
        focus = str(np.random.choice(bf)) if np.random.rand() < noise else str(np.random.choice(af))
        d = xs[:end - 1] + [focus]
        return [''.join(d), xt]

    for i in range(data_count):
        d = gen_data(ap, at)
        data.append(d)

    data = [to_emb(x) for x in data]

    def test(fp, mean, std):  # validation set has no noise
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

            _ = lambda x: np.abs((x.data.cpu().item() - mean) / std)
            print('     Good mean:', _(t.mean(ra)), '     Good max:', _(t.max(ra)), '   Good min:', _(t.min(ra)))
            print('     Bad mean:', _(t.mean(rb)), '     Bad max:', _(t.max(rb)), '   Bad min:', _(t.min(rb)))
        return

    def ann_focus_concat_test(xx):
        neck = 3
        w1 = t.nn.Linear((emb_size + 1) * 2, neck)
        w1.to(tdevice)

        w2 = t.nn.Linear(neck, (emb_size + 1) * 2)
        w2.to(tdevice)

        params = [
            {'params': list(w1.parameters())},
            {'params': list(w2.parameters())}
        ]
        optim = t.optim.SGD(params, lr=0.02)

        def fprop(x):
            o1 = x.mean(dim=0)
            focus = x[-1]
            o1 = t.cat([o1, focus])
            o = w1(o1).tanh()
            o = w2(o)
            return o, o1

        losses = []
        time_a = dt.datetime.now()
        for i in range(xx):
            loss = _h(fprop, optim, verbose=True)
            if i % 100 == 0:
                print(i, ' Loss:', loss)
            losses.append(loss)
        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))
        print(np.histogram(losses)[0])
        losses = np.asarray(losses[-100:])
        r = test(fprop, losses.mean(), losses.std())
        return

    def ann_mean_agg_test(xx):
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
        losses = []
        for i in range(xx):
            loss = _h(fprop, optim, verbose=False)
            losses.append(loss)
            if i % 100 == 0:
                print(i, ' Loss:', loss)

        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))
        print(np.histogram(losses)[0])
        losses = np.asarray(losses)
        r = test(fprop, losses.mean(), losses.std())
        return

    ann_focus_concat_test(2000)
    ann_mean_agg_test(2000)
    return


# use neck distribution, train for separation of nonlinear differences, use kmeans max separation for auto clustering
def neck_vs_loss_ae_clustering():
    '''
    Neck clustering:
        - only works after training, 
        - aggregrates data if they are linear transform(p1,p3) of each other, separates them of they are not
        - lower no. of layers, more clustering

    Loss clustering:
        - works even without training, but can only work on clusters with linear differences
    '''
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    '''
    Neck clustering:
        - only works after training, 
        - aggregrates data if they are linear transform of each other, separates them of they are not
        - lower no. of layers, more clustering

    Loss clustering:
        - works even without training, but can only work on clusters with linear differences
    '''

    tdevice = 'cpu'
    feat = 100

    p1 = lambda: np.concatenate([np.random.randn(70) + 1, np.random.randn(30) - 2])
    p2 = lambda: np.concatenate([np.random.randn(30) - 2, np.random.randn(70) + 1])  # p2 is nonlinear transform of p1
    p3 = lambda: np.concatenate([np.random.randn(50) - 1, np.random.randn(50) - 4])  # p3 is ~linear transform of p1
    p4 = lambda: np.sin([i / 4 for i in range(100)]) + np.random.rand(
        100) * 0.1  # p4 is another nonlinear transformation of p1

    def cp(p, c):
        px = [p() for i in range(c)]
        tx = t.tensor(px, dtype=t.float32)
        return tx

    t1 = cp(p1, 100)
    t2 = cp(p2, 30)
    t3 = cp(p3, 30)
    t4 = cp(p4, 40)
    data = t.cat([t3, t2, t4], dim=0)  # t1,t2,t4 should give 3 different clusters most of the time
    print(data.shape)

    def _h(fp, op):
        out = [fp(x) for x in data]
        _y = t.stack([x[0] for x in out])
        y = t.stack([x[1] for x in out])

        loss = t.nn.MSELoss()(_y, y)

        loss.backward()
        op.step()
        op.zero_grad()

        return loss.data.cpu().item()

    def get_clusters(l, separation=0.05):
        r2 = []
        cluster_cnt = 2
        while True:
            kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(X=l.reshape(-1, 1))
            r2.append([cluster_cnt, get_rel_mean_cluster_separation(kmeans, l), kmeans])
            if r2[-1][1] < separation:
                r2.pop(-1)
                break
            cluster_cnt += 1
        return r2[-1]

    def get_rel_mean_cluster_separation(kmobj, data):
        o = kmobj.predict(data.reshape(-1, 1))
        d = {}
        for i, v in enumerate(o):
            label = v
            value = data[i]
            if not d.__contains__(label):
                d[label] = [value]
            else:
                d[label].append(value)
        _c = np.random.choice
        separation = []
        for k in d:
            nk = list(filter(lambda x: x != k, d))
            if len(nk) == 0:
                break
            for i in range(200):
                separation.append(np.abs(_c(d[k]) - _c(d[_c(nk)])))
        separation = np.min(separation) / (data.max() - data.min()) if len(separation) > 0 else 0
        return separation

    class AE:
        @staticmethod
        def run(layers, cycle, neck=1, act=lambda x: t.mul(x, 1)):
            en, de, tp = AE.create_autoencoder(feat, neck_size=neck, layers=layers)
            op = t.optim.SGD(tp, lr=0.1)

            def _(d):
                _y = AE.fprop_ae(x_input=d, encoders=en, decoders=de, act=act)  # , act=t.sigmoid)
                y = d
                return _y, y

            def _1(d):
                n = AE.fprop_ae(x_input=d, encoders=en, act=act)
                return n

            for i in range(cycle):
                l = _h(_, op)
                # print(layers, '  ', i, '   ', l)

            def get_loss_dist(data):
                r = []
                for d in data:
                    _y, y = _(d)
                    r.append(t.nn.MSELoss()(_y.unsqueeze(0), y.unsqueeze(0)).data.cpu().item())
                l = np.asarray(r)
                c, b = np.histogram(l, bins=40)
                print(c)
                print(l.mean(), l.std())
                plt.plot(c, label=cycle)
                plt.show()

                # get_clusters(l)

                return l

            def get_neck_dist(data):
                r = []
                for d in data:
                    n = AE.fprop_ae(x_input=d, encoders=en, act=act)
                    r.append(n.data.cpu().item())
                l = np.asarray(r)
                c, b = np.histogram(l, bins=40)
                print(c)
                print(l.mean(), l.std())
                plt.plot(c, label=cycle)
                plt.show()
                cluster_cnt = get_clusters(l)
                return l

            get_neck_dist(data)
            get_loss_dist(data)
            return

        @staticmethod
        def create_autoencoder(input_size, neck_size, layers=1):
            encoders = []
            decoders = []
            enc_out_sizes = [int(x) for x in np.linspace(input_size, neck_size, layers + 1)]
            training_params = []

            for i in range(layers):
                in_ = enc_out_sizes[i]
                out_ = enc_out_sizes[i + 1]
                dense = t.nn.Linear(in_, out_)
                dense.to(tdevice)
                encoders.append(dense)
                training_params.append({'params': list(dense.parameters())})

            for i in range(layers, 0, -1):
                in_ = enc_out_sizes[i]
                out_ = enc_out_sizes[i - 1]
                dense = t.nn.Linear(in_, out_)
                dense.to(tdevice)
                decoders.append(dense)
                training_params.append({'params': list(dense.parameters())})

            return encoders, decoders, training_params

        @staticmethod
        def fprop_ae(x_input, encoders=[], decoders=[], act=lambda x: t.mul(x, 1)):
            if encoders:
                for m in encoders:
                    x_input = act(m(x_input))
            if decoders:
                for m in decoders:
                    x_input = act(m(x_input))
            return x_input

    AE.run(layers=1, cycle=300, neck=1, act=t.sigmoid)


def lstm_soft_tokenizer_test1():
    '''
    using lstm as a soft tokenizer, this is a 2 token example, 0 and 1
    [1, '81212345123']
    [1, '454123451234']
    [1, '5123451234']
    [1, '531234512345123']
    [1, '3451234542']
    [1, '23453034']
    [1, '123451234512']
    [1, '4258121234512']
    [0, '582765987659876598']
    [0, '7987659876598']
    [0, '7659876598765987659']
    [0, '7659876598']
    [0, '582765987659876598']
    [1, '5987659876582765987']
    [0, '876598765987659876']
    [0, '87659876598']
    :return:
    '''
    from sklearn.cluster import KMeans
    dlen = 100
    noise = 0.3
    emb_size = 10

    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    def token_a():
        pattern = '12345'
        alter_pattern = '34'
        new_token = ''
        for p in pattern:
            new_token += pattern
            if np.random.rand() > 0.7:
                new_token += np.random.choice(list(alter_pattern))
        return new_token

    def token_b():
        pattern = '98765'
        alter_pattern = '78'
        new_token = ''
        for p in pattern:
            new_token += pattern
            if np.random.rand() > 0.7:
                new_token += np.random.choice(list(alter_pattern))
        return new_token

    def gen_string(gt):
        dstring = ''
        while len(dstring) <= dlen:
            if np.random.rand() < noise:
                dstring += str(np.random.randint(0, 100000))
            dstring += gt()
            # if np.random.rand() > 0.5:
            #     dstring += token_a()
            # else:
            #     dstring += token_b()
        return dstring[:dlen]

    a = gen_string(token_a)
    b = gen_string(token_b)
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=4, batch_size=1,
                                                   num_of_layers=1, device=tdevice)
    raw_d = []
    emb_data = []

    for i in range(8):
        sizee = np.random.randint(5, 20)
        poss = np.min([np.random.randint(dlen), dlen - sizee])
        sa = a[poss:np.min([poss + sizee, len(a) - 1])]
        raw_d.append(sa)
        semb = to_emb(sa)
        out1, h1 = lstm(semb.unsqueeze(1), init)
        emb_data.append(h1[1].squeeze())

    for i in range(8):
        sizee = np.random.randint(5, 20)
        poss = np.min([np.random.randint(dlen), dlen - sizee])
        sb = b[poss:np.min([poss + sizee, len(b) - 1])]
        raw_d.append(sb)
    emb_data = [to_emb(x) for x in raw_d]
    emb_data = [lstm(x.unsqueeze(1), init)[1][1].squeeze() for x in emb_data]

    emb_data = t.stack(emb_data)
    emb_data = emb_data.cpu().data.numpy()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(emb_data)
    kmeansresult = kmeans.predict(emb_data)
    result = []
    for i in range(len(raw_d)):
        result.append([kmeansresult[i], raw_d[i]])
    return result


def lstm_soft_tokenizer_test3_bprop():  # conclusion, use hpos 0, do not train!
    '''
    This test uses different levels of noise to see how it affects cluster distance

    hpos 0
    Trained:       hpos: 0      epoch: 0
    Mean 0 dist: 0.085740454    Mean 1 dist: 0.08034747
    Noise: 0  Mean 0 dist: 0.08337033    Mean 1 dist: 0.08020365
    Noise: 0.3  Mean 0 dist: 0.083686255    Mean 1 dist: 0.079644
    Noise: 0.6  Mean 0 dist: 0.08642816    Mean 1 dist: 0.08586879
    Noise: 0.9  Mean 0 dist: 0.08821142    Mean 1 dist: 0.09035872
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 0      epoch: 50
    Mean 0 dist: 0.097721    Mean 1 dist: 0.07404581
    Noise: 0  Mean 0 dist: 0.10306854    Mean 1 dist: 0.07742782
    Noise: 0.3  Mean 0 dist: 0.098747276    Mean 1 dist: 0.075664364
    Noise: 0.6  Mean 0 dist: 0.100001805    Mean 1 dist: 0.078215666
    Noise: 0.9  Mean 0 dist: 0.09953197    Mean 1 dist: 0.07350741
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 0      epoch: 100
    Mean 0 dist: 0.07127062    Mean 1 dist: 0.04568113
    Noise: 0  Mean 0 dist: 0.0789597    Mean 1 dist: 0.06558163
    Noise: 0.3  Mean 0 dist: 0.09540307    Mean 1 dist: 0.066584796
    Noise: 0.6  Mean 0 dist: 0.09451891    Mean 1 dist: 0.06918518
    Noise: 0.9  Mean 0 dist: 0.09060608    Mean 1 dist: 0.07060514
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 0      epoch: 200
    Mean 0 dist: 0.07298227    Mean 1 dist: 0.053299107
    Noise: 0  Mean 0 dist: 0.074950665    Mean 1 dist: 0.0685109
    Noise: 0.3  Mean 0 dist: 0.07969724    Mean 1 dist: 0.07138041
    Noise: 0.6  Mean 0 dist: 0.07909531    Mean 1 dist: 0.068273015
    Noise: 0.9  Mean 0 dist: 0.07962487    Mean 1 dist: 0.07284217
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 0      epoch: 400
    Mean 0 dist: 0.043070335    Mean 1 dist: 0.07312809
    Noise: 0  Mean 0 dist: 0.06168162    Mean 1 dist: 0.08479026
    Noise: 0.3  Mean 0 dist: 0.07748585    Mean 1 dist: 0.08131081
    Noise: 0.6  Mean 0 dist: 0.07637892    Mean 1 dist: 0.084395714
    Noise: 0.9  Mean 0 dist: 0.078387454    Mean 1 dist: 0.08522397
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

    hpos 1
    Trained:       hpos: 1      epoch: 0
    Mean 0 dist: 0.11025281    Mean 1 dist: 0.20556316
    Noise: 0  Mean 0 dist: 0.112020105    Mean 1 dist: 0.22835408
    Noise: 0.3  Mean 0 dist: 0.14473037    Mean 1 dist: 0.23362632
    Noise: 0.6  Mean 0 dist: 0.146825    Mean 1 dist: 0.2465113
    Noise: 0.9  Mean 0 dist: 0.16124994    Mean 1 dist: 0.23776141
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 1      epoch: 50
    Mean 0 dist: 0.026892697    Mean 1 dist: 0.2056124
    Noise: 0  Mean 0 dist: 0.07063694    Mean 1 dist: 0.25189567
    Noise: 0.3  Mean 0 dist: 0.099446036    Mean 1 dist: 0.2817078
    Noise: 0.6  Mean 0 dist: 0.13513166    Mean 1 dist: 0.280791
    Noise: 0.9  Mean 0 dist: 0.13856702    Mean 1 dist: 0.27545655
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 1      epoch: 100
    Mean 0 dist: 0.13101763    Mean 1 dist: 0.15390229
    Noise: 0  Mean 0 dist: 0.19830903    Mean 1 dist: 0.18540102
    Noise: 0.3  Mean 0 dist: 0.25799206    Mean 1 dist: 0.20213817
    Noise: 0.6  Mean 0 dist: 0.27620098    Mean 1 dist: 0.21329604
    Noise: 0.9  Mean 0 dist: 0.30428806    Mean 1 dist: 0.21329749
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 1      epoch: 200
    Mean 0 dist: 0.21384151    Mean 1 dist: 0.184044
    Noise: 0  Mean 0 dist: 0.26144183    Mean 1 dist: 0.17323501
    Noise: 0.3  Mean 0 dist: 0.26641637    Mean 1 dist: 0.16037549
    Noise: 0.6  Mean 0 dist: 0.25684977    Mean 1 dist: 0.16037549
    Noise: 0.9  Mean 0 dist: 0.26100856    Mean 1 dist: 0.18366113
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    Trained:       hpos: 1      epoch: 400
    Mean 0 dist: 0.2137606    Mean 1 dist: 0.17700988
    Noise: 0  Mean 0 dist: 0.2314634    Mean 1 dist: 0.18622921
    Noise: 0.3  Mean 0 dist: 0.23173063    Mean 1 dist: 0.16729419
    Noise: 0.6  Mean 0 dist: 0.2504138    Mean 1 dist: 0.16967341
    Noise: 0.9  Mean 0 dist: 0.25338277    Mean 1 dist: 0.17086087
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    '''
    from sklearn.cluster import KMeans
    dlen = 100
    noises = [0, 0.3, 0.6, 0.9]
    emb_size = 10

    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    def token_a():
        pattern = '112233'
        alter_pattern = '34'
        new_token = ''
        for p in pattern:
            new_token += p
            if np.random.rand() > 0.7:
                new_token += np.random.choice(list(alter_pattern))
        return new_token

    def token_b():
        pattern = '332211'
        alter_pattern = '56'
        new_token = ''
        for p in pattern:
            new_token += p
            if np.random.rand() > 0.7:
                new_token += np.random.choice(list(alter_pattern))
        return new_token

    def gen_string(gt, noise):
        dstring = ''
        while len(dstring) <= dlen:
            if np.random.rand() < noise:
                dstring += str(np.random.randint(0, 100000))
            dstring += gt()
        return dstring[:dlen]

    a = gen_string(token_a, 0)
    b = gen_string(token_b, 0)
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=emb_size, batch_size=1,
                                                   num_of_layers=1,
                                                   device=tdevice)

    def __q(k, cnt):
        raw_d = []
        for i in range(cnt):
            sizee = np.random.randint(7, 20)
            poss = np.min([np.random.randint(dlen), dlen - sizee])
            sa = k[poss:np.min([poss + sizee, len(a) - 1])]
            raw_d.append(sa)
        return raw_d

    raw_d = []
    raw_d += __q(a, 10)
    raw_d += __q(b, 10)

    for epoch in [0]:
        for hpos in [0]:

            # ~~~~~~~~start training~~~~~~~~~~~~~~~~
            raw_x = [x[:-1] for x in raw_d]
            raw_y = [x[1:] for x in raw_d]

            emb_x = [to_emb(x) for x in raw_x]
            emb_y = [to_emb(x) for x in raw_y]

            op = t.optim.SGD(lstm.parameters(), lr=0.001)
            for i in range(epoch):
                _y = [lstm(x.unsqueeze(1), init)[0].squeeze() for x in emb_x]
                y = emb_y

                for ii in range(len(emb_x)):
                    _yy = _y[ii]
                    yy = y[ii]

                    loss = t.nn.MSELoss()(_yy, yy)
                    loss.backward(retain_graph=True)

                op.step()
                op.zero_grad()

            # ~~~~~~~~~stop training~~~~~~~~~~~~~~~~
            emb_data = [to_emb(x) for x in raw_d]
            emb_data = [lstm(x.unsqueeze(1), init)[1][hpos].squeeze() for x in emb_data]

            emb_data = t.stack(emb_data)
            emb_data = emb_data.cpu().data.numpy()
            kmeans = KMeans(n_clusters=2, random_state=0).fit(emb_data)
            kmeansresult = ml_helper.calculate_kmeans_l2_dist(emb_data, kmeans)
            result = []
            print('Trained:', '      hpos:', hpos, '     epoch:', epoch)
            for i in range(len(raw_d)):
                dd = [kmeansresult[i][0], kmeansresult[i][1], raw_d[i]]
                print(dd)
                result.append(dd)
            _f = lambda xx, r: np.mean([x[1] for x in list(filter(lambda x: x[0] == xx, r))])
            print('Mean 0 dist:', _f(0, result), '   Mean 1 dist:', _f(1, result))
            for noise in noises:
                a = gen_string(token_a, noise)
                b = gen_string(token_b, noise)

                raw_d = []
                raw_d += __q(a, 6)
                raw_d += __q(b, 6)

                emb_data = [to_emb(x) for x in raw_d]
                emb_data = [lstm(x.unsqueeze(1), init)[1][hpos].squeeze() for x in emb_data]

                emb_data = t.stack(emb_data)
                emb_data = emb_data.cpu().data.numpy()

                kmeansresult = ml_helper.calculate_kmeans_l2_dist(emb_data, kmeans)
                for i in range(len(raw_d)):
                    dd = [kmeansresult[i][0], kmeansresult[i][1], raw_d[i]]
                    print(dd)
                    result.append(dd)
                print('Noise:', noise, ' Mean 0 dist:', _f(0, result), '   Mean 1 dist:', _f(1, result))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')

    return result


def pca_and_svd():
    from sklearn.decomposition import PCA
    mat_a = np.random.rand(5, 5)
    # test numpy and scikearn 's PCA
    pca = PCA(n_components=5)
    pca.fit(mat_a)

    cov = np.cov(mat_a.transpose())
    ev, eig = np.linalg.eig(cov)  # the values here should be exactly the same as pca.explained_variance_

    U, S, V = np.linalg.svd(cov)
    # equality between SVD and PCA only works on square symmetric positive semidefinite matrix
    # S must be SAME as EV
    return


class sequence_clustering_v1:
    

    @staticmethod
    def sequence_anomaly_detector_lstm():
        '''
        Conclusion: Good, untrained lstm is only sensitive to the most recent events, so need reverse order
        sum cumulation cant work as well because everything becomes 1D
        :return:
        '''
        import plotly.graph_objects as go
        fig = go.Figure()
        emb_size = 5
        hid_size = 100
        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        lstm = t.nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, bidirectional=False)
        fc = t.rand(size=[hid_size, 8])

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def _h(path, lim):
            import os
            with open(os.path.dirname(__file__) + path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]
            data1 = [s[::-1] for s in data]
            data_emb = [str_to_emb(x) for x in data1]

            out_emb = [lstm(x.unsqueeze(1))[1][0].squeeze(0).mm(fc).squeeze().data.numpy() for x in data_emb]

            x = [x[0] for x in out_emb]
            y = [x[1] for x in out_emb]
            z = [x[2] for x in out_emb]
            return x, y, z, data

        x1, y1, z1, n1 = _h('/data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('/data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_with_fc():
        '''
        fc dim must be >3 for clustering to work in 3d space, else 2d or less!
        :return:
        '''
        import plotly.graph_objects as go
        fig = go.Figure()
        emb_size = 100
        out = 200

        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        # lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
        #                                                device=tdevice)
        modela = 2 * t.rand(size=[emb_size + out, out]) - 1
        bias = 2 * t.rand(size=[out]) - 1
        fc = t.rand(size=[out, 12])

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def fprop(data, p=t.zeros(size=[out])):
            # input = t.cat([data[0], p])
            # output = (input.unsqueeze(0).mm(modela).squeeze()).sigmoid()
            # # output = (input.unsqueeze(0).mm(modela).squeeze() + bias).sigmoid()
            # ldata = data[1:]
            # if ldata.size()[0] > 0:
            #     return fprop(ldata, output)
            # else:
            #     return output.unsqueeze(0).mm(fc).squeeze().softmax(dim=0)
            for d in data:
                ind = t.cat([d, p])
                p = (ind.unsqueeze(0).mm(modela).squeeze()).sigmoid()
            return p

        def _h(path, lim):
            with open('/'.join(__file__.split('/')[:-1]) + path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]
            data1 = [s[::-1] for s in data]

            out_emb = [fprop(x) for x in [str_to_emb(x) for x in data1]]
            x = [x[0] for x in out_emb]
            y = [x[1] for x in out_emb]
            z = [x[2] for x in out_emb]
            return x, y, z, data

        x1, y1, z1, n1 = _h('/data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('/data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_with_fc_and_pca():
        '''
        This one uses PCA for dim reduction, results are better because they are better spread out
        :return:
        '''
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        fig = go.Figure()
        emb_size = 100
        out = 200

        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        # lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
        #                                                device=tdevice)
        modela = 2 * t.rand(size=[emb_size + out, out]) - 1
        bias = 2 * t.rand(size=[out]) - 1
        fc = t.rand(size=[out, 12])

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def fprop(data, p=t.zeros(size=[out])):
            # input = t.cat([data[0], p])
            # output = (input.unsqueeze(0).mm(modela).squeeze()).sigmoid()
            # # output = (input.unsqueeze(0).mm(modela).squeeze() + bias).sigmoid()
            # ldata = data[1:]
            # if ldata.size()[0] > 0:
            #     return fprop(ldata, output)
            # else:
            #     return output.unsqueeze(0).mm(fc).squeeze().softmax(dim=0)
            for d in data:
                ind = t.cat([d, p])
                p = (ind.unsqueeze(0).mm(modela).squeeze()).sigmoid()
            return p

        def _h(path, lim):
            with open('/'.join(__file__.split('/')[:-1]) + path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]
            data1 = [s[::-1] for s in data]

            out_emb = [fprop(x) for x in [str_to_emb(x) for x in data1]]
            out_emb = t.stack(out_emb).data.numpy()
            pca = PCA(n_components=3)
            out2 = pca.fit_transform(out_emb)

            x = [x[0] for x in out2]
            y = [x[1] for x in out2]
            z = [x[2] for x in out2]
            return x, y, z, data

        x1, y1, z1, n1 = _h('/data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('/data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_with_fc_strong_hierarchical():
        '''
        Hierarchical structure is clearly seen here.
        :return:
        '''
        import plotly.graph_objects as go
        fig = go.Figure()
        emb_size = 100
        out = 200

        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        modela = 2 * t.rand(size=[emb_size + out, out]) - 1
        fc = t.rand(size=[out, 8])

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def fprop2(data):
            acum = t.zeros(size=[emb_size])
            decay = 1
            for i in range(len(data)):
                acum += data[i] / decay
                decay *= 1.2
            return acum

        def _h(path, lim):
            with open(path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]

            out_emb = [fprop2(x) for x in [str_to_emb(x) for x in data]]
            x = [x[0] for x in out_emb]
            y = [x[1] for x in out_emb]
            z = [x[2] for x in out_emb]
            return x, y, z, data

        x1, y1, z1, n1 = _h('./data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('./data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    def sequence_anomaly_detector_autoencoder():
        '''
        The hierarchical clustering is better than sequence_anomaly_detector_with_fc_strong_hierarchical() because
        parent directory is grouped more closely to its child
        :return:
        '''
        import plotly.graph_objects as go
        from ml_helper import TorchHelper as th
        fig = go.Figure()
        emb_size = 10
        out = 20

        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        # lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
        #                                                device=tdevice)
        modela = 2 * t.rand(size=[emb_size + out, out]) - 1
        enc, ep = th.create_linear_layers(layer_sizes=[out, 30, 10])
        dec, dp = th.create_linear_layers(layer_sizes=[10, 30, out])
        op = t.optim.SGD(params=ep + dp, lr=1e-1)

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def train(data, epoch):
            for i in range(epoch):
                _y, y, _ = fprop(data)

                loss = t.nn.MSELoss()(_y, y)
                print(loss)
                loss.backward()
                op.step()
                op.zero_grad()
            return

        def fprop(data):
            def _h(data, p=t.zeros(size=[out])):
                for d in data:
                    ind = t.cat([d, p])
                    p = (ind.unsqueeze(0).mm(modela).squeeze()).sigmoid()
                return p

            emb = [_h(d) for d in data]

            def _f(d, nn):
                out = d
                for n in nn:
                    out = n(out)
                return out

            def ff(d):
                encoding = _f(d, enc)
                output = _f(encoding, dec)
                return output, encoding

            yenc = [ff(x) for x in emb]
            _y = [x[0] for x in yenc]
            encodings = [x[1] for x in yenc]
            return t.stack(_y), t.stack(emb), t.stack(encodings)

        def _h(path, lim):
            with open(path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]
            data1 = [s[::-1] for s in data]
            out_emb = [str_to_emb(x) for x in data1]
            train(out_emb, 50)
            _y, _y, encodings = fprop(out_emb)
            encodings = encodings.cpu().data.numpy()
            x = [x[0] for x in encodings]
            y = [x[1] for x in encodings]
            z = [x[2] for x in encodings]
            return x, y, z, data

        x1, y1, z1, n1 = _h('./data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        #
        # x2, y2, z2, n2 = _h('./data/python_lib_paths.txt', 1000)
        #
        # # start plotting
        # fig.add_trace(go.Scatter3d(
        #     x=x2, y=y2, z=z2,
        #     hovertext=n2,
        #     hoverinfo='text',  # this means xzy info is removed from hover
        #     name="b",
        #     mode='markers',
        #     marker=dict(
        #         size=8,
        #         color='green',  # set color to an array/list of desired values
        #         opacity=0.7
        #     )
        # ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_with_fc2_for_string():
        '''
        visually bad
        :return:
        '''
        import plotly.graph_objects as go
        fig = go.Figure()
        emb_size = 100
        out = 200

        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        modela = 2 * t.rand(size=[emb_size + out, out]) - 1
        fc = t.rand(size=[out, 8])

        def str_to_emb(stri):
            def _h(_str):
                pos = [s.find(x) for x in _str]
                new_char = np.where(np.asarray(pos) == -1)[0]
                for pp in new_char:
                    pos[pp] = 0
                    print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
                if pos.__contains__(-1):
                    if pos.index(-1) >= 0:
                        print('Unknown char found:', _str[pos.index(-1)])
                pos = t.tensor(pos, dtype=t.int64, device=tdevice)
                return char_emb.index_select(dim=0, index=pos)

            tokens = stri.split('/')
            token_emb = [_h(x).mean(dim=0) for x in tokens]
            return token_emb

        def fprop2(data):
            acum = t.zeros(size=[emb_size])
            decay = 1
            for i in range(len(data)):
                acum += data[i] / decay
                decay *= 20
            return acum

        def _h(path, lim):
            with open(path, 'r') as f:
                data = f.readlines()
            data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]

            out_emb = [fprop2(x) for x in [str_to_emb(x) for x in data]]
            x = [x[0] for x in out_emb]
            y = [x[1] for x in out_emb]
            z = [x[2] for x in out_emb]
            return x, y, z, data

        x1, y1, z1, n1 = _h('./data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('./data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_mds():
        '''
        Conclusion: Half-good. Close points are related, far points may be related
        :return:
        '''
        import plotly.graph_objects as go
        from sklearn.manifold import MDS
        fig = go.Figure()
        emb_size = 10
        out = 5

        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        # lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
        #                                                device=tdevice)
        modela = 2 * t.rand(size=[emb_size + out, out]) - 1
        bias = 2 * t.rand(size=[out]) - 1

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def fprop(data, p=t.zeros(size=[out])):
            input = t.cat([data[0], p])
            output = (input.unsqueeze(0).mm(modela).squeeze()).sigmoid()
            # output = (input.unsqueeze(0).mm(modela).squeeze() + bias).sigmoid()
            ldata = data[1:]
            if ldata.size()[0] > 0:
                return fprop(ldata, output)
            else:
                return output
            return

        def _h(path, lim):
            with open(path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]
            data1 = [s[::-1] for s in data]
            data_emb = [str_to_emb(x) for x in data1]

            emb_out = [fprop(x).cpu().data.numpy() for x in data_emb]
            emb_out = np.asarray(emb_out)

            embedding = MDS(n_components=3)
            emb_out = embedding.fit_transform(emb_out)

            x = [x[0] for x in emb_out]
            y = [x[1] for x in emb_out]
            z = [x[2] for x in emb_out]
            return x, y, z, data

        x1, y1, z1, n1 = _h('./data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('./data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_with_fc_mds():
        '''
        While FC is good, MDS screws up clustering
        :return:
        '''
        import plotly.graph_objects as go
        from sklearn.manifold import MDS
        fig = go.Figure()
        emb_size = 150
        out = 500

        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        # lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
        #                                                device=tdevice)
        modela = 2 * t.rand(size=[emb_size + out, out]) - 1
        bias = 2 * t.rand(size=[out]) - 1
        fc = t.rand(size=[out, 5])

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def fprop(data, p=t.zeros(size=[out])):
            input = t.cat([data[0], p])
            output = (input.unsqueeze(0).mm(modela).squeeze()).sigmoid()
            # output = (input.unsqueeze(0).mm(modela).squeeze() + bias).sigmoid()
            ldata = data[1:]
            if ldata.size()[0] > 0:
                return fprop(ldata, output)
            else:
                return output.unsqueeze(0).mm(fc).squeeze().softmax(dim=0)
            return

        def _h(path, lim):
            with open(path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]
            data1 = [s[::-1] for s in data]
            data_emb = [str_to_emb(x) for x in data1]

            emb_out = [fprop(x).cpu().data.numpy() for x in data_emb]
            emb_out = np.asarray(emb_out)

            embedding = MDS(n_components=3)
            emb_out = embedding.fit_transform(emb_out)
            x = [x[0] for x in emb_out]
            y = [x[1] for x in emb_out]
            z = [x[2] for x in emb_out]
            return x, y, z, data

        x1, y1, z1, n1 = _h('./data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('./data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_lstm_mds():
        '''
        Conclusion: MDS seems to screw up micro clusters that are very close together yet still retain their distinct shapes
        :return:
        '''
        import plotly.graph_objects as go
        from sklearn.manifold import MDS
        fig = go.Figure()
        emb_size = 5
        s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
        char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1,
                                                       num_of_layers=1,
                                                       device=tdevice)

        def str_to_emb(_str):
            pos = [s.find(x) for x in _str]
            new_char = np.where(np.asarray(pos) == -1)[0]
            for pp in new_char:
                pos[pp] = 0
                print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
            if pos.__contains__(-1):
                if pos.index(-1) >= 0:
                    print('Unknown char found:', _str[pos.index(-1)])
            pos = t.tensor(pos, dtype=t.int64, device=tdevice)
            return char_emb.index_select(dim=0, index=pos)

        def _h(path, lim):
            import os
            with open(os.path.dirname(__file__) + path, 'r') as f:
                data = f.readlines()
            # data = [x[:-1] for x in data]  # remove last char \n for a start
            data = data[5:lim]
            data1 = [s[::-1] for s in data]
            data_emb = [str_to_emb(x) for x in data1]

            emb_out = [lstm(x.unsqueeze(1), init)[1][0].squeeze().cpu().data.numpy() for x in data_emb]
            embedding = MDS(n_components=3)
            emb_out = embedding.fit_transform(np.asarray(emb_out))
            x = [x[0] for x in emb_out]
            y = [x[1] for x in emb_out]
            z = [x[2] for x in emb_out]
            return x, y, z, data

        x1, y1, z1, n1 = _h('/data/nodejs_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            hovertext=n1,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="a",
            mode='markers',
            marker=dict(
                size=8,
                color='blue',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))

        x2, y2, z2, n2 = _h('/data/python_lib_paths.txt', 1000)

        # start plotting
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            hovertext=n2,
            hoverinfo='text',  # this means xzy info is removed from hover
            name="b",
            mode='markers',
            marker=dict(
                size=8,
                color='green',  # set color to an array/list of desired values
                opacity=0.7
            )
        ))
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.show()
        return

    @staticmethod
    def sequence_anomaly_detector_length_limit():
        '''
        Conclusion: Sequence length sensitvity is determined by hidden_layer size till optimal peak
        Emb size: 5     Hid size: 400    Max str len: 215
        Emb size: 10     Hid size: 400    Max str len: 190
        Emb size: 20     Hid size: 400    Max str len: 210
        Emb size: 50     Hid size: 400    Max str len: 65
        Emb size: 200     Hid size: 400    Max str len: 460
        Emb size: 5     Hid size: 10    Max str len: 45
        Emb size: 5     Hid size: 20    Max str len: 45
        Emb size: 5     Hid size: 40    Max str len: 55
        Emb size: 5     Hid size: 60    Max str len: 75
        Emb size: 5     Hid size: 80    Max str len: 440  <---peak, after this is a decline
        Emb size: 5     Hid size: 100    Max str len: 285
        Emb size: 5     Hid size: 300    Max str len: 270
        Emb size: 5     Hid size: 500    Max str len: 140
        Emb size: 5     Hid size: 800    Max str len: 70
        :return:
        '''
        emb_size = 5

        def f(emb_size, hid_size):
            s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
            char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
            lstm = t.nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, bidirectional=False)

            def str_to_emb(_str):
                pos = [s.find(x) for x in _str]
                new_char = np.where(np.asarray(pos) == -1)[0]
                for pp in new_char:
                    pos[pp] = 0
                    print('Unknown char found:', _str[pp], ' and replaced with WHITESPACE')
                if pos.__contains__(-1):
                    if pos.index(-1) >= 0:
                        print('Unknown char found:', _str[pos.index(-1)])
                pos = t.tensor(pos, dtype=t.int64, device=tdevice)
                return char_emb.index_select(dim=0, index=pos)

            def _h(lim):
                roots = ''.join(np.random.choice(list(s), lim))
                data = [
                    roots + np.random.choice(list(s)),
                    roots + np.random.choice(list(s)),
                ]
                data1 = [s[::-1] for s in data]
                data_emb = [str_to_emb(x) for x in data1]

                lstmc_hid_outputs = [lstm(x.unsqueeze(1))[1][0].squeeze().cpu().data.numpy() for x in data_emb]
                print(hid_size, ':', lim, ':', (lstmc_hid_outputs[0] - lstmc_hid_outputs[1]).mean())
                return

            for i in range(20, 500, 5):
                roots = ''.join(np.random.choice(list(s), i))
                data = [
                    roots + np.random.choice(list(s)),
                    roots + np.random.choice(list(s)),
                ]
                data1 = [s[::-1] for s in data]
                data_emb = [str_to_emb(x) for x in data1]

                lstmc_hid_outputs = [lstm(x.unsqueeze(1))[1][0].squeeze().cpu().data.numpy() for x in data_emb]
                diff = (lstmc_hid_outputs[0] - lstmc_hid_outputs[1]).mean()
                if diff == 0.0:
                    print('Emb size:', emb_size, '    Hid size:', hid_size, '   Max str len:', i)
                    break

        for emb_size in [5, 10, 20, 30, 50, 70, 100, 200]:
            f(emb_size, 400)
        for hid_size in [10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 800, 1000, 2000]:
            f(5, hid_size)

        return


if __name__ == '__main__':
    sequence_clustering_v1.sequence_anomaly_detector_with_fc()
