import torch as t
import numpy as np
import time
import datetime as dt
import copy
import ml_helper
from sklearn import tree
from sklearn.metrics import accuracy_score
import datetime as dt

tdevice = t.device('cuda')


def test_one_hot():
    x = np.random.rand(100, 2)
    y = np.zeros(100)
    for i in range(len(x)):
        y[i] = 1 if np.sum(x[i]) > 1 else 0

    x = t.tensor(x, dtype=t.float32)
    y = t.tensor(y, dtype=t.int64)

    oh = t.eye(2)
    y = oh.index_select(0, y)

    L1 = t.nn.Linear(2, 1000)
    L2 = t.nn.Linear(1000, 2)

    opt = t.optim.Adam(list(L1.parameters()) + list(L2.parameters()), lr=0.01)

    def closs(y, y_):
        return y.sub(y_).pow(2).mean()

    for i in range(100):
        y_ = t.nn.Softmax()(L2(L1(x)).sigmoid())
        t1 = time.time()
        # bceloss = t.nn.BCELoss()(y_, y)
        t2 = time.time()
        # bceloss.backward(retain_graph=True)
        t3 = time.time()

        cus_loss = closs(y_, y)
        t4 = time.time()
        cus_loss.backward(retain_graph=True)
        t5 = time.time()

        # print('BCE:', str(t2 - t1), '    BCE bw:', str(t3 - t2), '   cL:', str(t4 - t3), '   cL bw:', str(t5 - t4))
        wrong = y_.round().sub(y).abs().sum().data.item()
        print(wrong)
        opt.step()
        opt.zero_grad()


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


if __name__ == '__main__':
    continuous_timeseries_test()
