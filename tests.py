import torch as t
import numpy as np
import time
import copy
import ml_helper

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


def test_grad_accumulation():
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


def test_lr():
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


def test_lstm_order_sensitivity():
    return


def test_lstm_noisy_seq():
    # bad at convergence, very slow
    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    isize = 5
    data_count = 20
    time_length = 100
    emb = t.rand([10, isize], device=tdevice)

    a = []
    b = []

    ya = t.tensor([1, 0.0], device=tdevice)
    yb = t.tensor([0.0, 1.0], device=tdevice)

    # ~~~~~~~~~~~~~~~Data creation~~~~~~~~~~~~~~~~~~`
    while True:
        ttt = ''.join(np.random.choice(num, time_length))
        if ttt.__contains__('123'):
            if len(a) < data_count: a.append(ttt)
        else:
            if len(b) < data_count: b.append(ttt)
        if len(a) >= data_count and len(b) >= data_count:
            break

    def tensorify(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        return a

    a = tensorify(a)
    b = tensorify(b)

    def to_emb(a):
        a = [emb.index_select(dim=0, index=x) for x in a]
        return a

    emba = to_emb(a)
    embb = to_emb(b)

    for hc in [0, 1]:
        # ~~~~~~~~~~~~~~~Model creation per output~~~~~~~~~~~~~~~~~~
        lstm, init = ml_helper.TorchHelper.create_lstm_cell(input_size=isize, output_size=isize, batch_size=1,
                                                            device=tdevice)

        h_w = t.nn.Linear(isize, 2)
        h_w.to(tdevice)
        params = [
            {'params': list(lstm.parameters())},
            {'params': list(h_w.parameters())}
        ]
        optim = t.optim.SGD(params, lr=1 / data_count / 2)

        def fp(s):
            hc = init
            for ss in s:
                hc = lstm(ss.unsqueeze(0), hc)
            return hc

        outa = [fp(s) for s in emba]
        outb = [fp(s) for s in embb]

        def _hh(out, y, i):  # i=0 is h, i=1 is c
            ha = t.cat([x[i] for x in out])
            _y = t.softmax(h_w(ha), dim=1)
            losses = [t.nn.BCELoss()(yy, y) for yy in _y]
            [x.backward(retain_graph=True) for x in losses]
            return losses

        for i in range(100):
            la = _hh(outa, ya, hc)
            lb = _hh(outb, yb, hc)
            optim.step()
            optim.zero_grad()
            print(i, '   hc:', hc, '  Loss:', t.stack(la + lb).mean().data.item())

    return


def timeseries_test():
    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    isize = 5
    data_count = 100
    time_length = 100
    emb = t.rand([10, isize], device=tdevice)
    lr = 0.2
    a = []
    b = []

    ya = t.tensor([1, 0.0], device=tdevice)
    yb = t.tensor([0.0, 1.0], device=tdevice)

    # ~~~~~~~~~~~~~~~Data creation~~~~~~~~~~~~~~~~~~`
    while True:
        ttt = ''.join(np.random.choice(num, time_length))
        if ttt.__contains__('123'):
            if len(a) < data_count: a.append(ttt)
        else:
            if len(b) < data_count: b.append(ttt)
        if len(a) >= data_count and len(b) >= data_count:
            break

    def tensorify(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        return a

    a = tensorify(a)
    b = tensorify(b)

    def to_emb(a):
        a = [emb.index_select(dim=0, index=x) for x in a]
        return a

    def bprop(x, cl):
        loss = t.nn.BCELoss()(x, cl)
        return loss

    def _h(fp, a, b, op):
        aout = [fp(x) for x in a]
        bout = [fp(x) for x in b]

        _y = undo_onehot(aout) + undo_onehot(bout)
        y = undo_onehot([ya] * len(aout)) + undo_onehot([yb] * len(bout))
        scores = ml_helper.evaluate(_y, y)
        print(' F1', scores['F1'], '   TP:', scores['True positive'], '  TN:', scores['True negative'])
        aloss = [bprop(x, ya) for x in aout]
        bloss = [bprop(x, yb) for x in bout]

        loss = aloss + bloss

        [l.backward(retain_graph=True) for l in loss]
        op.step()
        op.zero_grad()
        return np.mean([l.data.item() for l in loss])

    def undo_onehot(d):
        return [i.data.cpu().numpy().argmax() for i in d]

    emba = to_emb(a)
    embb = to_emb(b)

    # # ~~~~~~~~~~~~~~~~~attention model~~~~~~~~~~~~~~~~~~~~~~~
    # print('~~~~~~~~~~~~`Attention model test~~~~~~~~~~~')  # sucks!
    # depth = 80
    # field_size = 5
    # unit = 20
    # # first layer 20 units of 5 fields each
    # first_layer = [t.nn.Linear(field_size * isize, depth) for i in range(unit)]
    # [x.to(tdevice) for x in first_layer]
    #
    # # 2nd layer 1 unit, 10 fields x depth, out=2
    # second_layer = t.nn.Linear(unit * depth, 2)
    # second_layer.to(tdevice)
    #
    # am_params = [
    #     {'params': sum([list(x.parameters()) for x in first_layer], [])},
    #     {'params': list(second_layer.parameters())},
    # ]
    # am_optim = t.optim.SGD(am_params, lr=0.01 / data_count / 2)
    #
    # def am_fprop(x):
    #     fl = []
    #     for i in range(20):
    #         start = i * field_size
    #         tt = x[start:start + field_size].reshape(-1)
    #         r = first_layer[i](tt)
    #         fl.append(r)
    #     sli = t.cat(fl)
    #     output = second_layer(sli).softmax(dim=0)
    #     return output
    #
    # def am_test():
    #     asd = ''.join(np.random.choice(num, time_length))
    #     r = am_fprop(to_emb(tensorify([asd]))[0])
    #     return asd.__contains__('123'), r
    #
    # for i in range(300):
    #     print('Iter:', i)
    #     loss = _h(am_fprop, emba, embb, am_optim)
    #     print(' Loss:', loss)
    #
    # print('~~~~~~~~~~~~~~~~Conv Neural Network test~~~~~~~~~~~~~~~~~~~~')
    # filter_size = 10
    # depth = 400
    #
    # filter_w = t.nn.Linear(filter_size * isize, depth)
    # filter_w.to(tdevice)
    # last_w = t.nn.Linear(depth, 2)
    # last_w.to(tdevice)
    #
    # params = [
    #     {'params': list(filter_w.parameters())},
    #     {'params': list(last_w.parameters())}
    # ]
    # cnn_optim = t.optim.SGD(params, lr=lr / data_count / 2)
    #
    # def cnn_fprop(x):
    #     tmp = []
    #     for i in range(len(x) - filter_size + 1):
    #         d = x[i: i + filter_size].reshape(-1)
    #         o = filter_w(d)
    #         tmp.append(o)
    #     r = last_w(t.stack(tmp).mean(dim=0)).softmax(dim=0)
    #     return r
    #
    # for i in range(100):
    #     loss = _h(cnn_fprop, emba, embb, cnn_optim)
    #     print(loss)

    print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')  # failed
    out_size = 100
    layers = 7
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=isize, output_size=out_size, batch_size=1,
                                                   num_of_layers=layers,
                                                   device=tdevice)
    last_w = t.nn.Linear(out_size * layers, 2)
    last_w.to(tdevice)

    last_w2 = t.nn.Linear(100 * 100, 2)
    last_w2.to(tdevice)

    params = [
        {'params': list(lstm.parameters())},
        {'params': list(last_w.parameters())},
        {'params': list(last_w2.parameters())}
    ]
    lstm_optim = t.optim.SGD(params, lr=lr / data_count / 2)

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

    def lstm_test(f):
        asd = ''.join(np.random.choice(num, time_length))
        r = f(to_emb(tensorify([asd]))[0])
        return asd.__contains__('123'), r

    # for i in range(100):
    #     print('Iter[0]:', i)
    #     loss = _h(lstm_fprop, emba, embb, lstm_optim)
    #     print(' Loss:', loss)
    for i in range(100):
        print('Iter[1]:', i)
        loss = _h(lstm_fprop1, emba, embb, lstm_optim)
        print(' Loss:', loss)
    for i in range(100):
        print('Iter[2]:', i)
        loss = _h(lstm_fprop2, emba, embb, lstm_optim)
        print(' Loss:', loss)

    print('~~~~~~~~~~~~~~~~ANN test~~~~~~~~~~~~~~~~~~~~')
    return


def context_cluster_test():
    dim = 7
    a = np.random.rand(dim)
    b = np.random.rand(dim)

    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    data_count = 100
    time_length = 100
    emb = t.rand([10, dim], device=tdevice)
    lr = 0.2
    context_x = []
    context_y = []

    ya = t.tensor([1, 0.0], device=tdevice)
    yb = t.tensor([0.0, 1.0], device=tdevice)

    # ~~~~~~~~~~~~~~~Data creation~~~~~~~~~~~~~~~~~~`
    while True:
        ttt = ''.join(np.random.choice(num, time_length))
        if ttt.__contains__('123'):
            if len(context_x) < data_count: context_x.append(ttt)
        else:
            if len(context_y) < data_count: context_y.append(ttt)
        if len(context_x) >= data_count and len(context_y) >= data_count:
            break
    # mix and match clusters
    class_a = []
    class_b = []

    while True:
        current_context = 'x' if np.random.rand() > 0.5 else 'y'
        current_label = 'a' if np.random.rand() > 0.5 else 'b'

        if current_label == 'a' and current_context == 'x':
            current_class = class_a
        else:
            current_class = class_b

        current_context = context_x if current_context == 'x' else context_y
        current_label = a if current_label == 'a' else b
        rnd_context = current_context[np.random.randint(len(current_context))]
        raise Exception('Not Done!')
        if len(class_a) >= 80 and len(class_b) >= 80:
            break

    return

def cumulative_sequential_nn_test():
    return


if __name__ == '__main__':
    context_cluster_test()
