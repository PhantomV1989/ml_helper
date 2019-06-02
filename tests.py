import torch as t
import numpy as np
import time
import datetime as dt
import copy
import ml_helper
from sklearn import tree
from sklearn.metrics import accuracy_score
import datetime as dt

tdevice = t.device('cpu')


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


def test_lstm_noisy_seq():
    # bad at convergence, very slow
    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    isize = 5
    data_count = 20
    time_length = 10
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


def noisy_timeseries_test():
    # conclusion: noise <50% to see good validation results
    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    isize = 3
    data_count = 200
    time_length = 7
    emb = t.rand([10, isize], device=tdevice)
    lr = 0.2
    a = []
    b = []

    ya = t.tensor([[1.0, 0]] * data_count, device=tdevice)
    yb = t.tensor([[0, 1.0]] * data_count, device=tdevice)

    y = t.cat([ya, yb])

    # ~~~~~~~~~~~~~~~Data creation~~~~~~~~~~~~~~~~~~`
    while True:
        ttt = ''.join(np.random.choice(num, time_length))
        if ttt.__contains__('123'):
            if len(a) < data_count: a.append(ttt)
        else:
            if len(b) < data_count: b.append(ttt)
        if len(a) >= data_count and len(b) >= data_count:
            break

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = [emb.index_select(dim=0, index=x) for x in a]
        return a

    emba = to_emb(a)
    embb = to_emb(b)

    def _h(fp, op):
        aout = [fp(x) for x in emba]
        bout = [fp(x) for x in embb]

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

        while True:
            asd = ''.join(np.random.choice(num, time_length))
            if asd.__contains__('123'):
                if len(a) < 50:
                    a.append(asd)
            else:
                if len(b) < 50:
                    b.append(asd)
            if len(a) >= 50 and len(b) >= 50:
                break
        ra = [fp(to_emb([x])[0])[0].data.item() for x in a]
        rb = [fp(to_emb([x])[0])[0].data.item() for x in b]

        correct = len(list(filter(lambda x: x > 0.5, ra))) + len(list(filter(lambda x: x <= 0.5, rb)))
        wrong = len(list(filter(lambda x: x < 0.5, ra))) + len(list(filter(lambda x: x >= 0.5, rb)))

        return {'correct': correct, 'wrong': wrong}

    def att_model():
        print('~~~~~~~~~~~~`Attention model test~~~~~~~~~~~')  # sucks!
        depth = 80
        field_size = 5
        unit = 20
        # first layer 20 units of 5 fields each
        first_layer = [t.nn.Linear(field_size * isize, depth) for i in range(unit)]
        [x.to(tdevice) for x in first_layer]

        # 2nd layer 1 unit, 10 fields x depth, out=2
        second_layer = t.nn.Linear(unit * depth, 2)
        second_layer.to(tdevice)

        am_params = [
            {'params': sum([list(x.parameters()) for x in first_layer], [])},
            {'params': list(second_layer.parameters())},
        ]
        am_optim = t.optim.SGD(am_params, lr=0.1 / data_count / 2)

        def am_fprop(x):
            fl = []
            for i in range(20):
                start = i * field_size
                tt = x[start:start + field_size].reshape(-1)
                r = first_layer[i](tt)
                fl.append(r)
            sli = t.cat(fl)
            output = second_layer(sli).softmax(dim=0)
            return output

        def am_test():
            asd = ''.join(np.random.choice(num, time_length))
            r = am_fprop(to_emb([asd])[0])
            return asd.__contains__('123'), r

        for i in range(300):
            print('Iter:', i)
            loss = _h(am_fprop, am_optim)
            print(' Loss:', loss)
        return

    def cnn_test():
        print('~~~~~~~~~~~~~~~~Conv Neural Network test~~~~~~~~~~~~~~~~~~~~')
        filter_size = 10
        depth = 100

        filter_w = t.nn.Linear(filter_size * isize, depth)
        filter_w.to(tdevice)
        last_w = t.nn.Linear(depth, 2)
        last_w.to(tdevice)

        params = [
            {'params': list(filter_w.parameters())},
            {'params': list(last_w.parameters())}
        ]
        cnn_optim = t.optim.SGD(params, lr=0.04)

        def cnn_fprop(x):
            tmp = []
            for i in range(0, len(x) - filter_size + 1, 2):
                d = x[i: i + filter_size].reshape(-1)
                o = filter_w(d)
                tmp.append(o)
            r = last_w(t.stack(tmp).mean(dim=0)).softmax(dim=0)
            return r

        (test)
        for i in range(1000):
            loss = _h(cnn_fprop, cnn_optim)
            print(loss)
        r = [test(cnn_fprop) for i in range(100)]
        return r

    def lstm_test():
        print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')
        out_size = 10
        layers = 1
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=isize, output_size=out_size, batch_size=1,
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
        lstm_optim = t.optim.SGD(params, lr=0.4)

        def lstm_fprop(x):  # upper 60~72%  on 100 iter
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
            r = f(to_emb([asd])[0])
            return asd.__contains__('123'), r

        # for i in range(100):
        #     print('Iter[0]:', i)
        #     loss = _h(lstm_fprop, lstm_optim)
        #     print(' Loss:', loss)
        # r0 = [test(lstm_fprop) for i in range(50)]

        # for i in range(100):  # 70~80% valid on 4:3, noise:signal
        #     print('Iter[1]:', i)
        #     loss = _h(lstm_fprop1, lstm_optim)
        #     print(i, ' Loss:', loss)
        # r1 = [test(lstm_fprop1) for i in range(50)]

        for i in range(100):  # 68~73% valid on 4:3, noise:signal
            print('Iter[2]:', i)
            loss = _h(lstm_fprop2, lstm_optim)
            print(i, ' Loss:', loss)
        r2 = [test(lstm_fprop2) for i in range(50)]
        return

    def ann_test():
        # # 74~77% valid on 4:3, noise:signal, loss 0.467
        print('~~~~~~~~~~~~~~~~ANN test~~~~~~~~~~~~~~~~~~~~')
        w1 = t.nn.Linear(time_length * isize, 1000)
        w2 = t.nn.Linear(1000, 2)

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

        (emba, embb, test)
        for i in range(100):
            loss = _h(ann_fprop, optim)
            print(i, ' Loss:', loss)
        r = [test(ann_fprop) for i in range(100)]
        return r

    def dtree_test():
        # noise to signal ratio about 70% noise,, overfits data at 100% accuracy, validation results 65~72%
        clf = tree.DecisionTreeClassifier()
        xa = [np.asarray(x.reshape(-1)) for x in emba]
        xb = [np.asarray(x.reshape(-1)) for x in embb]
        xx = xa + xb
        clf = clf.fit(xx, y)

        def fp(xx):
            return clf.predict(xx.reshape(1, -1))[0]

        r = [test(fp) for i in range(100)]
        return r

    def recursive_ann_test():
        # very slow, sub 50~60
        context = 400
        w1 = t.nn.Linear(context + isize, context)
        w2 = t.nn.Linear(context, 2)

        params = [
            {'params': w1.parameters()},
            {'params': w2.parameters()}
        ]

        optim = t.optim.SGD(params, lr=0.03)

        def fprop(x):
            init = t.tensor([0.0] * context)
            for i in range(time_length):
                in_ = t.cat((init, x[0]))
                init = w1(in_)
            r = w2(init).softmax(dim=0)
            return r

        (emba, embb, test)
        for i in range(100):
            loss = _h(fprop, optim)
            print(i, ' Loss:', loss)
        r = [test(fprop) for i in range(100)]
        return r

    # qq4 = cnn_test()
    qq = ann_test()
    # qq1 = recursive_ann_test()
    # qq2 = dtree_test()
    qq3 = lstm_test()
    return


def context_cluster_test():  # conclusion, use dtree
    dim = 7
    a = t.rand(dim, device=tdevice).unsqueeze(0)
    b = t.rand(dim, device=tdevice).unsqueeze(0)

    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    data_count = 300
    time_length = 5
    emb = t.rand([10, dim], device=tdevice)
    context_x = []
    context_y = []

    ya = t.tensor([1, 0.0], device=tdevice)
    yb = t.tensor([0.0, 1.0], device=tdevice)

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = [emb.index_select(dim=0, index=x) for x in a]
        return a

    # ~~~~~~~~~~~~~~~Data creation~~~~~~~~~~~~~~~~~~`
    while True:
        ttt = ''.join(np.random.choice(num, time_length))
        if ttt.__contains__('1'):
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
        rnd_context = to_emb(rnd_context)

        combined = [current_label] + rnd_context
        combined = t.stack(combined).squeeze()

        if len(current_class) < 80:
            current_class.append(combined)
        if len(class_a) >= 80 and len(class_b) >= 80:
            break

    def bprop(x, cl):
        loss = t.nn.BCELoss()(x, cl)
        return loss

    def _h(fp, op):
        aout = [fp(x) for x in class_a]
        bout = [fp(x) for x in class_b]

        _y = [x[0].data.cpu().item() for x in [aout + bout][0]]
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

    def _eval(fp):
        aout = [fp(x) for x in class_a]
        bout = [fp(x) for x in class_b]

        _y = [x[0].data.cpu().item() for x in [aout + bout][0]]
        y = undo_onehot([ya] * len(aout)) + undo_onehot([yb] * len(bout))
        scores = ml_helper.evaluate_optimized(_y, y)
        return scores

    def undo_onehot(d):
        return [i.data.cpu().numpy().argmax() for i in d]

    def test(f):
        asd = ''.join(np.random.choice(num, time_length))
        asd1 = to_emb(asd)

        xx = t.stack([a] + asd1).squeeze()
        yy = t.stack([b] + asd1).squeeze()

        r1 = f(xx)
        r2 = f(yy)

        return [asd.__contains__('1'), r1, r2]

    # # ~~~~~~~~~~~~~~~~~~~~Unordered cluster test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # context_current_w = t.nn.Linear(dim * 2, 100)
    # context_current_w.to(tdevice)
    #
    # final_w = t.nn.Linear(100, 2)
    # final_w.to(tdevice)
    #
    # params = [
    #     {'params': context_current_w.parameters()},
    #     {'params': final_w.parameters()},
    # ]
    #
    # optim = t.optim.SGD(params, lr=0.01 / data_count / 2)
    #
    # def uc_fprop(x):
    #     x1 = t.mean(x[1:], dim=0)
    #     x2 = x[0:1].squeeze()
    #     xx = t.cat((x1, x2))
    #     o = context_current_w(xx)
    #     r = final_w(o).softmax(dim=0)
    #     return r
    #
    # for i in range(1000):
    #     loss = _h(uc_fprop, optim)
    #     print(loss)

    # ~~~~~~~~~~~~~~~~~~~~Unordered cluster test v2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    w1 = t.nn.Linear(dim * (time_length + 1), 100)
    w1.to(tdevice)

    w11 = t.nn.Linear(100, 300)
    w11.to(tdevice)

    w2 = t.nn.Linear(300, 2)
    w2.to(tdevice)

    params = [
        {'params': w1.parameters()},
        {'params': w11.parameters()},
        {'params': w2.parameters()},
    ]
    optim = t.optim.SGD(params, lr=0.1 / data_count / 2)

    def uc_fprop2(x):
        x = x.reshape(-1)
        o = w11(w1(x))
        r = w2(o).softmax(dim=0)
        return r

    for i in range(300):
        loss = _h(uc_fprop2, optim)
        # test(uc_fprop2)
        print(loss)
    qwe = _eval(uc_fprop2)

    def dtree():
        aa = [x.reshape(-1).data.cpu().numpy() for x in class_a]
        bb = [x.reshape(-1).data.cpu().numpy() for x in class_b]

        ya = [0] * len(aa)
        yb = [1] * len(bb)

        x = aa + bb
        y = ya + yb
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x, y)
        accuracy = accuracy_score(y, clf.predict(x))
        print('Dtree train Accuracy:', accuracy)

        # ~~~~~test~~~~~~~
        wrong = 0
        for i in range(100):
            asd = ''.join(np.random.choice(num, time_length))
            asd1 = to_emb(asd)

            xx = t.stack([a] + asd1).squeeze().reshape(-1).data.numpy()
            yy = t.stack([b] + asd1).squeeze().reshape(-1).data.numpy()

            label = 0 if asd.__contains__('1') else 1
            r1 = clf.predict([xx])[0]
            r2 = clf.predict([yy])[0]
            if label != r1:
                wrong += 1
            # print('Truth:', label, '    r1:', r1, '    r2:', r2)
        print('Dtree valid Accuracy:', (100 - wrong) / 100)
        return

    dtree()
    return




if __name__ == '__main__':

