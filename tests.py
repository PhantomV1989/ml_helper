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


def noise_training_noisy_negatives(x):
    # Instead of starting with a noisy base pattern, this example starts the data with healthy patterns
    # Negative samples are obtained by adding noises to the healthy data, where some of this noises are still part of
    # the healthy pattern.
    # Usecase: Detecting negative anomalies in a largely healthy dataset.
    # Goal: Model's ability to get high True Negatives when negative set contains healthy noises

    # Note: you should be very careful of how noises look here. Because I am using very short units (1 letter) to
    # represent noises, they very easily part of the healthy data, which makes hard to differentiate.
    # So you have 2 choices here, either make healthy patterns form a very small subset of the universal set or expand
    # the size of universal set by taking into account the sequences of noises

    # Results
    # 30% noise, 69% acc, 100% TP, 39% TN, 2000 iter
    # 50% noise, 87% acc, 100% TP, 75.4% TN, 2000 iter, might be due to lesser healthy variety
    # 70% noise  70% acc, 96% TP, 43% TN, 2000 iter

    healthy_patterns = ['123', '798']  # more variations mean higher entropy, harder to differentiate
    s = list('123456789')

    data_count = 100
    time_len = 15

    noise_level = x

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
        np.random.seed(dt.datetime.now().microsecond)
        random_unhealthy = list(''.join(np.random.choice(healthy_patterns, time_len * 5)))
        start = np.random.randint(0, len(random_unhealthy) - time_len)
        random_unhealthy = random_unhealthy[start:start + time_len]
        for ii in range(len(random_unhealthy)):
            if np.random.rand() > noise_level:
                random_unhealthy[ii] = np.random.choice(s)  # perturbs positive with noise
        label_a.append(''.join(np.random.choice(healthy_patterns, time_len * 5))[start:start + time_len])
        label_b.append(''.join(random_unhealthy))
    label_a = [to_emb(x) for x in label_a]
    label_b = [to_emb(x) for x in label_b]

    ya = t.tensor([[1.0, 0]] * data_count, device=tdevice)
    yb = t.tensor([[0, 1.0]] * data_count, device=tdevice)

    y = t.cat([ya, yb])

    def test(fp):  # validation set has no noise
        a = []
        b = []
        ref_s_a = ''.join(healthy_patterns)
        for i in range(50):
            # insert a 2 letter pattern not found in A
            neg_s = ''.join(np.random.choice(s, 2))
            while ref_s_a.__contains__(neg_s):
                neg_s = ''.join(np.random.choice(s, 2))

            np.random.seed(dt.datetime.now().microsecond)
            random_unhealthy = list(''.join(np.random.choice(healthy_patterns, time_len * 5)))
            start = np.random.randint(0, len(random_unhealthy) - time_len)

            random_unhealthy = random_unhealthy[start:start + time_len - 2]
            isert = np.random.randint(0, len(random_unhealthy))
            random_unhealthy.insert(isert, neg_s[1])
            random_unhealthy.insert(isert, neg_s[0])

            a.append(''.join(np.random.choice(healthy_patterns, time_len * 5))[start:start + time_len])
            b.append(''.join(random_unhealthy))
        ax = [to_emb(x) for x in a]
        bx = [to_emb(x) for x in b]

        ra = [fp(x)[0].data.item() for x in ax]
        rb = [fp(x)[0].data.item() for x in bx]

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
        layers = 2
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
        lstm_optim = t.optim.SGD(params, lr=0.06)

        def lstm_fprop2(x):
            # 30% noise, 69% acc, 100% TP, 39% TN, 2000 iter
            # 50% noise, 87% acc, 100% TP, 75.4% TN, 2000 iter, might be due to lesser healthy variety
            # 70% noise  70% acc, 96% TP, 43% TN, 2000 iter
            out, h = lstm(x.unsqueeze(1), init)
            ii = h[1].reshape(-1)
            r = last_w(ii).softmax(dim=0)  # 0 is h, 1 is c
            return r

        def one(f, lstm_optim):
            for i in range(2000):
                if i % 10 == 0:
                    loss = _h(f, lstm_optim, verbose=True)
                    print(i, '  Noise:', noise_level, ' Loss:', loss)
                else:
                    loss = _h(f, lstm_optim, verbose=False)

            r = [test(f) for i in range(50)]
            print('correct', np.mean(([x['correct'] for x in r])) / 100)
            print('tp', np.mean(([x['tp'] for x in r])) / 50)
            print('tn', np.mean(([x['tn'] for x in r])) / 50)
            return r

        r1 = one(lstm_fprop2, lstm_optim)
        return

    lstm_test()
    return


def noise_training_noisy_negatives_lstm_autoencoder():
    # neck 3
    # Validating against noise at  0
    #      Good mean: 0.0012862583389505744      Good max: 0.016436414793133736    Good min: 1.3219647598816664e-11
    #      Bad mean: 0.0010775895789265633      Bad max: 0.01479392871260643    Bad min: 9.606537787476555e-12
    # Validating against noise at  0.005
    #      Good mean: 0.001119917957112193      Good max: 0.01643643155694008    Good min: 9.555378710501827e-11
    #      Bad mean: 0.001019008457660675      Bad max: 0.014795759692788124    Bad min: 9.094947017729282e-11
    # Validating against noise at  0.01
    #      Good mean: 0.0012289904989302158      Good max: 0.016442973166704178    Good min: 2.220446049250313e-12
    #      Bad mean: 0.0011299269972369075      Bad max: 0.014795748516917229    Bad min: 2.4294255496215555e-10
    # Validating against noise at  0.1
    #      Good mean: 0.001111797639168799      Good max: 0.016234537586569786    Good min: 3.533578429859574e-10
    #      Bad mean: 0.001092425431124866      Bad max: 0.02461032196879387    Bad min: 9.208989126818778e-11
    # Validating against noise at  0.3
    #      Good mean: 0.0012299318332225084      Good max: 0.01644359901547432    Good min: 4.893286331686397e-10
    #      Bad mean: 0.001266053644940257      Bad max: 0.04200202599167824    Bad min: 4.83741047219155e-10
    # Validating against noise at  0.5
    #      Good mean: 0.001072052400559187      Good max: 0.016396580263972282    Good min: 6.177778288929403e-10
    #      Bad mean: 0.0014332759892567992      Bad max: 0.06241960451006889    Bad min: 7.927131173701696e-11
    # Validating against noise at  0.7
    #      Good mean: 0.0011095075169578195      Good max: 0.015226420946419239    Good min: 5.705678152168048e-10
    #      Bad mean: 0.001663834322243929      Bad max: 0.07727223634719849    Bad min: 6.296865251442796e-10

    healthy_patterns = ['123', '798']  # more variations mean higher entropy, harder to differentiate
    s = list('123456789')

    data_count = 100
    max_time_len = 80
    min_time_len = 20

    data = []

    emb_size = 8

    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    def _h(fp, op, verbose=True):
        out = [fp(x) for x in data]
        _y = t.stack([x[0] for x in out])
        y = t.stack([x[1] for x in out])

        loss = t.nn.MSELoss()(_y, y)

        loss.backward()
        op.step()
        op.zero_grad()

        return loss.data.cpu().item()

    for i in range(data_count):
        np.random.seed(dt.datetime.now().microsecond)
        start = np.random.randint(0, max_time_len * 4)
        data.append(''.join(np.random.choice(healthy_patterns, max_time_len * 5))[
                    start:start + np.random.randint(min_time_len, max_time_len)])
    data = [to_emb(x) for x in data]

    def test(fp):  # validation set has no noise
        test_count = 100

        for noise in [0, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7]:
            print('Validating against noise at ', noise)
            a = []
            b = []
            for i in range(test_count):
                np.random.seed(dt.datetime.now().microsecond)
                start = np.random.randint(0, max_time_len * 4)
                dd = np.random.choice(healthy_patterns, max_time_len * 5)[
                     start:start + np.random.randint(min_time_len, max_time_len)]
                for ii in range(len(dd)):
                    if np.random.rand() < noise:
                        dd[ii] = np.random.choice(s)
                b.append(''.join(dd))

                start = np.random.randint(0, max_time_len * 4)
                a.append(''.join(np.random.choice(healthy_patterns, max_time_len * 5))[
                         start:start + np.random.randint(min_time_len, max_time_len)])

            def _f(fp, d):
                _y, y = fp(to_emb(d))
                return t.pow(t.sub(_y, y), 2)

            ra = t.stack([_f(fp, x) for x in a])
            rb = t.stack([_f(fp, x) for x in b])

            _ = lambda x: x.data.cpu().item()
            print('     Good mean:', _(t.mean(ra)), '     Good max:', _(t.max(ra)), '   Good min:', _(t.min(ra)))
            print('     Bad mean:', _(t.mean(rb)), '     Bad max:', _(t.max(rb)), '   Bad min:', _(t.min(rb)))
        return

    def lstm_test():
        print('~~~~~~~~~~~~~~~~LSTM test~~~~~~~~~~~~~~~~~~~~')
        out_size = 10
        layers = 2
        neck = 3
        lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=out_size, batch_size=1,
                                                       num_of_layers=layers,
                                                       device=tdevice)
        neck_w = t.nn.Linear(out_size * layers, neck)
        neck_w.to(tdevice)

        last_w = t.nn.Linear(neck, out_size * layers)
        last_w.to(tdevice)

        params = [
            # {'params': list(lstm.parameters())},
            {'params': list(neck_w.parameters())},
            {'params': list(last_w.parameters())}
        ]
        lstm_optim = t.optim.SGD(params, lr=0.08)

        def lstm_fprop2(x):
            out, h = lstm(x.unsqueeze(1), init)
            h1 = h[1].reshape(-1)
            ii = neck_w(h1)
            r = last_w(ii)
            return r, h1

        def one(f, lstm_optim):
            for i in range(500):
                if i % 10 == 0:
                    loss = _h(f, lstm_optim, verbose=True)
                    print(i, ' Loss:', loss)
                else:
                    loss = _h(f, lstm_optim, verbose=False)

            r = test(f)
            return r

        r1 = one(lstm_fprop2, lstm_optim)
        return

    lstm_test()
    return


# undone
def auto_learning_rate_optimizer_test():
    data_count = 100
    data_len = 10

    a = t.rand([data_count, data_len], device=tdevice)
    b = t.rand([data_count, data_len], device=tdevice)

    ya = t.tensor([[1.0, 0]] * data_count, device=tdevice)
    yb = t.tensor([[0, 1.0]] * data_count, device=tdevice)

    y = t.cat([ya, yb])

    return





if __name__ == '__main__':
    models_similarity_test_on_same_data()
