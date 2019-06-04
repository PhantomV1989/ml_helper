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

    healthy_patterns = ['123', '747']  # more variations mean higher entropy, harder to differentiate
    s = list('1234567890')

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
        lstm_optim = t.optim.SGD(params, lr=0.1)

        def lstm_fprop2(x):
            # 30% noise, 69% acc, 100% TP, 39% TN, 2000 iter
            # 50% noise, 83% acc, 34% TP, 27% TN, 2000 iter
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
            print('tp', np.mean(([x['tp'] for x in r])) / 100)
            print('tn', np.mean(([x['tn'] for x in r])) / 100)
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
    noise_training_noisy_negatives(0.5)
    noise_training_noisy_negatives(0.7)
