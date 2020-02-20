import torch as t
import numpy as np
import time
import datetime as dt
import copy
import ml_helper
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import datetime as dt
import pickle
from sklearn.preprocessing import scale

tdevice = t.device('cpu')
sigmoid = lambda x: 1 / (1 + np.exp(-x))
isigmoid = lambda x: np.log(x / (1 - x))


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


# anomaly detection autuencoder AUC-ROC improvement test
def autoencoder_auc_roc_test():
    # 1 layer:
    #     Validating against noise at  0
    #         Good mean: 0.002284168265759945      Good max: 0.02517140470445156    Good min: 2.523217190741889e-10
    #         Bad mean: 0.0023619933053851128      Bad max: 0.028664903715252876    Bad min: 4.785114526839607e-10
    #     Validating against noise at  0.005
    #         Good mean: 0.0022366896737366915      Good max: 0.020402882248163223    Good min: 1.13411502411509e-09
    #         Bad mean: 0.0024241823703050613      Bad max: 0.019856005907058716    Bad min: 1.140144867406434e-09
    #     Validating against noise at  0.01
    #         Good mean: 0.0023201622534543276      Good max: 0.022928638383746147    Good min: 9.240608278560103e-10
    #         Bad mean: 0.0023004086688160896      Bad max: 0.017451390624046326    Bad min: 9.866916172995843e-10
    #     Validating against noise at  0.03
    #         Good mean: 0.002311129355803132      Good max: 0.020541377365589142    Good min: 5.374900524657278e-11
    #         Bad mean: 0.0024332781322300434      Bad max: 0.021967843174934387    Bad min: 1.0943246309125243e-11
    #     Validating against noise at  0.05
    #         Good mean: 0.002324569970369339      Good max: 0.020081238821148872    Good min: 6.050360212839223e-11
    #         Bad mean: 0.002443743636831641      Bad max: 0.02426290698349476    Bad min: 2.0463630789890885e-12
    #     Validating against noise at  0.07
    #         Good mean: 0.002308210590854287      Good max: 0.019881971180438995    Good min: 3.00048341728143e-09
    #         Bad mean: 0.0024800612591207027      Bad max: 0.02091060020029545    Bad min: 5.130118552187923e-10
    #     Validating against noise at  0.09
    #         Good mean: 0.002314807614311576      Good max: 0.022100992500782013    Good min: 1.5967316358000971e-10
    #         Bad mean: 0.002576353494077921      Bad max: 0.028380462899804115    Bad min: 1.524202986047385e-11
    #     Validating against noise at  0.1
    #         Good mean: 0.002326227957382798      Good max: 0.01722235232591629    Good min: 1.352479017668884e-09
    #         Bad mean: 0.002514583058655262      Bad max: 0.021465705707669258    Bad min: 1.2825296380469808e-10
    #     Validating against noise at  0.2
    #         Good mean: 0.002401921432465315      Good max: 0.02384854294359684    Good min: 2.3338140309192568e-09
    #         Bad mean: 0.0028238489758223295      Bad max: 0.03296823427081108    Bad min: 2.2027180079930986e-10
    #     Validating against noise at  0.3
    #         Good mean: 0.0022675232030451298      Good max: 0.02218928374350071    Good min: 4.0828140868143237e-10
    #         Bad mean: 0.0031791646033525467      Bad max: 0.031853653490543365    Bad min: 9.672262990534364e-13
    #     Validating against noise at  0.4
    #         Good mean: 0.0023205566685646772      Good max: 0.023025741800665855    Good min: 3.304911899704166e-12
    #         Bad mean: 0.0035575362853705883      Bad max: 0.03103339858353138    Bad min: 8.535394613318203e-11
    #     Validating against noise at  0.5
    #         Good mean: 0.0023653407115489244      Good max: 0.018190525472164154    Good min: 3.4583536034915596e-10
    #         Bad mean: 0.0038876456674188375      Bad max: 0.032337043434381485    Bad min: 5.143627745951562e-10
    #     Validating against noise at  0.6
    #         Good mean: 0.002331378636881709      Good max: 0.018727412447333336    Good min: 9.672262990534364e-11
    #         Bad mean: 0.004259637091308832      Bad max: 0.0367763452231884    Bad min: 3.482014676592371e-11
    #     Validating against noise at  0.7
    #         Good mean: 0.0023389742709696293      Good max: 0.02069368027150631    Good min: 3.0812774554078715e-10
    #         Bad mean: 0.004549822304397821      Bad max: 0.04007226228713989    Bad min: 1.2325109821631486e-09
    #     Validating against noise at  0.8
    #         Good mean: 0.002389859640970826      Good max: 0.02745024487376213    Good min: 3.864816022769446e-09
    #         Bad mean: 0.004518182948231697      Bad max: 0.03511567786335945    Bad min: 1.3787726516056864e-10
    #     Validating against noise at  0.9
    #         Good mean: 0.0023389847483485937      Good max: 0.024100642651319504    Good min: 1.0449330289930003e-10
    #         Bad mean: 0.005010637454688549      Bad max: 0.04340135306119919    Bad min: 1.667022075935165e-09
    #     Validating against noise at  1
    #         Good mean: 0.0023028424475342035      Good max: 0.02056620456278324    Good min: 8.530065542800003e-12
    #         Bad mean: 0.005462391301989555      Bad max: 0.042834993451833725    Bad min: 1.3046843605479808e-09
    #
    # 2 layers:
    #     Validating against noise at  0
    #         Good mean: 0.00037910198443569243      Good max: 0.008572401478886604    Good min: 6.913669636787745e-11
    #         Bad mean: 0.00032942279358394444      Bad max: 0.004621895030140877    Bad min: 2.936539900133539e-10
    #     Validating against noise at  0.005
    #         Good mean: 0.0003559112665243447      Good max: 0.008723326027393341    Good min: 1.6548762360457658e-09
    #         Bad mean: 0.0003805026353802532      Bad max: 0.006068185903131962    Bad min: 2.5421886817866834e-10
    #     Validating against noise at  0.01
    #         Good mean: 0.0004111455345991999      Good max: 0.010896891355514526    Good min: 1.6914469824769185e-11
    #         Bad mean: 0.00036199853639118373      Bad max: 0.006269656587392092    Bad min: 4.106937012693379e-12
    #     Validating against noise at  0.03
    #         Good mean: 0.0003638255875557661      Good max: 0.009856145828962326    Good min: 5.374900524657278e-11
    #         Bad mean: 0.0003517355944495648      Bad max: 0.00568230589851737    Bad min: 1.590056086797631e-09
    #     Validating against noise at  0.05
    #         Good mean: 0.000385304621886462      Good max: 0.006601903587579727    Good min: 1.6118306689349993e-10
    #         Bad mean: 0.000397569703636691      Bad max: 0.006537513807415962    Bad min: 1.2789769243681803e-13
    #     Validating against noise at  0.07
    #         Good mean: 0.0003362018906045705      Good max: 0.004029432777315378    Good min: 2.8141045049778768e-11
    #         Bad mean: 0.00036669656401500106      Bad max: 0.005232878960669041    Bad min: 1.9586332555832087e-11
    #     Validating against noise at  0.09
    #         Good mean: 0.0003177803591825068      Good max: 0.005671401973813772    Good min: 2.2027180079930986e-10
    #         Bad mean: 0.00041723239701241255      Bad max: 0.010139383375644684    Bad min: 6.568967592102126e-12
    #     Validating against noise at  0.1
    #         Good mean: 0.00035575151559896767      Good max: 0.007221526466310024    Good min: 4.191926805674484e-10
    #         Bad mean: 0.00036910100607201457      Bad max: 0.006101359613239765    Bad min: 2.877698079828406e-13
    #     Validating against noise at  0.2
    #         Good mean: 0.00044296792475506663      Good max: 0.006950298324227333    Good min: 1.2768461843393197e-09
    #         Bad mean: 0.0005105073214508593      Bad max: 0.010831589810550213    Bad min: 1.0027179087046534e-10
    #     Validating against noise at  0.3
    #         Good mean: 0.0003504917840473354      Good max: 0.005901690106838942    Good min: 2.751221472863108e-11
    #         Bad mean: 0.0006585303344763815      Bad max: 0.00912557728588581    Bad min: 3.1232705310912934e-10
    #     Validating against noise at  0.4
    #         Good mean: 0.00034039997262880206      Good max: 0.007053515408188105    Good min: 2.1393589122453704e-09
    #         Bad mean: 0.0008360185893252492      Bad max: 0.01025404129177332    Bad min: 6.87805368215777e-12
    #     Validating against noise at  0.5
    #         Good mean: 0.0003784925502259284      Good max: 0.00746538769453764    Good min: 4.2597925187237706e-11
    #         Bad mean: 0.0010167004074901342      Bad max: 0.008974318392574787    Bad min: 2.063451631784119e-10
    #     Validating against noise at  0.6
    #         Good mean: 0.00036307028494775295      Good max: 0.004207672085613012    Good min: 9.094947017729282e-11
    #         Bad mean: 0.0012349721509963274      Bad max: 0.013199139386415482    Bad min: 3.1391778065881226e-11
    #     Validating against noise at  0.7
    #         Good mean: 0.00035447964910417795      Good max: 0.007389870006591082    Good min: 1.035971308738226e-11
    #         Bad mean: 0.0014148644404485822      Bad max: 0.015594497323036194    Bad min: 3.3921665476555063e-10
    #     Validating against noise at  0.8
    #         Good mean: 0.00036639513564296067      Good max: 0.007503138389438391    Good min: 7.469580509678053e-11
    #         Bad mean: 0.0017347825923934579      Bad max: 0.012979328632354736    Bad min: 7.847944516470307e-12
    #     Validating against noise at  0.9
    #         Good mean: 0.0003706445568241179      Good max: 0.008320853114128113    Good min: 5.684341886080802e-14
    #         Bad mean: 0.0019319429993629456      Bad max: 0.015653399750590324    Bad min: 1.6187051699034782e-11
    #     Validating against noise at  1
    #         Good mean: 0.0003708285803440958      Good max: 0.007267403416335583    Good min: 2.686739719592879e-12
    #         Bad mean: 0.0021194927394390106      Bad max: 0.018723754212260246    Bad min: 2.2737367544323206e-11
    #
    # 3 layers:
    #     Validating against noise at  0
    #         Good mean: 0.0005001783138141036      Good max: 0.007299751043319702    Good min: 8.15427725342488e-11
    #         Bad mean: 0.0004264674207661301      Bad max: 0.0066557894460856915    Bad min: 3.1391778065881226e-11
    #     Validating against noise at  0.005
    #         Good mean: 0.00043954598368145525      Good max: 0.0061861551366746426    Good min: 1.2789769243681803e-11
    #         Bad mean: 0.0004141135432291776      Bad max: 0.005062348674982786    Bad min: 1.255671122635249e-10
    #     Validating against noise at  0.01
    #         Good mean: 0.00043465467751957476      Good max: 0.006969059817492962    Good min: 1.2960654771632107e-10
    #         Bad mean: 0.0004722992598544806      Bad max: 0.007058221846818924    Bad min: 3.2724845056009144e-10
    #     Validating against noise at  0.03
    #         Good mean: 0.0004873007710557431      Good max: 0.006805243901908398    Good min: 2.2115731468375088e-10
    #         Bad mean: 0.000508435012307018      Bad max: 0.005720475222915411    Bad min: 2.936539900133539e-10
    #     Validating against noise at  0.05
    #         Good mean: 0.00047728241770528257      Good max: 0.006311750039458275    Good min: 5.252340784522858e-10
    #         Bad mean: 0.0005190390511415899      Bad max: 0.00865363422781229    Bad min: 2.6284396881237626e-10
    #     Validating against noise at  0.07
    #         Good mean: 0.0004925963003188372      Good max: 0.006627635098993778    Good min: 3.304911899704166e-10
    #         Bad mean: 0.0005015972419641912      Bad max: 0.005633826367557049    Bad min: 5.103153455365828e-10
    #     Validating against noise at  0.09
    #         Good mean: 0.00044547009747475386      Good max: 0.007177541498094797    Good min: 3.84549281307045e-10
    #         Bad mean: 0.000554045254830271      Bad max: 0.007008823566138744    Bad min: 1.581703656938771e-10
    #     Validating against noise at  0.1
    #         Good mean: 0.00044216043897904456      Good max: 0.004791925195604563    Good min: 1.6422418980255316e-10
    #         Bad mean: 0.0005635041161440313      Bad max: 0.01106132660061121    Bad min: 5.130118552187923e-12
    #     Validating against noise at  0.2
    #         Good mean: 0.000456175155704841      Good max: 0.006257114000618458    Good min: 1.879385536085465e-10
    #         Bad mean: 0.0007300454890355468      Bad max: 0.007532994728535414    Bad min: 1.8366144161063858e-09
    #     Validating against noise at  0.3
    #         Good mean: 0.0004109722503926605      Good max: 0.005219771061092615    Good min: 5.571223482547794e-10
    #         Bad mean: 0.0010315965628251433      Bad max: 0.011555517092347145    Bad min: 7.51754214434186e-12
    #     Validating against noise at  0.4
    #         Good mean: 0.000458752503618598      Good max: 0.008390052244067192    Good min: 3.2724845056009144e-10
    #         Bad mean: 0.001161719672381878      Bad max: 0.016884248703718185    Bad min: 1.5102727957128081e-09
    #     Validating against noise at  0.5
    #         Good mean: 0.0004660885315388441      Good max: 0.007424922659993172    Good min: 2.765467854715098e-10
    #         Bad mean: 0.0014560545096173882      Bad max: 0.014032334089279175    Bad min: 2.4356836547667626e-09
    #     Validating against noise at  0.6
    #         Good mean: 0.00044111619354225695      Good max: 0.005438199266791344    Good min: 2.0463630789890885e-12
    #         Bad mean: 0.0016458937898278236      Bad max: 0.015526636503636837    Bad min: 2.494893180937652e-10
    #     Validating against noise at  0.7
    #         Good mean: 0.00046901137102395296      Good max: 0.010707736946642399    Good min: 1.5076295767357806e-10
    #         Bad mean: 0.0020034858025610447      Bad max: 0.015731081366539    Bad min: 3.206324095117452e-11
    #     Validating against noise at  0.8
    #         Good mean: 0.0004416914307512343      Good max: 0.004969323519617319    Good min: 4.863309754910006e-11
    #         Bad mean: 0.0021800368558615446      Bad max: 0.02151661366224289    Bad min: 1.711114805402758e-09
    #     Validating against noise at  0.9
    #         Good mean: 0.00046305719297379255      Good max: 0.007389080710709095    Good min: 4.452118673725636e-10
    #         Bad mean: 0.002386486856266856      Bad max: 0.02204880490899086    Bad min: 3.1125182431424037e-09
    #     Validating against noise at  1
    #         Good mean: 0.00048249828978441656      Good max: 0.005267511587589979    Good min: 1.6914469824769185e-11
    #         Bad mean: 0.0028613535687327385      Bad max: 0.02369607985019684    Bad min: 2.6478019776732253e-10

    data_count = 700
    max_time_len = 600
    min_time_len = 300

    tanh_scale = 0.03

    data = []

    emb_size = 20

    ap = np.power(np.random.rand(10, 10), 2)
    at = 10 * np.random.rand(10, 10)

    bp = np.power(np.random.rand(10, 10), 2)
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

    def _h(fp, op):
        out = [fp(x) for x in data]
        _y = t.stack([x[0] for x in out])
        y = t.stack([x[1] for x in out])

        loss = t.nn.MSELoss()(_y, y)

        loss.backward()
        op.step()
        op.zero_grad()

        return loss.data.cpu().item()

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

    class AE:
        @staticmethod
        def run(layers, cycle):
            en, de, tp = AE.create_autoencoder(emb_size + 1, neck_size=3, layers=layers)
            op = t.optim.SGD(tp, lr=0.02)

            def _(d):
                _y = AE.fprop_ae(x_input=d, encoders=en, decoders=de).mean(dim=0)
                y = d.mean(dim=0)
                return _y, y

            for i in range(cycle):
                l = _h(_, op)
                print(layers, '  ', i, '   ', l)

            test(_)
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

    AE.run(layers=1, cycle=500)
    AE.run(layers=2, cycle=500)
    AE.run(layers=3, cycle=500)
    return


# ae clustering for muliple distributions
def multi_ae_clustering_test():
    data_count = 300
    max_time_len = 50
    min_time_len = 1

    tanh_scale = 0.03

    data = []
    s = list('0123456789')
    emb_size = 20
    emb = t.rand([10, emb_size], device=tdevice)

    def create_pattern_distribution():
        events = np.power(np.random.rand(10, 10), 3)
        times = 10 * np.random.rand(10, 10)
        return events, times

    def to_emb(x):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in x[0]]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        time = t.tensor(x[1], dtype=t.float32, device=tdevice)
        f = []
        for i in range(len(x[0])):
            f.append(t.cat([a[i], time[i].unsqueeze(0)]))
        f = t.stack(f)
        return f

    # the original 'healthy' data contains 2 types of distribution
    A, B = create_pattern_distribution(), create_pattern_distribution()

    # C shall be the 'unhealthy' distribution
    C = create_pattern_distribution()

    save('data_generators', [A, B, C])

    def _h(fp, op, verbose=True):
        out = [fp(x) for x in data]
        _y = t.stack([x[0] for x in out])
        y = t.stack([x[1] for x in out])

        loss = t.nn.MSELoss()(_y, y)

        loss.backward()
        op.step()
        op.zero_grad()

        return loss.data.cpu().item()

    def gen_data_sets(pt_list, cum_p):
        '''

        :param pt_list: [a,b,c] 3 types of distributions
        :param cum_p: [0.5 0.7, 1] cumulative probabilities of sampling a particular distribution
        :return:
        '''
        xs = [np.random.choice(s)]
        xt = [0]
        curr = 0
        while curr < max_time_len:
            np.random.seed(dt.datetime.now().microsecond)
            if curr > len(xt) - 1:
                return gen_data_sets(pt_list, cum_p)
            ct = xt[curr]
            cl = int(xs[curr])

            rng = np.random.rand()
            for i, v in enumerate(cum_p):
                if rng <= v:
                    dist_p, dist_time = pt_list[i]
                    break

            xpp = dist_p[cl]
            xpt = dist_time[cl]
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

    def healthy_data():
        return gen_data_sets([A], [1]) if np.random.rand() < 0.6 else gen_data_sets([B], [1])

    def unhealthy_data(noise):
        h = 1 - noise
        return gen_data_sets([A, C], [h, 1]) if np.random.rand() < 0.6 else gen_data_sets([B, C], [h, 1])

    for i in range(data_count):
        data.append(healthy_data())

    data = [to_emb(x) for x in data]

    def test(fp, mean, std):  # validation set has no noise
        test_count = 100

        for noise in [0, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            print('Validating against noise at ', noise)
            a = [healthy_data() for x in range(test_count)]
            b = [unhealthy_data(noise) for x in range(test_count)]

            def _f(fp, d):
                _y, y = fp(to_emb(d))
                return t.pow(t.sub(_y, y), 2)

            ra = t.stack([_f(fp, x) for x in a])
            rb = t.stack([_f(fp, x) for x in b])

            _ = lambda x: np.abs((x.data.cpu().item() - mean) / std)
            print('     Good mean:', _(t.mean(ra)), '     Good max:', _(t.max(ra)), '   Good min:', _(t.min(ra)))
            print('     Bad mean:', _(t.mean(rb)), '     Bad max:', _(t.max(rb)), '   Bad min:', _(t.min(rb)))
        return

    def create_ae():
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
        return w1, w2, optim

    def ann_focus_concat_test(xx):

        w1, w2, optim = create_ae()

        def fprop(x):
            o1 = x.mean(dim=0)
            focus = x[-1]
            o1 = t.cat([o1, focus])
            o = w1(o1).tanh()
            o = w2(o)
            return o, o1

        time_a = dt.datetime.now()
        losses = []
        for i in range(xx):
            loss = _h(fprop, optim, verbose=True)
            losses.append(loss)
            if i % 100 == 0:
                print(i, ' Loss:', loss)
        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))
        losses = np.asarray(losses[-100:])
        print(np.histogram(losses, bins=10)[0])
        r = test(fprop, losses.mean(), losses.std())
        return

    def ann_focus_concat_with_clustering_test(xx):
        ae_set = []
        while True:
            w1, w2, optim = create_ae()
            ae_set.append([w1, w2, optim])

            def fprop(x):
                o1 = x.mean(dim=0)
                focus = x[-1]
                o1 = t.cat([o1, focus])
                o = w1(o1).tanh()
                o = w2(o)
                return o, o1

            losses, cnt = [], 0
            while True:
                cnt += 1
                loss = _h(fprop, optim, verbose=True)
                losses.append(loss)
                if len(losses) > 1000:
                    losses.pop(0)
                if cnt % 100 == 0:
                    print(cnt, ' Loss:', loss)
                if len(losses) >= 1000 and cnt % 100 == 0:
                    l = np.asarray(losses)
                    if l[50:].mean() > l[:50].mean():
                        print(np.histogram(l, bins=10)[0])
                        break

            losses = np.asarray(losses)
            mn, std = np.histogram(losses)

            def _x(d):
                _y, y = fprop(d)
                loss = t.nn.MSELoss()(_y, y)
                return loss.data.cpu().item()

            ldata = [[_x(d), d] for d in data]
            new_data = []
            for d in ldata:
                if np.abs(d[0] - mn) / std > 3:
                    new_data.append(d[1])
            if len(new_data) == 0:
                save('asd', ae_set)
                break
            else:
                data = new_data

        time_b = dt.datetime.now()
        print('Finished training in ', (time_b - time_a))
        losses = np.asarray(losses[-100:])
        print(np.histogram(losses, bins=10)[0])
        r = test(fprop, losses.mean(), losses.std())
        return

    # ann_focus_concat_test(2000)
    ann_focus_concat_with_clustering_test(2000)
    return


def aekmeans_vs_kmeans_clustering_test():  # failed
    '''
    Objective is to compare the clustering performance between ae+kmeans and just kmeans only
    0 Noise: 0  A: [1.48, 1.6, 2.66, 0.0]  B: [1.76, 0.72, 1.02, 0.0]  C: [1.76, 2.1, 1.66, 0.6]  D: [1.52, 1.18, 1.68, 0.6]
    0 Noise: 0.3  A: [2.26, 1.78, 1.28, 0.58]  B: [2.44, 1.26, 0.34, 0.44]  C: [1.64, 2.22, 0.92, 1.0]  D: [1.08, 1.72, 0.58, 1.5]
    0 Noise: 0.6  A: [2.08, 1.28, 1.7, 0.96]  B: [2.04, 1.3, 0.9, 1.2]  C: [0.98, 1.68, 1.88, 1.56]  D: [1.4, 1.48, 1.5, 1.22]
    0 Noise: 0.9  A: [1.2, 1.3, 1.2, 1.16]  B: [0.94, 1.26, 1.28, 1.1]  C: [1.42, 1.4, 1.98, 1.62]  D: [1.08, 1.32, 1.28, 1.22]
    :return:
    '''
    from sklearn.cluster import KMeans
    noises = [0, 0.3, 0.6, 0.9]
    xlen = 30
    data_info = [
        {
            'pattern': "5367",
            "size": 50
        },
        {
            'pattern': "34579",
            "size": 50
        },
        {
            'pattern': "958",
            "size": 50
        },
        {
            'pattern': "46",
            "size": 50
        }
    ]
    emb_size = 10
    encs = 50
    neck = 5
    epoch = 50
    emb = t.rand([10, emb_size], device=tdevice)

    def to_emb(a):
        a = [t.tensor([int(x) for x in y], device=tdevice) for y in a]
        a = t.cat([emb.index_select(dim=0, index=x) for x in a])
        return a

    def gen_data(di, noise, data_length):
        data = []
        for _d in di:
            for i in range(_d['size']):
                _tdata = ''
                while len(_tdata) < 2 * data_length:
                    if np.random.rand() > noise:
                        _tdata += _d['pattern']
                    else:
                        _tdata += str(np.random.randint(0, 10))
                offset = np.random.randint(data_length - 1)
                _tdata = _tdata[offset:offset + data_length]
                data.append(_tdata)
        return data

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

    def fprop_ae(x_input, encoders=[], decoders=[], act=lambda x: t.mul(x, 1)):
        if encoders:
            for m in encoders:
                x_input = act(m(x_input))
        if decoders:
            for m in decoders:
                x_input = act(m(x_input))
        return x_input

    def eval_clus(_emb_data, kmeans, mode=1):
        if mode == 1:
            t = []
            c, start = 0, 0
            clu = kmeans.predict(_emb_data)
            for d in data_info:
                for i in range(d['size']):
                    c += clu[start + i]
                t.append(c / d['size'])
                c = 0
                start = start + i + 1
            return t
        elif mode == 2:
            asd = 5
        return

    def _h(_emb_data):
        _emb_data = t.stack(_emb_data)
        _emb_data = _emb_data.cpu().data.numpy()
        kmeans = KMeans(n_clusters=len(data_info), random_state=0).fit(_emb_data)
        kmeansresult = eval_clus(_emb_data, kmeans)
        return kmeansresult

    for noise in noises:
        hpos = 1  # 0 is hidden, 1 is cell
        lstm1, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=encs, batch_size=1,
                                                        num_of_layers=1, device=tdevice)

        _data = gen_data(data_info, noise, xlen)
        _datax = [x[:-1] for x in _data]
        _datay = [x[1:] for x in _data]
        data = [to_emb(d) for d in _data]
        _emb_data = []
        for x in data:
            out, hid = lstm1(x.unsqueeze(1), init)
            _emb_data.append(hid[hpos].squeeze())
        kmeansresult0 = _h(_emb_data)

        # train ae ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        en, de, tp = create_autoencoder(encs, neck_size=neck, layers=3)
        op = t.optim.SGD(tp, lr=0.01)
        for i in range(30):
            _aed = [fprop_ae(d, en, de) for d in _emb_data]
            _y = t.stack(_aed)
            y = t.stack(_emb_data)

            loss = t.nn.MSELoss()(_y, y)

            loss.backward(retain_graph=True)
            op.step()
            op.zero_grad()

        _emb_data1 = []
        for x in data:
            out, hid = lstm1(x.unsqueeze(1), init)
            f1 = fprop_ae(hid[hpos].squeeze(), en)
            _emb_data1.append(f1)
        kmeansresult1 = _h(_emb_data1)

        # train ae with lstm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        en, de, tp = create_autoencoder(encs, neck_size=neck, layers=3)
        tp.append({'params': lstm1.parameters()})
        op = t.optim.SGD(tp, lr=0.01)

        def _hfprop(x, en, de):
            out, hid = lstm1(x.unsqueeze(1), init)
            _aed = fprop_ae(hid[hpos].squeeze(), en, de)
            return _aed

        for i in range(epoch):
            _aed = [_hfprop(d, en, de) for d in data]
            _y = t.stack(_aed)
            y = t.stack(_emb_data)

            loss = t.nn.MSELoss()(_y, y)

            loss.backward(retain_graph=True)
            op.step()
            op.zero_grad()

        _emb_data1 = [_hfprop(d, en, []) for d in data]
        kmeansresult2 = _h(_emb_data1)

        # lstm offset only ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lstm2, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=emb_size, batch_size=1,
                                                        num_of_layers=1, device=tdevice)
        op = t.optim.SGD(lstm2.parameters(), lr=0.02)

        for i in range(epoch):
            _out = [lstm2(to_emb(d).unsqueeze(1), init) for d in _datax]  # out, hid
            _y = t.stack([o[0].squeeze() for o in _out])
            y = t.stack([to_emb(d) for d in _datay])

            loss = t.nn.MSELoss()(_y, y)

            loss.backward(retain_graph=True)
            op.step()
            op.zero_grad()

        _emb_data2 = [lstm2(d.unsqueeze(1), init)[1][hpos].squeeze() for d in data]
        kmeansresult3 = _h(_emb_data2)
        print(hpos, "Noise:", noise, ' A:', kmeansresult0, ' B:', kmeansresult1, ' C:', kmeansresult2, ' D:',
              kmeansresult3)
    return


def lstm_soft_tokenizer_test2_no_bprop():
    '''
    This test uses different levels of noise to see how it affects cluster distance
    Trained:
    [0, 0.02631144, '4122331123233112']
    [0, 0.090461254, '141223']
    [0, 0.08847344, '1412233112323311223']
    [0, 0.026971348, '132423434141223311']
    [0, 0.06214873, '2233314122']
    [0, 0.06693762, '33113242343414122']
    [0, 0.06846356, '11324234341']
    [0, 0.026988888, '1132423434141223311']
    [0, 0.08162957, '23311322333141223']
    [0, 0.03473699, '34311322343311322']
    [1, 0.0787026, '7779988777998887']
    [1, 0.0205024, '99878777998']
    [1, 0.046467982, '99888779988']
    [1, 0.05154396, '777998877799888']
    [1, 0.0372857, '7799887779988877998']
    [0, 0.154383, '88778997888777']
    [1, 0.021930618, '99887799878777998']
    [1, 0.055801775, '8877899']
    [1, 0.112696275, '9788877779']
    [1, 0.020577176, '7799878777998']
    Noise: 0
    [0, 0.14843024, '112233112233']
    [0, 0.035121046, '2233311322']
    [0, 0.06742181, '13242331123243341']
    [1, 0.04365811, '77989878779988']
    [1, 0.0823978, '78877799887']
    [1, 0.010371991, '98787799888779798']
    Noise: 0.3
    [0, 0.0205399, '2234311']
    [0, 0.19889672, '423311223333']
    [0, 0.19180232, '223333']
    [0, 0.15321185, '87779978877']
    [1, 0.13129285, '77997887775878']
    [1, 0.0555881, '758782979788878']
    Noise: 0.6
    [1, 0.11507003, '333112323334814']
    [0, 0.09610344, '4112233311232333481']
    [0, 0.110246696, '242333338111312234']
    [0, 0.07479594, '9878776']
    [1, 0.025162404, '79988777998']
    [0, 0.07717113, '39899878776']
    Noise: 0.9
    [0, 0.07466697, '1223347039114223']
    [0, 0.10580783, '34684112234']
    [0, 0.060882155, '39114223333468411']
    [1, 0.059285812, '9887799']
    [1, 0.044279788, '77863079988']
    [1, 0.032780427, '8877797988']
    :return:
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
        pattern = '998877'
        alter_pattern = '78'
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
                dstring += str(np.random.randint(10000, 100000))
            dstring += gt()
        return dstring[:dlen]

    a = gen_string(token_a, 0)
    b = gen_string(token_b, 0)
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=4, batch_size=1,
                                                   num_of_layers=1, device=tdevice)

    def __q(k, cnt):
        raw_d = []
        emb_data = []
        for i in range(cnt):
            sizee = np.random.randint(7, 20)
            poss = np.min([np.random.randint(dlen), dlen - sizee])
            sa = k[poss:np.min([poss + sizee, len(a) - 1])]
            raw_d.append(sa)
        return raw_d

    raw_d = []
    raw_d += __q(a, 10)
    raw_d += __q(b, 10)

    emb_data = [to_emb(x) for x in raw_d]
    emb_data = [lstm(x.unsqueeze(1), init)[1][0].squeeze() for x in emb_data]

    emb_data = t.stack(emb_data)
    emb_data = emb_data.cpu().data.numpy()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(emb_data)
    kmeansresult = ml_helper.calculate_kmeans_l2_dist(emb_data, kmeans)
    result = []
    print('Trained:')
    for i in range(len(raw_d)):
        dd = [kmeansresult[i][0], kmeansresult[i][1], raw_d[i]]
        print(dd)
        result.append(dd)

    for noise in noises:
        print('Noise:', noise)
        a = gen_string(token_a, noise)
        b = gen_string(token_b, noise)

        raw_d = []
        raw_d += __q(a, 3)
        raw_d += __q(b, 3)

        emb_data = [to_emb(x) for x in raw_d]
        emb_data = [lstm(x.unsqueeze(1), init)[1][0].squeeze() for x in emb_data]

        emb_data = t.stack(emb_data)
        emb_data = emb_data.cpu().data.numpy()

        kmeansresult = ml_helper.calculate_kmeans_l2_dist(emb_data, kmeans)
        for i in range(len(raw_d)):
            dd = [kmeansresult[i][0], kmeansresult[i][1], raw_d[i]]
            print(dd)
            result.append(dd)

    return result


def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return


def untrained_lstm_clustering2():
    import plotly.graph_objects as go
    fig = go.Figure()
    emb_size = 5
    s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
    char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
                                                   device=tdevice)

    def gen_data_a():
        s1 = '1234567890'
        s2 = 'abc'
        data = []
        for i in range(50):
            _s = ''
            for ii in range(10):
                if np.random.rand() < i / 100:
                    _s += np.random.choice(list(s2))
                else:
                    _s += s1[ii]
            data.append(_s)
        return data

    def gen_data_b():
        s1 = 'abcdefghij'
        s2 = '123'
        data = []
        for i in range(50):
            _s = ''
            for ii in range(10):
                if np.random.rand() < i / 100:
                    _s += np.random.choice(list(s2))
                else:
                    _s += s1[ii]
            data.append(_s)
        return data

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

    def _h(gdata):
        data = gdata()
        data1 = [s[::-1] for s in data]
        data_emb = [str_to_emb(x) for x in data1]

        lstmc_hid_outputs = [lstm(x.unsqueeze(1), init)[1][0].squeeze().cpu().data.numpy() for x in data_emb]
        x = [x[0] for x in lstmc_hid_outputs]
        y = [x[1] for x in lstmc_hid_outputs]
        z = [x[2] for x in lstmc_hid_outputs]
        return x, y, z, data

    x1, y1, z1, n1 = _h(gen_data_a)

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

    x2, y2, z2, n2 = _h(gen_data_b)

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


def untrained_lstm_clustering():
    '''
    Conclusion: Good, but untrained lstm is only sensitive to the most recent events, so need reverse order
    Trained lstm attempts to converge everything to singularity, bad
    RNN fails to even separate events at all
    :return:
    '''
    import plotly.graph_objects as go
    fig = go.Figure()
    emb_size = 5
    s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
    char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
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
        with open(path, 'r') as f:
            data = f.readlines()
        # data = [x[:-1] for x in data]  # remove last char \n for a start
        data = data[5:lim]
        data1 = [s[::-1] for s in data]
        data_emb = [str_to_emb(x) for x in data1]

        lstmc_hid_outputs = [lstm(x.unsqueeze(1), init)[1][0].squeeze().cpu().data.numpy() for x in data_emb]
        x = [x[0] for x in lstmc_hid_outputs]
        y = [x[1] for x in lstmc_hid_outputs]
        z = [x[2] for x in lstmc_hid_outputs]
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


def a():
    '''
    Note! The axis are not uniform in scale
    :return:
    '''
    import plotly.graph_objects as go
    import os
    krng = 0.05
    xname = 0.5
    gz = 0.03

    fig = go.Figure()

    def get_anomaly_score(point, data, vector_field, grid_size, x=0.5):
        plot_l
        scores = []
        mx = np.arctanh(0.99) * 0.3 * grid_size
        l2 = np.linalg.norm
        acom_x = []
        for i, d in enumerate(data):
            v = d - point
            l2d = l2(v)
            if l2d < grid_size * 5:  # co-direct 1, oppo -1, right-angles = 0
                vpt = vector_field[i]
                angle = np.arccos(np.dot(v, vpt) / l2(vpt) / l2(v))
                angle = np.pi / 2 if np.isnan(angle) else angle
                manifold_reducer = 1 - np.tanh(np.abs(np.tan(angle)))  # 1 means max reducing, 0 means no reducing

                total_reducer = x * manifold_reducer + 1 - x
                dist_total_reducer = np.tanh(mx * total_reducer / l2d)
                score = 1 - dist_total_reducer
                acom_x.append([manifold_reducer, dist_total_reducer, angle / np.pi * 180, total_reducer, l2d])
                scores.append(score)
        tscore = np.min(scores) if len(scores) > 0 else 1.0
        tscore = np.power(tscore, 5.14 * np.power(len(scores), -0.5139))
        return tscore

    def normalize_coordinates(dataset):
        minMax = []
        for i in range(dataset.shape[1]):
            v = np.copy(dataset[:, i])
            dataset[:, i] = (v - v.min()) / (v.max() - v.min())
            minMax.append([v.min(), v.max() - v.min()])

        def f(x):
            for i in range(x.shape[1]):
                x[:, i] = (x[:, i] - minMax[i][0]) / minMax[i][1]
            return x

        return dataset, f

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

    def get_principle_components(dataset, density_interval=0.005, min_size=0):
        axis_cnt = np.shape(dataset)[1]
        dense_labels, dense_partitions = get_density_partitions(dataset, density_interval=density_interval,
                                                                min_size=min_size)
        pc_bucket = np.zeros_like(dataset)
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
            max_axis_range = [[cluster_points[:, x].min(), cluster_points[:, x].max()] for x in range(axis_cnt)]
            max_range = np.linalg.norm(
                np.asarray([x[0] for x in max_axis_range]) - np.asarray([x[1] for x in max_axis_range]))
            rng = max_range * krng
            point_bucket = [[] for x in cluster_points]

            for i in range(len(cluster_points) - 1):
                cpoint = cluster_points[i]
                for ii in range(i + 1, len(cluster_points)):
                    npoint = cluster_points[ii]
                    if np.linalg.norm(cpoint - npoint) < rng:  # fills 2 clusters simultaneusly
                        point_bucket[i].append(npoint)
                        point_bucket[ii].append(cpoint)
            den = np.max([len(x) for x in point_bucket])
            for bi in range(len(point_bucket)):
                if len(point_bucket[bi]) > 0:
                    points = np.asarray(point_bucket[bi])
                    pc = PCA(n_components=1).fit(np.asarray(points)).components_  # alrddy len 1
                    pc_bucket[cluster_pos[bi]] = pc * len(point_bucket[bi]) / den

        _debug = []
        for i in range(len(dataset)):
            _debug.append(dataset[i])
            _debug.append(dataset[i] + pc_bucket[i] / 10)
            _debug.append([None] * axis_cnt)
        plot_l(_debug)
        return pc_bucket

    def plot_l(dataset):
        x = [x[0] for x in dataset]
        y = [x[1] for x in dataset]
        z = [x[2] for x in dataset]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines', line=dict(color='red', width=1),
        ))
        return

    def apply_labels(dataset, labels):
        ldata = {}
        for i, d in enumerate(dataset):
            label = labels[i]
            if not label in ldata:
                ldata[label] = [d]
            else:
                ldata[label].append(d)
        return ldata

    def rrgb():
        return 'rgb(' + str(np.random.randint(0, 255)) + ',' + str(np.random.randint(0, 255)) + ',' + str(
            np.random.randint(0, 255)) + ')'

    def plot_d(dataset, name, htext):
        x = [x[0] for x in dataset]
        y = [x[1] for x in dataset]
        z = [x[2] for x in dataset]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            hovertext=htext, name=name, mode='markers', hoverinfo='text',
            marker=dict(size=8, color=rrgb(), opacity=0.9,
                        line=dict(width=1, color='black'))
        ))
        return

    with open('./test_data', 'rb') as f:
        data, names = pickle.load(f)

    data, foo = normalize_coordinates(data)
    plot_d(data, 'data', names)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    rand_points = []  #
    rand_scores = []
    nvectors = get_principle_components(data, krng)
    for i in [0.2]:  # np.arange(0.27, 0.32, 0.01):
        for j in np.arange(0, 1, 0.03):
            for k in np.arange(0, 1, 0.03):
                new_point = [i, j, k]
                score = get_anomaly_score(new_point, data, nvectors, grid_size=gz, x=xname)
                qwe = 1
                if score < qwe:
                    rand_scores.append(score)
                    rand_points.append(new_point)

    for i in np.arange(0, 1, 0.04):  # np.arange(0.27, 0.32, 0.01):
        for j in [0.7]:
            for k in np.arange(0, 1, 0.02):
                new_point = [i, j, k]
                score = get_anomaly_score(new_point, data, nvectors, grid_size=gz, x=xname)
                qwe = 1
                if score < qwe:
                    rand_scores.append(score)
                    rand_points.append(new_point)

    x = [x[0] for x in rand_points]
    y = [x[1] for x in rand_points]
    z = [x[2] for x in rand_points]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        hovertext=[str(x) for x in rand_scores], name=str(krng) + '-' + str(xname) + '-' + str(gz), mode='markers',
        marker=dict(size=6, color=np.asarray(rand_scores), opacity=0.8, colorscale='RdYlGn', reversescale=True)
    ))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

    fig.show()
    return


def b():
    '''
    Note! The axis are not uniform in scale
    :return:
    '''
    import plotly.graph_objects as go
    krng = 0.1
    xname = 0.6
    gz = 0.03

    fig = go.Figure()

    def get_anomaly_score(point, data, vector_field, grid_size, x=0.5):
        plot_l
        scores = []
        mx = np.arctanh(0.99) * 0.3 * grid_size
        l2 = np.linalg.norm
        acom_x = []
        for i, d in enumerate(data):
            v = d - point
            l2d = l2(v)
            if l2d < grid_size * 5:  # co-direct 1, oppo -1, right-angles = 0
                vpt = vector_field[i]
                angle = np.arccos(np.dot(v, vpt) / l2(vpt) / l2(v))
                angle = np.pi / 2 if np.isnan(angle) else angle
                manifold_reducer = 1 - np.tanh(np.abs(np.tan(angle)))  # 1 means max reducing, 0 means no reducing

                total_reducer = x * manifold_reducer + 1 - x
                dist_total_reducer = np.tanh(mx * total_reducer / l2d)
                score = 1 - dist_total_reducer
                acom_x.append([manifold_reducer, dist_total_reducer, angle / np.pi * 180, total_reducer, l2d])
                scores.append(score)
        tscore = np.min(scores) if len(scores) > 0 else 1.0
        tscore = np.power(tscore, 1.5 * np.power(len(scores), -0.92) + 0.5)
        return tscore

    def normalize_coordinates(dataset):
        minMax = []
        for i in range(dataset.shape[1]):
            v = np.copy(dataset[:, i])
            dataset[:, i] = (v - v.min()) / (v.max() - v.min())
            minMax.append([v.min(), v.max() - v.min()])

        def f(x):
            for i in range(x.shape[1]):
                x[:, i] = (x[:, i] - minMax[i][0]) / minMax[i][1]
            return x

        return dataset, f

    def get_principle_components(dataset, rng):
        axis_cnt = np.shape(dataset)[1]
        pc_bucket = np.zeros_like(dataset)

        point_bucket = [[] for x in dataset]

        for i in range(len(dataset) - 1):
            cpoint = dataset[i]
            for ii in range(i + 1, len(dataset)):
                npoint = dataset[ii]
                if np.linalg.norm(cpoint - npoint) < rng:  # fills 2 clusters simultaneusly
                    point_bucket[i].append(npoint)
                    point_bucket[ii].append(cpoint)
        den = np.max([len(x) for x in point_bucket])
        for bi in range(len(point_bucket)):
            if len(point_bucket[bi]) > 0:
                points = np.asarray(point_bucket[bi])
                pc = PCA(n_components=1).fit(np.asarray(points)).components_  # alrddy len 1
                pc_bucket[bi] = pc * len(point_bucket[bi]) / den

        _debug = []
        for i in range(len(dataset)):
            _debug.append(dataset[i])
            _debug.append(dataset[i] + pc_bucket[i] / 10)
            _debug.append([None] * axis_cnt)
        plot_l(_debug)
        return pc_bucket

    def plot_l(dataset):
        x = [x[0] for x in dataset]
        y = [x[1] for x in dataset]
        z = [x[2] for x in dataset]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines', line=dict(color='red', width=1),
        ))
        return

    def rrgb():
        return 'rgb(' + str(np.random.randint(0, 255)) + ',' + str(np.random.randint(0, 255)) + ',' + str(
            np.random.randint(0, 255)) + ')'

    def plot_d(dataset, name, htext):
        x = [x[0] for x in dataset]
        y = [x[1] for x in dataset]
        z = [x[2] for x in dataset]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            hovertext=htext, name=name, mode='markers', marker=dict(size=3, color='blue', opacity=1)
        ))
        return

    with open('./test_data', 'rb') as f:
        data, names = pickle.load(f)

    data, foo = normalize_coordinates(data)
    plot_d(data, 'data', names)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    rand_points = []  #
    rand_scores = []
    nvectors = get_principle_components(data, krng)
    # for i in [1]:  # np.arange(0.27, 0.32, 0.01):
    #     for j in np.arange(0, 1, 0.03):
    #         for k in np.arange(0, 1, 0.03):
    #             new_point = [i, j, k]
    #             score = get_anomaly_score(new_point, data, nvectors, grid_size=gz, x=xname)
    #             qwe = 1
    #             if score < qwe:
    #                 rand_scores.append(score)
    #                 rand_points.append(new_point)

    for i in np.arange(0, 1, 0.04):  # np.arange(0.27, 0.32, 0.01):
        for j in [0.48]:
            for k in np.arange(0, 1, 0.04):
                new_point = [i, j, k]
                score = get_anomaly_score(new_point, data, nvectors, grid_size=gz, x=xname)
                qwe = 1
                if score < qwe:
                    rand_scores.append(score)
                    rand_points.append(new_point)
    # for new_point in [[.72, 0.2189, 0.64]]:
    #     score = get_anomaly_score(new_point, data, nvectors, grid_size=gz, x=xname)
    #     qwe = 1
    #     if score < qwe:
    #         rand_scores.append(score)
    #         rand_points.append(new_point)

    x = [x[0] for x in rand_points]
    y = [x[1] for x in rand_points]
    z = [x[2] for x in rand_points]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        hovertext=[str(x) for x in rand_scores], name=str(krng) + '-' + str(xname) + '-' + str(gz), mode='markers',
        marker=dict(size=7, color=np.asarray(rand_scores), opacity=0.7, colorscale='RdYlGn', reversescale=True)
    ))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new points end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

    fig.show()
    return


def sequence_anomaly_detector2(decay=0.05):
    '''
    Conclusion: Good, untrained lstm is only sensitive to the most recent events, so need reverse order
    :return:
    '''
    import plotly.graph_objects as go
    fig = go.Figure()
    emb_size = 3
    s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
    char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
    lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
                                                   device=tdevice)

    def fprop1(s):
        v, sb = s[0], s[1:]
        for i in range(len(sb)):
            k = 1 - i * decay
            v += k * sb[i]
        return v

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
        with open(path, 'r') as f:
            data = f.readlines()
        # data = [x[:-1] for x in data]  # remove last char \n for a start
        data = data[5:lim]
        data1 = [s[::-1] for s in data]
        data_emb = [str_to_emb(x) for x in data1]

        # qwe = fprop1(data_emb[0])
        # lstmc_hid_outputs = [lstm(x.unsqueeze(1), init)[1][0].squeeze().cpu().data.numpy() for x in data_emb]
        lstmc_hid_outputs = [fprop1(x) for x in data_emb]
        x = [x[0] for x in lstmc_hid_outputs]
        y = [x[1] for x in lstmc_hid_outputs]
        z = [x[2] for x in lstmc_hid_outputs]
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


def dissecting_lstm_test():
    input_size, hidden_size, num_of_layers = 2, 1, 1
    lstm = t.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_of_layers,
                     bidirectional=False)
    x = t.rand(size=[1, 1, input_size])
    y, h_c = lstm(x)

    w_ii = lstm.weight_ih_l0[0]
    w_if = lstm.weight_ih_l0[1]
    w_ig = lstm.weight_ih_l0[2]
    w_io = lstm.weight_ih_l0[3]

    b_ii = lstm.bias_ih_l0[0]
    b_if = lstm.bias_ih_l0[1]
    b_ig = lstm.bias_ih_l0[2]
    b_io = lstm.bias_ih_l0[3]

    w_hi = lstm.weight_hh_l0[0]
    w_hf = lstm.weight_hh_l0[1]
    w_hg = lstm.weight_hh_l0[2]
    w_ho = lstm.weight_hh_l0[3]

    b_hi = lstm.bias_hh_l0[0]
    b_hf = lstm.bias_hh_l0[1]
    b_hg = lstm.bias_hh_l0[2]
    b_ho = lstm.bias_hh_l0[3]
    return


def sequence_anomaly_detector():
    '''
    Reduces to a 2D plot, out size determines the maximum length of seq it can capture. small out~3 means long seq
    cannot be captured effectively
    :return:
    '''
    import plotly.graph_objects as go
    fig = go.Figure()
    emb_size = 100
    out = 3

    s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
    char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
    modela = 2 * t.rand(size=[emb_size + out, out]) - 1
    fc = t.rand(size=[out, 3])

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
            return output.softmax(dim=0)
        return

    def _h(path, lim):
        with open(path, 'r') as f:
            data = f.readlines()
        # data = [x[:-1] for x in data]  # remove last char \n for a start
        data = data[5:lim]
        data1 = [s[::-1] for s in data]
        data_emb = [str_to_emb(x) for x in data1]

        lstmc_hid_outputs = [fprop(x) for x in data_emb]
        x = [x[0] for x in lstmc_hid_outputs]
        y = [x[1] for x in lstmc_hid_outputs]
        z = [x[2] for x in lstmc_hid_outputs]
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
    However, training epoch doesnt seem to affect clustering much
    :return:
    '''
    import plotly.graph_objects as go
    from ml_helper import TorchHelper as th
    fig = go.Figure()
    emb_size = 10
    out = 80  # this must be about 80~150 to capture long sequences

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

    with open('/'.join(__file__.split('/')[:-1]) + '/data/nodejs_lib_paths.txt', 'r') as f:
        data = f.readlines()

    def h(epoch, data):
        data1 = [s[::-1] for s in data]
        out_emb = [str_to_emb(x) for x in data1]
        train(out_emb, epoch)
        _y, _y, encodings = fprop(out_emb)
        encodings = encodings.cpu().data.numpy()
        x1 = [x[0] for x in encodings]
        y1 = [x[1] for x in encodings]
        z1 = [x[2] for x in encodings]
        n1 = data
        return x1, y1, z1, n1

    x1, y1, z1, n1 = h(10, data[5:1000])
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

    x2, y2, z2, n2 = h(0, data[207:212])
    # start plotting
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        hovertext=n2,
        hoverinfo='text',  # this means xzy info is removed from hover
        name="a",
        mode='markers',
        marker=dict(
            size=8,
            color='red',  # set color to an array/list of desired values
            opacity=0.7
        )
    ))

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


def sequence_anomaly_detector_with_pca():
    '''
    fc dim must be >3 for clustering to work in 3d space, else 2d or less!
    :return:
    '''
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    import json
    fig = go.Figure()
    emb_size = 15
    out = 100

    s = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~Â®\n'
    char_emb = t.rand(size=[len(s), emb_size], dtype=t.float32, device=tdevice)
    # lstm, init = ml_helper.TorchHelper.create_lstm(input_size=emb_size, output_size=3, batch_size=1, num_of_layers=1,
    #                                                device=tdevice)
    modela = 2 * t.rand(size=[emb_size + out, out]) - 1

    char_emb = t.tensor(np.loadtxt('char_emb.txt'), dtype=t.float32)
    modela = t.tensor(np.loadtxt('modela.txt'), dtype=t.float32)

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
        data = ['asd', 'qwe']
        data1 = [s[::-1] for s in data]

        out_emb = [fprop(x) for x in [str_to_emb(x) for x in data1]]
        out_emb = t.stack(out_emb).data.numpy()
        pca = PCA(n_components=3)
        out2 = pca.fit_transform(out_emb)

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


if __name__ == '__main__':
    sequence_anomaly_detector_with_fc_and_pca()
