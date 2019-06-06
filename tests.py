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
    noise_training_noisy_negatives_autoencoder_ann_vs_lstm()
