import tensorflow as tf
import torch as t
import pickle


class tf_helpers():
    @staticmethod
    def create_layers(layer_sizes, name=''):
        output = []
        for i in range(len(layer_sizes) - 1):
            if not output:
                rng_seed = tf.random_normal([layer_sizes[i], layer_sizes[i + 1]], stddev=0.1)
                output.append(tf.Variable(rng_seed, name=name + '_w' + str(i)))
        if name != '':
            output = tf.identity(output, name=name)
        return output

    @staticmethod
    def matmul_activate(tf_matrix, activations=[], name=''):
        # example, activations = [[],tf.tanh,tf.sigmoid,[]]
        # eg matmul_activate(create_layers
        output = []
        for i in range(len(tf_matrix) - 1):
            input = tf_matrix[0] if i == 0 else output
            if i < len(activations):
                if activations[i]:
                    input = activations[i](input)
            output = tf.matmul(input, tf_matrix[i + 1])
        if name != '':
            output = tf.identity(output, name=name)
        return output


class torch_helpers():
    @staticmethod  # enforces a strict structure
    def create_lstm(input_dim, output_size, batch_size, num_of_layers, bidirectional=False):
        t.manual_seed(1)
        # data details
        hidden_size = output_size

        # model details
        num_directions = 2 if bidirectional else 1

        lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_of_layers,
                       bidirectional=bidirectional)

        # initialize the hidden state.
        hidden = (t.randn(num_of_layers * num_directions, batch_size, hidden_size),  # this is for h_0,
                  t.randn(num_of_layers * num_directions, batch_size, hidden_size))  # this is for c_0, cell state

        # inputs = t.randn(seq_len, batch_size, input_dim)
        # out, hidden = lstm(inputs, hidden)
        return lstm, hidden

    @staticmethod
    def prod_estimator(x, y, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=1000):
        # for forms Ca^m.b^n
        # can only be used for +ve numbers,
        w_coefficient = t.randn([1], dtype=dtype, device=device,
                                requires_grad=True)  # t.randn(1, device=device, dtype=dtype, requires_grad=True)
        w_power = t.ones(x.shape[1], y.shape[1], device=device, dtype=dtype, requires_grad=True)

        # w_power = t.tensor([[2],[1]], device=device, dtype=dtype, requires_grad=True)

        def fprop(x):
            y_ = t.log(x)
            y_ = y_.mm(w_power)  # this controls mul/divide
            y_ = t.sum(y_, dim=1)
            y_ = t.exp(y_)
            y_ = y_.mul(w_coefficient)  # accounts for coefficient
            y_ = y_.reshape([len(y_), 1])
            return y_

        loss_fn = lambda y_, y: (y_ - y).pow(2).mean()

        def bprop(x, y, w):
            w_power, w_coefficient = w[0], w[1]
            y_ = fprop(x)
            l = loss_fn(y_, y)
            l.backward()

            with t.no_grad():
                w_power -= lr * w_power.grad
                w_coefficient -= lr * w_coefficient.grad
                print(i, ' ', l.item())

                w_power.grad.zero_()
                w_coefficient.grad.zero_()
            return [w_power, w_coefficient]

        for i in range(iter):
            bprop(x, y, [w_power, w_coefficient])
        return [w_power, w_coefficient], fprop, bprop

    @staticmethod # unfinished
    def sum_estimator(x, y, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=500000):
        # for forms Aa^x+Bb^y+Cc^z
        w_coefficient = t.ones([x.shape[1], 1], dtype=dtype, device=device,
                               requires_grad=True)
        w_power = t.ones([x.shape[1]], dtype=dtype, device=device,
                         requires_grad=True)  # t.randn(1, device=device, dtype=dtype, requires_grad=True)

        def fprop(x):
            y_ = t.pow(x, w_power)
            y_ = y_.mm(w_coefficient)
            return y_

        loss_fn = lambda y_, y: (y_ - y).pow(2).mean()

        def bprop(x, y, w):
            w_power, w_coefficient = w[0], w[1]
            y_ = fprop(x)
            l = loss_fn(y_, y)
            l.backward()
            with t.no_grad():
                w_power -= lr * w_power.grad
                w_coefficient -= lr * w_coefficient.grad
                print(l.item())
                w_power.grad.zero_()
                w_coefficient.grad.zero_()
            return [w_power, w_coefficient]

        for i in range(iter):
            bprop(x, y, [w_power, w_coefficient])
        return [w_power, w_coefficient], fprop, bprop

    @staticmethod
    def composite_estimator(x, y, hidden=10, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=50):


        composite_set = [
            torch_helpers.prod_estimator(x, y, iter=0),
            torch_helpers.prod_estimator(x, y, iter=0)
        ]

        w_hidden = t.randn([hidden * len(composite_set), y.shape[1]], dtype=dtype, device=device,
                           requires_grad=True)  # t.randn(1, device=device, dtype=dtype, requires_grad=True)

        set_output = [x[1](x) for x in composite_set]
        set_output = t.cat(set_output, dim=1)



        # fn_ar = [[t.sigmoid,3],[t.tanh,3]]
        return
