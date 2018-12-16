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
    @staticmethod
    def prod_estimator(x, y, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=500000, opt='diff'):
        # for forms Ca^m.b^n
        # can only be used for +ve numbers,
        w_coefficient = t.randn([1], dtype=dtype,
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

        for i in range(iter):
            y_ = fprop(x)
            l = loss_fn(y_, y)
            l.backward()

            with t.no_grad():
                w_power -= lr * w_power.grad
                w_coefficient -= lr * w_coefficient.grad
                print(l.item())

                w_power.grad.zero_()
                w_coefficient.grad.zero_()

        l = l.data.item()

        for i in range(iter):
            with t.no_grad():
                def _h(w):
                    l = loss_fn(fprop(x), y).data.item()
                    d = t.randn(w.shape)
                    w -= lr * d
                    l1 = loss_fn(fprop(x), y).data.item()

                    print(l1, ' ', l)
                    if l1 < l:
                        return w
                    else:
                        return w + 2 * lr * d
                    return w
                w_coefficient = _h(w_coefficient)
                w_power = _h(w_power)
        return w_coefficient, w_power, fprop

    @staticmethod
    def sum_estimator(x,y, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=500000, opt='diff'):
        # for forms Aa^x+Bb^y+Cc^z

        return