import numpy as np
from esig import tosig
import copy


class TensorAlgebra:
    def __init__(self, values):
        self._values = {}
        for val in values:
            self._values[len(np.array(val).shape)] = np.array(val)


    def __getitem__(self, item):
        if isinstance(item, int):
            if self.dim() == 0 and item != 0:
                return 0.

            return self._values.get(item, np.zeros([self.dim()]*item))


        try:
            tensor = self._values[len(item)]
            for i in item:
                tensor = tensor[i]

            return tensor

        except:
            return 0.


    def __len__(self):
        return max(self._values.keys())

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self*other

    def __pow__(self, power):
        return self.pow(power)

    def __add__(self, other):
        s = []

        assert self.dim() == other.dim() or self.dim()*other.dim() == 0, "Dimensions do not match"

        order = max(self.order(), other.order())

        for i in range(order + 1):
            s.append(self[i] + other[i])

        return TensorAlgebra(s)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1.)*other

    def __div__(self, other):
        assert isinstance(other, float) or isinstance(other, int), "Can not divide: invalid type"

        return self*(1/float(other))

    def __repr__(self):
        s = str([np.array(val).tolist() for val in self._values.values()])
        return "TensorAlgebra(%s)"%s

    def mul(self, other, truncate=None, restrict=None):
        if isinstance(other, float) or isinstance(other, int):
            return TensorAlgebra([other*val for val in self._values.values()])


        multiplied = {}
        for tensor1 in self._values.values():
            for tensor2 in other._values.values():
                order = len(tensor1.shape) + len(tensor2.shape)

                if restrict is not None and order != restrict:
                    continue

                if truncate is not None and order > truncate:
                    continue

                new_tensor = np.tensordot(tensor1, tensor2, axes=0)

                multiplied[order] = multiplied.get(order, []) + [new_tensor]


        values_mul = []

        for order in sorted(multiplied.keys()):
            values_mul.append(np.sum(multiplied[order], axis=0))


        return TensorAlgebra(values_mul)


    def pow(self, power, truncate=None):
        assert isinstance(power, int) and power >= 0, "Power has to be a non-negative integer"

        if power == 0:
            return TensorAlgebra([np.array(1.)])

        prod = None

        for _ in range(power):

            if prod is None:
                prod = self

            else:
                prod = prod.mul(self, truncate=truncate)

        return prod


    def inv(self, max_N = 10, truncate=None):
        a0 = float(self[0])
        assert a0 != 0, "Inverse does not exist"

        s = copy.deepcopy(ZERO)

        mul_n = copy.deepcopy(ONE)
        basis = copy.deepcopy(ONE) - self/a0

        for n in range(max_N):
            if n > 0:
                mul_n = mul_n.mul(basis, truncate=truncate)

            s += mul_n

        return s / a0

    def dim(self):
        try:
            return list(self._values.values())[-1].shape[0]

        except:
            return 0

    def truncate(self, n):
        return TensorAlgebra([value for order, value in self._values.items() if order <= n])


    def order(self):
        return max(self._values.keys())

    def norm(self):
        return np.sum([np.linalg.norm(value) for value in self._values.values()])


ONE = TensorAlgebra([np.array(1.)])
ZERO = TensorAlgebra([0.])

def exp(a, max_n=10):
    s = copy.deepcopy(ZERO)

    for n in range(max_n):
        s += a**n * (1 / np.math.factorial(n))


    return s

def log(a, max_n=10):
    a0 = float(a[0])

    s = TensorAlgebra([np.log(a0)])

    for n in range(1, max_n):
        s += (-1.)**n / float(n) * (ONE - a/a0)**n

    return s


def get_keys(dim, order):
    s = tosig.sigkeys(dim, order)
    tuples = []
    for t in s.split():
        if len(t) > 2:
            t = t.replace(")", ",)")

        tuples.append(eval(t))

    return tuples


def get_logkeys(dim, order):
    s = tosig.logsigkeys(dim, order)
    tuples = []
    for t in s.split():
        if isinstance(eval(t), int):
            t = "[%s]"%t

        tuples.append(eval(t))

    return tuples


def sig2tensor(sig, dim, sig_order):
    tensor_vals = {}
    for key, val in zip(get_keys(dim, sig_order), sig):
        order = len(key)
        if order not in tensor_vals:
            tensor_vals[order] = np.zeros([dim]*order)

        new_key = tuple(k - 1 for k in key)
        tensor_vals[order][new_key] = val

    return TensorAlgebra(tensor_vals.values())


def logsig2tensor(logsig, dim, sig_order):
    tensor_vals = {}
    for key, val in zip(get_logkeys(dim, sig_order), logsig):
        order = len(key)
        if order not in tensor_vals:
            tensor_vals[order] = np.zeros([dim]*order)

        new_key = tuple(k - 1 for k in key)
        tensor_vals[order][new_key] = val

    return TensorAlgebra(tensor_vals.values())





if __name__ == "__main__":
    v0 = np.array(1.)
    v1 = np.array([3., 1.])
    v2 = np.array([[4., 1.], [1., 1.]])
    v3 = np.array([[[3., 2.], [3., 1.]], [[2., 5.], [0., 1.]]])

    a = TensorAlgebra([v0, v1, v2])
