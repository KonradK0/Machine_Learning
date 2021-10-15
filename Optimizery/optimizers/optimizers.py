import numpy as np


class Optimizer:

    def __init__(self, initial_params):
        assert isinstance(initial_params, dict)
        assert all([isinstance(v, np.ndarray) for v in initial_params.values()])
        assert all([v.size > 0 for v in initial_params.values()])
        self._params = {k: v.copy() for k, v in initial_params.items()}

    def get_params(self):
        return {k: v.copy() for k, v in self._params.items()}

    def get_keys(self):
        return set(self._params.keys())

    def step(self, gradients):
        raise NotImplementedError()


class GradientDescent(Optimizer):

    def __init__(self, initial_params, learning_rate):
        super().__init__(initial_params)
        self.learning_rate = learning_rate

    def step(self, gradients):
        assert set(gradients.keys()) == set(self._params.keys())
        for k in gradients:
            self._params[k] -= self.learning_rate * gradients[k]


class Momentum(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.momentum = {key: np.zeros(len(initial_params[key])) for key in initial_params.keys}

    def step(self, gradients):
        assert set(gradients.keys()) == set(self._params.keys())
        for k in gradients:
            self.momentum[k] = self.momentum[k] * self.gamma + self.learning_rate * gradients[k]
            self._params[k] -= self.momentum[k]



class Nesterov(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma):
        super().__init__(initial_params)
        self.training_phase = True
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.momentum = {param: np.zeros(len(initial_params[param])) for param in initial_params}

    def step(self, gradients):
        assert set(gradients.keys()) == set(self._params.keys())
        for param in self._params.keys():
            self.momentum = self.gamma * self.momentum[param] + self.learning_rate * gradients[param]
            self._params[param] -= self.momentum[param]

    def _get_training_params(self):
        tmp_momentum = {param: np.zeros(len(self._params[param])) for param in self._params.keys()}
        training_params = {param: np.zeros(len(self._params[param])) for param in self._params.keys()}
        for param in self._params.keys():
            tmp_momentum[param] = self.gamma * self.momentum[param]
            training_params[param] = self._params[param] - tmp_momentum[param]
        return training_params

    def _get_test_params(self):
        return super().get_params()

    def get_params(self):
        if self.training_phase:
            return self._get_training_params()
        else:
            return self._get_test_params()


class Adagrad(Optimizer):

    def __init__(self, initial_params, learning_rate, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.h = {param: np.zeros(len(self._params[param])) for param in initial_params}

    def step(self, gradients):
        assert set(gradients.keys()) == set(self._params.keys())
        for param in self._params.keys():
            print(gradients[param])
            print(np.square(gradients[param]))
            print(self.h[param])
            self.h[param] += np.square(gradients[param])
            self._params[param] -= self.learning_rate * (gradients[param] / np.sqrt(self.h[param] + self.epsilon))


class RMSProp(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.h = {param: np.zeros(len(self._params[param])) for param in initial_params}

    def step(self, gradients):
        assert set(gradients.keys()) == set(self._params.keys())
        for param in self._params.keys():
            self.h[param] = self.gamma * self.h[param] + (1 - self.gamma) * (np.square(gradients[param]))
            self._params[param] -= self.learning_rate * (gradients[param] / np.sqrt(self.h[param] + self.epsilon))


class Adadelta(Optimizer):

    def __init__(self, initial_params, gamma, epsilon):
        super().__init__(initial_params)
        self.gamma = gamma
        self.epsilon = epsilon
        self.h = {param: np.zeros(len(self._params[param])) for param in initial_params}
        self.d = {param: np.zeros(len(self._params[param])) for param in initial_params}
        # UZUPEŁNIĆ

    def step(self, gradients):
        assert set(gradients.keys()) == set(self._params.keys())
        prev_step_params = {k: v.copy() for k, v in self._params}
        for param in self._params.keys():
            self.h[param] = self.gamma * self.h[param] + (1 - self.gamma) * np.square(gradients[param])
            self._params[param] -= np.sqrt(self.d[param] + self.epsilon) * (
                    gradients[param] / np.sqrt(self.h[param] + self.epsilon))
            self.d = self.gamma * self.d[param] + (1 - self.gamma) * np.square(
                self._params[param] - prev_step_params[param])


class Adam(Optimizer):

    def __init__(self, initial_params, learning_rate, beta1, beta2, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.steps = 0
        self.m = {param: np.zeros(len(self._params[param])) for param in initial_params}
        self.v = {param: np.zeros(len(self._params[param])) for param in initial_params}
        # UZUPEŁNIĆ

    def step(self, gradients):
        assert set(gradients.keys()) == set(self._params.keys())
        self.steps += 1
        est_m = {param: np.zeros(len(self._params[param])) for param in self._params}
        est_v = {param: np.zeros(len(self._params[param])) for param in self._params}
        for param in self._params.keys():
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * gradients[param]
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * np.square(gradients[param])
            est_m[param] = self.m[param] / (1 - self.beta1 ** self.steps)
            est_v[param] = self.v[param] / (1 - self.beta2 ** self.steps)
            self._params[param] -= self.learning_rate * est_m / (np.sqrt(est_v) + self.epsilon)

        # UZUPEŁNIĆ
