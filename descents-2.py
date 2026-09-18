import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        n_k = self.lambda_ * (self.s0 / (self.s0 + iteration))**self.p
        return n_k

# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        loss_history = []
        self.iteration = 0
    
        loss_history.append(self.model.compute_loss(self.model.X_train, self.model.y_train))
    
        prev_delta = None
    
        while self.iteration < self.max_iter:
            delta = self._step()  
    
            loss_history.append(self.model.compute_loss(self.model.X_train, self.model.y_train))
    
            if np.isnan(delta).any():
                break
    
            if prev_delta is not None:
                if float((delta * delta).sum()) < self.tolerance:
                    break
    
            prev_delta = delta
            # self.iteration += 1
    
        self.model.loss_history = loss_history.copy()

        
        # TODO: implement
        # в конце также приcваивает атрибуту модели полученный loss_history

# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train

        grad = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)

        delta = -lr * grad
        self.model.w = self.model.w + delta
        return delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        X_train = self.model.X_train
        y_train = self.model.y_train
        n = X_train.shape[0]
    
        idx = np.random.randint(0, n, size=self.batch_size)
    
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        grad = self.model.compute_gradients(X_batch, y_batch)
    
        lr = self.lr_schedule.get_lr(self.iteration)
        delta = -lr * grad
        self.model.w += delta
        return delta


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size


    def _update_weights(self) -> np.ndarray:
        X = self.model.X_train
        y = self.model.y_train
        n, d = X.shape
    
        if self.grad_memory is None:
            self.grad_memory = np.zeros((n, d), dtype=float)
            self.avg_grad = np.zeros(d, dtype=float)
    
        idx = np.random.randint(0, n, size=self.batch_size)
        idx = np.unique(idx)
    
        new_grads = np.zeros((len(idx), d), dtype=float)
        for t, j in enumerate(idx):
            new_grads[t] = self.model.compute_gradients(X[j:j+1], y[j:j+1]).reshape(-1)
        # тут частично есть циклы, но в условии в подсказках сказано так использовать
        old_grads = self.grad_memory[idx]
    
        self.avg_grad += (new_grads - old_grads).sum(axis=0) / n
        self.grad_memory[idx] = new_grads
    
        lr = self.lr_schedule.get_lr(self.iteration)
        delta = -lr * self.avg_grad
        self.model.w += delta
    
        return delta

class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        X = self.model.X_train
        y = self.model.y_train
    
        grad = self.model.compute_gradients(X, y)
        lr = self.lr_schedule.get_lr(self.iteration)
    
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
    
        self.velocity = self.beta * self.velocity + lr * grad
    
        delta = -self.velocity
        self.model.w += delta
        return delta

class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        X = self.model.X_train
        y = self.model.y_train
    
        grad = self.model.compute_gradients(X, y)
        lr = self.lr_schedule.get_lr(self.iteration)
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
        if self.v is None:
            self.v = np.zeros_like(self.model.w)
        t = self.iteration+ 1
        
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v +(1.0 - self.beta2)* (grad ** 2)
    
        m_hat = self.m/(1.0 - self.beta1 ** t)
        v_hat = self.v/(1.0 - self.beta2 ** t)
    
        delta = -lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.model.w += delta
        return delta

# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект

        X = self.model.X_train
        y = self.model.y_train
        w = self.model.loss_function.analytic_solution(X, y)
        self.model.w = w
