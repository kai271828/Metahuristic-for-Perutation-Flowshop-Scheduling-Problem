import numpy as np
from numba.experimental import jitclass

from utils import Solution


@jitclass
class SimulatedAnnealing:
    def __init__(
        self,
        epoch_len,
        alpha,
        stopcriterion,
    ):
        self.epoch_len = epoch_len
        self.alpha = alpha
        self.stopcriterion = stopcriterion

    def search(self, problem, temperature, verbose=False):

        i = Solution(problem.sol_length)
        l = self.epoch_len
        k = 0

        while temperature > self.stopcriterion:
            if verbose:
                print(f"[Step {k + 1}] Current Temperature: {temperature}")

            for l in range(self.epoch_len):
                a = np.random.randint(0, high=i.sol.shape[-1])
                b = np.random.randint(0, high=i.sol.shape[-1])
                if a == b:
                    b = (a + 1) % i.sol.shape[-1]

                j = i.swap_neighborhood(a, b)
                if problem.evaluate(i.sol) <= problem.evaluate(j):
                    i.set_sol(j)
                    if verbose:
                        print(f"Set solution to {i} since it is better.")
                elif (
                    np.exp(
                        (problem.evaluate(i.sol) - problem.evaluate(j)) / temperature
                    )
                    > np.random.rand()
                ):
                    i.set_sol(j)
                    if verbose:
                        print(f"Set solution to {i} but it is NOT better.")

            k += 1
            temperature = self.calculate_control(temperature)
            l = self.calculate_control(l)

        return i, k

    def calculate_control(self, temperature):
        return temperature * self.alpha

    def calculate_length(self, l):
        return l
