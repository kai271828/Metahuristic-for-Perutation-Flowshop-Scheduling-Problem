import numpy as np

from utils import Solution


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
                j = i.swap_neighborhood()
                if problem.evaluate(i.sol) <= problem.evaluate(j):
                    i.set_sol(j)
                    if verbose:
                        print(f"Set solution to {i} since it is better.")
                elif (
                    np.exp((problem.evaluate(i.sol) - problem.evaluate(j)) / self.c)
                    > np.random.rand()
                ):
                    i.set_sol(j)
                    if verbose:
                        print(f"Set solution to {i} but it isn't better.")

            k += 1
            temperature = self.calculate_control(temperature)
            l = self.calculate_control(l)

        return i, k

    def calculate_control(self, temperature):
        return temperature * self.alpha

    def calculate_length(self, l):
        return l
