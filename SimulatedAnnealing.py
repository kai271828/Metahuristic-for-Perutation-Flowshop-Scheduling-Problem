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
        ffe = 0
        record = {"per_k": [], "per_ffe": [], "temperature": []}

        while temperature > self.stopcriterion:
            record["temperature"].append(temperature)
            if verbose:
                print(f"[Step {k + 1}] Current Temperature: {temperature}")

            for l in range(self.epoch_len):
                a = np.random.randint(0, high=i.sol.shape[-1])
                b = np.random.randint(0, high=i.sol.shape[-1])
                if a == b:
                    b = (a + 1) % i.sol.shape[-1]

                j = i.swap_neighborhood(a, b)
                ffe += 1

                old_makespan = problem.evaluate(i.sol)
                new_makespan = problem.evaluate(j)

                if new_makespan <= old_makespan:
                    i.set_sol(j)
                    record["per_ffe"].append(new_makespan)
                    record["per_k"].append(new_makespan)
                    if verbose:
                        print(f"Set solution to {j} since {new_makespan} is better.")

                elif (
                    np.exp((old_makespan - new_makespan) / temperature)
                    > np.random.rand()
                ):
                    i.set_sol(j)
                    record["per_ffe"].append(new_makespan)
                    record["per_k"].append(new_makespan)
                    if verbose:
                        print(f"Set solution to {j} but it is NOT better.")
                        print(
                            f"Exp value: {np.exp((old_makespan - new_makespan) / temperature)}"
                        )
                else:
                    record["per_ffe"].append(old_makespan)
                    if verbose:
                        print("Didn't set anything.")
                        print(
                            f"Exp value: {np.exp((old_makespan - new_makespan) / temperature)}"
                        )

            k += 1
            temperature = self.calculate_control(temperature)
            l = self.calculate_control(l)

        return i, k, ffe, record

    def calculate_control(self, temperature):
        return temperature * self.alpha

    def calculate_length(self, l):
        return l
