import numpy as np

from utils import Solution


class SimulatedAnnealing:
    def __init__(
        self,
        temperature,
        epoch_len,
        alpha,
        stopcriterion,
    ):
        self.temperature = temperature
        self.epoch_len = epoch_len
        self.alpha = alpha
        self.stopcriterion = stopcriterion

    def search(self, problem, init_sol=None, verbose=False):
        i = Solution(length=problem.sol_length, init_sol=init_sol)
        temperature = self.temperature
        length = self.epoch_len
        k = 0
        ffe = 0
        record = {"per_update": [], "per_ffe": [], "temperature": []}

        while temperature > self.stopcriterion:
            record["temperature"].append(temperature)
            if verbose:
                print(f"[Epoch {k + 1}] Current Temperature: {temperature}")

            for l in range(length):
                if verbose:
                    print(f"[Step {l + 1}]")

                # get random 2 indices
                a, b = np.random.choice(problem.sol_length, 2, replace=False)

                j = i.swap_neighborhood(a, b)
                ffe += 1

                old_makespan = (
                    i.makespan if hasattr(i, "makespan") else problem.evaluate(i)
                )
                new_makespan = problem.evaluate(j)

                if new_makespan <= old_makespan:
                    i.sol = j.sol.copy()
                    i.makespan = new_makespan
                    record["per_ffe"].append(new_makespan)
                    record["per_update"].append(new_makespan)
                    if verbose:
                        print(
                            f"Set solution to {j} since {new_makespan} is better than {old_makespan}."
                        )
                else:
                    random_num = np.random.rand()
                    if np.exp((old_makespan - new_makespan) / temperature) > random_num:
                        i.sol = j.sol.copy()
                        i.makespan = new_makespan
                        record["per_ffe"].append(new_makespan)
                        record["per_update"].append(new_makespan)
                        if verbose:
                            print(
                                f"Set solution to {j} but {new_makespan} is NOT better than {old_makespan} since {np.exp((old_makespan - new_makespan) / temperature)} > {random_num}."
                            )
                    else:
                        record["per_ffe"].append(old_makespan)
                        if verbose:
                            print(
                                f"Didn't set anything since {np.exp((old_makespan - new_makespan) / temperature)} <= {random_num}."
                            )

            k += 1
            temperature = self.calculate_control(temperature)
            length = self.calculate_length(length)

        return i, k, ffe, record

    def calculate_control(self, temperature):
        return temperature * self.alpha

    def calculate_length(self, l):
        return l
