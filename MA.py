import numpy as np

from utils import Solution
from SimulatedAnnealing import SimulatedAnnealing


class MA:
    def __init__(
        self, cross_prob=0.5, muta_prob=0.2, epoch_len=10, alpha=0.9, temperature=3500
    ):
        self.cross_prob = cross_prob
        self.muta_prob = muta_prob
        self.epoch_len = epoch_len
        self.alpha = alpha
        self.temperature = temperature

    def search(self, problem, n=20, t=10):
        self.problem = problem

        pop = [Solution(self.problem.sol_length) for i in range(n)]
        sa = SimulatedAnnealing(self.epoch_len, self.alpha, 1)

        for i in range(t):
            children_list = []

            for j in range(n // 2):
                random_indices = np.random.choice(n, 2, replace=False)
                cross_rand = np.random.random()

                children = self.cross_over(
                    pop[random_indices[0]], pop[random_indices[1]]
                )

                # Mutation
                for child in children:
                    muta_rand = np.random.random()
                    if muta_rand < self.muta_prob:
                        a, b = np.random.choice(
                            self.problem.sol_length, 2, replace=False
                        )

                        child.set_sol(child.swap_neighborhood(a, b))

                    children_list.append(child)

            pop.extend(children_list)
            pop = self.select_survivors(pop, n)

            for j in range(n - 5, n):
                solution, _, _, _ = sa.search(
                    self.problem, self.temperature, init_sol=pop[j]
                )
                pop[j].set_sol(solution.sol)

        return pop

    def select_survivors(self, pop, num):
        temp = sorted(
            [(self.problem.evaluate(element.sol), element) for element in pop],
            key=lambda x: x[0],
        )
        return [element[-1] for element in temp[:num]]

    def cross_over(self, sol_a, sol_b):
        a = Solution(self.problem.sol_length)
        a.set_sol(sol_a.sol)
        b = Solution(self.problem.sol_length)
        b.set_sol(sol_b.sol)
        return a, b
