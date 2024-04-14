import numpy as np
from tqdm.auto import tqdm

from utils import Solution, Population
from SimulatedAnnealing import SimulatedAnnealing


class MemeticAlgorithm:
    def __init__(
        self, cross_prob=0.5, muta_prob=0.2, epoch_len=10, alpha=0.9, temperature=3500
    ):
        self.cross_prob = cross_prob
        self.muta_prob = muta_prob
        self.epoch_len = epoch_len
        self.alpha = alpha
        self.temperature = temperature

    def search(self, problem, p_size=20, t=10):
        self.problem = problem

        # Initialization
        pop = Population(length=problem.sol_length, size=p_size)
        sa = SimulatedAnnealing(self.epoch_len, self.alpha, 1)

        # Evaluation

        for i in tqdm(range(t)):
            # Mating selection
            # tournament selection
            children_list = []

            # Reproduction
            for j in range(n // 2):
                # Crossover
                # order/linear order/partially-mapped crossover/cycle crossover
                random_indices = np.random.choice(n, 2, replace=False)
                cross_rand = np.random.random()

                children = self.cross_over(
                    pop[random_indices[0]], pop[random_indices[1]]
                )

                # Mutation
                # insert
                for child in children:
                    muta_rand = np.random.random()
                    if muta_rand < self.muta_prob:
                        a, b = np.random.choice(
                            self.problem.sol_length, 2, replace=False
                        )

                        child.set_sol(child.swap_neighborhood(a, b))

                    children_list.append(child)

            pop.extend(children_list)
            # Evaluation

            # Environmental selection
            pop = self.select_survivors(pop, n)

            # Local search
            # move operator should not be the same as crossover and mutation
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


class Population:
    def __init__(
        self,
        problem,
        size=20,
        k=5,
        sa_ratio=0.2,
        sa_temperature=5000,
        sa_epoch_length=15,
        sa_alpha=0.98,
        sa_stopcriterion=1,
    ):
        # mutated offsprings from local search + best N from kN random solutions
        self._pop = []

        sa_num = int(size * sa_ratio)
        temp = [Solution(length=problem.sol_length) for i in range(sa_num)]
        sa = SimulatedAnnealing(sa_epoch_length, sa_alpha, sa_stopcriterion)

        for p in temp:
            sa_p, _, _, _ = sa.search(problem, temperature, init_sol=p)
            self._pop.append(sa_p)

        temp = [Solution(length=problem.sol_length) for i in range(size * k)]
        temp.sort(key=lambda x: problem.evaluate(x))

        self._pop.extend(temp[: (size - sa_num)])

        assert len(self._pop) == size, "There are some bugs in your init of Population"

    def __getitem__(self, key):
        return self._pop[key]

    def __setitem__(self, key, value):
        self._pop[key] = value

    def evaluate(self):
        pass