import numpy as np
from tqdm.auto import tqdm

from utils import Solution
from SimulatedAnnealing import SimulatedAnnealing


class MemeticAlgorithm:
    def __init__(
        self,
        p_size=20,
        init_k=2,
        init_ls=None,
        init_ls_ratio=0.2,
        tournament_k=2,
        offspring_m=2,
        mutate_prob=0.2,
        end_ls=None,
        end_ls_ratio=0.2,
    ):
        self.p_size = p_size
        self.init_k = init_k
        self.tournament_k = tournament_k
        self.offspring_m = offspring_m
        self.mutate_prob = mutate_prob
        self.init_ls_ratio = init_ls_ratio
        assert (
            init_ls is not None
        ), "You have to assign an init_ls object with 'search(problem, init_sol=...)' method"
        self.init_ls = init_ls
        self.end_ls_ratio = end_ls_ratio
        assert (
            end_ls is not None
        ), "You have to assign an end_ls object with 'search(problem, init_sol=...)' method"
        self.end_ls = end_ls

    def search(self, problem, num_iter=10, verbose=False):

        # Initialization
        pop = Population(size=self.p_size, problem=problem, init_k=self.init_k)
        record = []

        # mutate offspring of local search of some initial solution
        mutate_indices = np.random.choice(
            pop.size, int(pop.size * self.init_ls_ratio), replace=False
        )

        for index in mutate_indices:
            ls_result, _, _, _ = self.init_ls.search(problem, init_sol=pop[index])

            i, j, k = np.random.choice(problem.sol_length, 3, replace=False)
            mutated = MASolution(init_sol=ls_result)
            mutated.swap_3_mutate(i, j, k)

            pop[index] = mutated

        if verbose:
            print("Original populations:")
            for p in pop:
                print(p)

        for i in tqdm(range(num_iter)):
            if verbose:
                print(f"[Iteration {i + 1}]")

            # Evaluation
            pop.evaluate_and_sort(problem=problem)

            if verbose:
                print("After evaluation:")
                for p in pop:
                    print(f"Solutino: {p}\nMakespan: {p.makespan}\n\n")

            # Reproduction
            for j in range(int(pop.size // 2 * self.offspring_m)):
                # Mating selection (tournament selection)
                parent_id_1 = pop.k_tournament(self.tournament_k)
                parent_id_2 = pop.k_tournament(self.tournament_k)

                # Crossover
                # order/linear order/partially-mapped crossover/cycle crossover
                offspring_1, offspring_2 = pop.crossover(parent_id_1, parent_id_2)

                if verbose:
                    print(f"Offspring 1: {offspring_1}\nOffspring 2: {offspring_2}\n")

                random_num = np.random.random()
                if random_num < self.mutate_prob:
                    start, end = sorted(
                        np.random.choice(problem.sol_length, 2, replace=False)
                    )
                    offspring_1.insertion_mutate(start, end)

                    if verbose:
                        print(f"Offspring 1 mutated: {offspring_1}\n")
                pop.append(offspring_1)

                random_num = np.random.random()
                if random_num < self.mutate_prob:
                    start, end = sorted(
                        np.random.choice(problem.sol_length, 2, replace=False)
                    )
                    offspring_2.insertion_mutate(start, end)

                    if verbose:
                        print(f"Offspring 2 mutated: {offspring_2}\n")

                pop.append(offspring_2)

            # Evaluation
            pop.evaluate_and_sort(problem=problem)
            record.append(pop[0].makespan)

            if verbose:
                print("After evaluation:")
                for p in pop:
                    print(f"Solution: {p}\nMakespan: {p.makespan}\n\n")

            # Environmental selection
            pop.environmental_select()

            if verbose:
                print("After environmental selection:")
                for p in pop:
                    print(f"Solution: {p}\nMakespan: {p.makespan}\n\n")

            # Local search
            # move operator should not be the same as crossover and mutation
            for j in range(
                self.p_size - int(self.p_size * self.end_ls_ratio), self.p_size
            ):
                solution, _, _, record = self.end_ls.search(problem, init_sol=pop[j])
                solution.makespan = record["per_ffe"][-1]
                pop[j] = solution

        return pop, record


class MASolution(Solution):
    def __init__(self, length=20, init_sol=None):
        super().__init__(length=length, init_sol=init_sol)

    def swap_3_mutate(self, i, j, k):
        temp = self[i]
        self[i] = self[j]
        self[j] = self[k]
        self[k] = temp

    def insertion_mutate(self, start, end):
        """Mutate the solution by insertion.
        Start index sould less than end index"""
        temp = self[start]
        for i in range(start, end):
            if i + 1 < len(self.sol):
                self[i] = self[i + 1]
        self[end] = temp


class Population:
    def __init__(
        self,
        size,
        problem,
        init_k,
    ):
        # best N from kN random solutions
        self._pop = []
        self._size = size
        self._length = problem.sol_length

        temp = [MASolution(length=self._length) for i in range(self._size * init_k)]

        temp.sort(key=lambda x: problem.evaluate(x))

        self._pop = temp[: (self._size)]

    def __getitem__(self, key):
        return self._pop[key]

    def __setitem__(self, key, value):
        self._pop[key] = value

    @property
    def size(self):
        return self._size

    @property
    def length(self):
        return self._length

    @property
    def diversity(self):
        return len(set([p.makespan for p in self]))

    def append(self, p):
        self._pop.append(p)

    def evaluate_and_sort(self, problem):
        for p in self._pop:
            p.makespan = problem.evaluate(p)

        self._pop.sort(key=lambda x: x.makespan)

    def k_tournament(self, k):
        """Return a index of best p from k random solutions
        The Object(self) should be sorted."""
        random_indices = np.random.choice(self.size, k, replace=False)
        return np.min(random_indices)

    def crossover(self, parent_id_1, parent_id_2):
        offspring_1 = MASolution(init_sol=self._pop[parent_id_1])
        offspring_2 = MASolution(init_sol=self._pop[parent_id_2])

        # Randomly select two points
        point_1, point_2 = sorted(np.random.choice(self.length, 2, replace=False))

        # Create sets to keep track of selected elements
        selected_1 = set(offspring_1[point_1:point_2])
        selected_2 = set(offspring_2[point_1:point_2])

        # The remaining elements in parent 2 are added to offspring 1 in order, and vice versa
        index_1 = index_2 = point_2
        for i in range(self.length):
            if self._pop[parent_id_2][i] not in selected_1:
                offspring_1[index_1 % self.length] = self._pop[parent_id_2][i]
                index_1 += 1
            if self._pop[parent_id_1][i] not in selected_2:
                offspring_2[index_2 % self.length] = self._pop[parent_id_1][i]
                index_2 += 1

        return offspring_1, offspring_2

    def environmental_select(self):
        """Select the best self.size solutions to survive"""
        # TODO: Common environmental selection
        self._pop = self._pop[: self.size]
