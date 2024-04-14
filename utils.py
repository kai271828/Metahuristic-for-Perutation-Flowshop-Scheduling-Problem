import ctypes
import numpy as np
from SimulatedAnnealing import SimulatedAnnealing


class PFSProblem:
    def __init__(self, input_dir, lib_dir="lib/evaluate.so"):
        self.lib = ctypes.CDLL(lib_dir)
        self.lib.evaluate.restype = ctypes.c_int32
        self.lib.evaluate.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
            ctypes.c_int32,
            ctypes.c_int32,
        ]

        with open(input_dir) as input_file:
            num_jobs, num_machines, _ = input_file.readline().split()

            self.num_jobs = int(num_jobs)
            self.num_machines = int(num_machines)

            table = []

            for i in range(self.num_machines):
                table.append(list(map(int, input_file.readline().split())))

            self.mj_table = np.array(table)

    @property
    def sol_length(self):
        return self.num_jobs

    def evaluate(self, sol):
        """
        Calculate the makespan for a given solution.
        """
        # machine_cache = np.zeros(self.num_machines)

        # for job_id in sol:
        #     for machine_id in range(self.num_machines):
        #         if machine_id != 0:
        #             machine_cache[machine_id] = (
        #                 max(machine_cache[machine_id - 1], machine_cache[machine_id])
        #                 + self.mj_table[machine_id, job_id]
        #             )
        #         else:
        #             machine_cache[machine_id] = (
        #                 machine_cache[machine_id] + self.mj_table[machine_id, job_id]
        #             )

        # return machine_cache[-1]
        return self.lib.evaluate(
            sol,
            mj_table,
            num_machines,
            num_jobs,
        )


class Solution:
    def __init__(self, length=20, init_sol=None):
        if init_sol is not None:
            self._sol = init_sol.copy()
        else:
            self._sol = np.random.permutation(length)

    def __repr__(self):
        return ", ".join(map(lambda x: str(x + 1), self.sol.tolist()))

    def __getitem__(self, key):
        return self._sol[key]

    def __setitem__(self, key, value):
        self._sol[key] = value

    def swap_neighborhood(self, i, j):
        """
        Return a new solution obtained by swap two elements
        """
        new_sol = Solution(init_sol=self.sol.copy())

        new_sol[i] = self.sol[j]
        new_sol[j] = self.sol[i]

        return new_sol

    @property
    def sol(self):
        return self._sol

    @sol.setter
    def sol(self, sol):
        self._sol = sol


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
