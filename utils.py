import ctypes
import numpy as np


class PFSProblem:
    def __init__(self, input_dir, lib_dir="lib/evaluate.so"):
        self.lib = ctypes.CDLL(lib_dir)
        self.lib.evaluate.restype = ctypes.c_int64
        self.lib.evaluate.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(
                dtype=np.uintp, ndim=1, flags="C"
            ),  # np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags="C_CONTIGUOUS"),
            ctypes.c_int64,
            ctypes.c_int64,
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

        mj_table_ptr = (
            self.mj_table.__array_interface__["data"][0]
            + np.arange(self.mj_table.shape[0]) * self.mj_table.strides[0]
        ).astype(np.uintp)

        return self.lib.evaluate(
            sol.sol,
            mj_table_ptr,
            self.num_machines,
            self.num_jobs,
        )


class Solution:
    def __init__(self, length=20, init_sol=None):
        if init_sol is not None:
            self.sol = init_sol.sol.copy()
        else:
            self.sol = np.random.permutation(length)

    def __repr__(self):
        return ", ".join(map(lambda x: str(x), self.sol.tolist()))

    def __getitem__(self, key):
        return self.sol[key]

    def __setitem__(self, key, value):
        self.sol[key] = value

    def swap_neighborhood(self, i, j):
        """
        Return a new solution obtained by swap two elements
        """
        new_sol = Solution(init_sol=self)

        new_sol[i] = self[j]
        new_sol[j] = self[i]

        return new_sol

    @property
    def sol(self):
        return self._sol

    @sol.setter
    def sol(self, sol):
        self._sol = sol

    @property
    def makespan(self):
        return self._makespan

    @makespan.setter
    def makespan(self, makespan):
        self._makespan = makespan
