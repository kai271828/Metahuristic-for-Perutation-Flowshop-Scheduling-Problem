import fire
import numpy as np
from typing import Union

from utils import TFSProblem, Solution
from SimulatedAnnealing import SimulatedAnnealing


def main(
    data_dir: str = "data/tai20_5_1.txt",
    epoch_len: int = 1,
    alpha: float = 0.99,
    stopcriterion: Union[float, int] = 1,
    temperature: Union[float, int] = 1000,
    times: int = 20,
    output_dir: str = "output",
    verbose: bool = False,
):
    p = TFSProblem(data_dir)
    sa = SimulatedAnnealing(epoch_len, alpha, stopcriterion)

    best = np.inf
    best_sol = None
    worst = 0
    worst_sol = None
    record = []

    for i in range(times):
        solution, search_steps = sa.search(p, temperature, verbose=verbose)
        makespan = p.evaluate(solution.sol)
        record.append(makespan)

        print(f"\n\nFinal solution: {solution} after {search_steps} steps.")
        print(f"The minimum makespain is {makespain}.")

        if makespan < best:
            best = makespain
            best_sol = solution
        elif makespan > worst:
            worst = makespain
            worst_sol = solution

    print(f"\n\nBest solutin {best_sol} has makespan {best}")
    print(f"Worst solutin {worst_sol} has makespan {worst}")
    print(f"Average makespan {sum(record) / len(record)}")


if __name__ == "__main__":
    fire.Fire(main)
