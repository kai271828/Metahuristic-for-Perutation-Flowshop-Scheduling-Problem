import fire
import numpy as np
from numba import jit
from typing import Union
from tqdm.auto import tqdm
from datetime import datetime

from utils import TFSProblem, Solution
from SimulatedAnnealing import SimulatedAnnealing


@jit
def main(
    data_dir: str = "data/tai20_5_1.txt",
    epoch_len: int = 1,
    alpha: float = 0.99,
    stopcriterion: Union[float, int] = 1,
    temperature: Union[float, int] = 1000,
    times: int = 20,
    log_dir: Union[str, None] = None,
    verbose: bool = False,
    run_search: bool = False,
):
    p = TFSProblem(data_dir)
    sa = SimulatedAnnealing(epoch_len, alpha, stopcriterion)

    best = np.inf
    best_sol = None
    worst = 0
    worst_sol = None
    record = []

    log_file = (
        open(f"{log_dir}/{datetime.now().strftime('%Y-%m-%d %H:%M')}.txt", "w+")
        if log_dir is not None
        else None
    )

    p_bar = range(times) if run_search else tqdm(range(times))
    for i in p_bar:
        solution, search_steps = sa.search(p, temperature, verbose=verbose)
        makespan = p.evaluate(solution.sol)
        record.append(makespan)

        if verbose:
            print(f"\n\nFinal solution: {solution} after {search_steps} steps.")
            print(f"The makespan is {makespan}.")
        if log_file is not None:
            log_file.write(
                f"[Experiment {i + 1}] get solution \t{solution}\t after {search_steps} steps.\n"
            )
            log_file.write(f"The makespan is {makespan}.\n")

        if makespan < best:
            best = makespan
            best_sol = solution
        elif makespan > worst:
            worst = makespan
            worst_sol = solution

    avg = sum(record) / times

    if not run_search:
        print(f"\n\nBest solutin {best_sol} has makespan {best}")
        print(f"Worst solutin {worst_sol} has makespan {worst}")
        print(f"Average makespan {avg}")

    if log_file is not None:
        log_file.write(f"\n\nBest solutin {best_sol} has makespan {best}\n")
        log_file.write(f"Worst solutin {worst_sol} has makespan {worst}\n")
        log_file.write(f"Average makespan {avg}\n")
        log_file.close()

    return best, avg, worst


if __name__ == "__main__":
    fire.Fire(main)
