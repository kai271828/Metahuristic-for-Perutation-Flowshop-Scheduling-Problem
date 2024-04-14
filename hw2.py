import fire
import numpy as np
from typing import Union

from utils import PFSProblem
from SimulatedAnnealing import SimulatedAnnealing
from MemeticAlgorithm import MemeticAlgorithm


def main(
    data_dir: str = "data/tai20_5_1.txt",
    init_ls_temperature: Union[float, int] = 5000,
    init_ls_epoch_len: int = 10,
    init_ls_epoch_alpha: float = 0.98,
    init_ls_ratio: float = 0.2,
    population_size: int = 20,
    init_k: int = 5,
    tournament_k: int = 2,
    offspring_m: int = 2,
    mutate_prob: float = 0.2,
    end_ls_temperature: Union[float, int] = 5000,
    end_ls_epoch_len: int = 10,
    end_ls_epoch_alpha: float = 0.98,
    end_ls_ratio: float = 0.2,
    num_iter: int = 10,
    verbose: bool = False,
):
    p = PFSProblem(data_dir)
    init_sa = SimulatedAnnealing(
        init_ls_temperature, init_ls_epoch_len, init_ls_epoch_alpha, 1
    )
    end_sa = SimulatedAnnealing(
        end_ls_temperature, end_ls_epoch_len, end_ls_epoch_alpha, 1
    )
    ma = MemeticAlgorithm(
        p_size=population_size,
        init_k=init_k,
        init_ls=init_sa,
        init_ls_ratio=init_ls_ratio,
        tournament_k=tournament_k,
        offspring_m=offspring_m,
        mutate_prob=mutate_prob,
        end_ls=end_sa,
        end_ls_ratio=end_ls_ratio,
    )

    results = ma.search(p, num_iter=num_iter, verbose=verbose)
    results.evaluate_and_sort(problem=p)

    for i, result in enumerate(results):
        print(f"{i + 1}-th solution:{result}")
        print(f"Makespan:{result.makespan}\n")


if __name__ == "__main__":
    fire.Fire(main)
