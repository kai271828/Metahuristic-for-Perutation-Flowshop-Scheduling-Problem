import concurrent.futures
import fire
import numpy as np
from typing import Union


from utils import PFSProblem
from SimulatedAnnealing import SimulatedAnnealing
from MemeticAlgorithm import MemeticAlgorithm


def main(
    data_dir: str = "data/tai20_5_1.txt",
    init_sa_temperature: Union[float, int] = 5000,
    init_sa_epoch_len: int = 10,
    init_sa_alpha: float = 0.98,
    init_sa_ratio: float = 0.2,
    population_size: int = 20,
    init_k: int = 5,
    tournament_k: int = 2,
    offspring_m: int = 2,
    mutate_prob: float = 0.2,
    end_sa_temperature: Union[float, int] = 5000,
    end_sa_epoch_len: int = 10,
    end_sa_alpha: float = 0.98,
    end_sa_ratio: float = 0.2,
    num_iter: int = 10,
    times: int = 20,
    verbose: bool = False,
):
    p = PFSProblem(data_dir)
    init_sa = SimulatedAnnealing(
        init_sa_temperature, init_sa_epoch_len, init_sa_alpha, 1
    )
    end_sa = SimulatedAnnealing(
        init_sa_temperature, init_sa_epoch_len, init_sa_alpha, 1
    )
    ma = MemeticAlgorithm(
        p_size=population_size,
        init_k=init_k,
        init_ls=init_sa,
        init_ls_ratio=init_sa_ratio,
        tournament_k=tournament_k,
        offspring_m=offspring_m,
        mutate_prob=mutate_prob,
        end_ls=end_sa,
        end_ls_ratio=end_sa_ratio,
    )

    best = np.inf
    best_sol = None
    worst = 0
    worst_sol = None
    makespan_record = []
    diversity_record = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(times):

            print(f"[Experiment {i + 1}]")
            results, ma_makespan_record, ma_diversity_record = ma.search(
                p, num_iter=num_iter, verbose=verbose
            )
            results.evaluate_and_sort(problem=p)

            print(f"Best makespan evolution in this experiment: {ma_makespan_record}")
            print(f"Diversity evolution in this experiment: {ma_diversity_record}")

            diversity_record.append(results.diversity)
            makespan_record.append(results[0].makespan)
            if results[0].makespan < best:
                best = results[0].makespan
                best_sol = results[0]
            if results[0].makespan > worst:
                worst = results[0].makespan
                worst_sol = results[0]

    mean = np.mean(makespan_record)
    std = np.std(makespan_record)

    print(f"\n\nBest solutin {best_sol} has makespan {best}")
    print(f"Worst solutin {worst_sol} has makespan {worst}")
    print(f"Mean: {mean}, Std: {std}\n")
    print(f"Makespan record: {makespan_record}")
    print(f"Diversity record: {diversity_record}")


if __name__ == "__main__":
    fire.Fire(main)
