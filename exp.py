import fire
from typing import Union

from utils import TFSProblem, Solution
from SimulatedAnnealing import SimulatedAnnealing


def main(
    data_dir: str = "data/tai20_5_1.txt",
    epoch_len: int = 1,
    alpha: float = 0.99,
    stopcriterion: Union[float, int] = 1,
    temperature: Union[float, int] = 1000,
    verbose: bool = False,
):
    p = TFSProblem(data_dir)

    sa = SimulatedAnnealing(epoch_len, alpha, stopcriterion)

    solution, search_steps = sa.search(p, temperature, verbose=verbose)

    print(solution, search_steps)


if __name__ == "__main__":
    fire.Fire(main)
