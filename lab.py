import fire
import numpy as np
from numba import jit
from typing import Union

from utils import TFSProblem, Solution
from MA import MA


@jit
def main(
    data_dir="data/tai20_5_1.txt",
    cross_prob=0.5,
    muta_prob=0.2,
    epoch_len=10,
    alpha=0.9,
    temperature=3500,
):
    p = TFSProblem(data_dir)
    ma = MA(cross_prob, muta_prob, epoch_len, alpha, temperature)
    results = ma.search(p)

    results = sorted(
        [(p.evaluate(element.sol), element) for element in results], key=lambda x: x[0]
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    fire.Fire(main)
