import fire
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

from hw1 import main


def search(
    data_dir: str = "data/tai20_5_1.txt",
    min_epoch_len=1,
    max_epoch_len=11,
    min_alpha=0.8,
    max_alpha=0.99,
    min_stopcriterion=1,
    max_stopcriterion=2,
    min_temperature=100,
    max_temperature=10000,
    times=20,
    search_times=10000,
    metric="avg",
):
    best_performance = np.inf

    for i in tqdm(range(search_times)):
        param = {
            "el": np.random.randint(min_epoch_len, max_epoch_len),
            "al": min_alpha + np.random.rand() * (max_alpha - min_alpha),
            "sc": np.random.randint(min_stopcriterion, max_stopcriterion),
            "t": np.random.randint(min_temperature, max_temperature),
        }

        best, avg, worst = main(
            data_dir=data_dir,
            epoch_len=param["el"],
            alpha=param["al"],
            stopcriterion=param["sc"],
            temperature=param["t"],
            times=times,
            log_dir=None,
            verbose=False,
            run_search=True,
        )

        if metric == "avg" and avg < best_performance:
            best_performance = avg
            best_param = param
        elif metric == "best" and best < best_performance:
            best_performance = best
            best_param = param
        elif metric == "worst" and worst < best_performance:
            best_performance = worst
            best_param = param

    print(
        f"""The best setting is 
    epoch_len={best_param['el']},  
    alpha={best_param['al']},
    stopcriterion={best_param['sc']},
    temperature={best_param['t']}"""
    )


if __name__ == "__main__":
    fire.Fire(search)
