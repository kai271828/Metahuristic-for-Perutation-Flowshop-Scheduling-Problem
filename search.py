import fire
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

from exp import main


def search(
    data_dir: str = "data/tai20_5_1.txt",
    epoch_len=[1, 11],
    alpha=[0.8, 0.99],
    stopcriterion=[1, 2],
    temperature=[100, 10000],
    times=20,
    search_times=10000,
    metric="avg",
):
    best_performance = np.inf

    for i in tqdm(range(search_times)):
        param = {
            "el": np.random.randint(epoch_len[0], epoch_len[1]),
            "al": alpha[0] + np.random.rand() * (alpha[1] - alpha[0]),
            "sc": np.random.randint(stopcriterion[0], stopcriterion[1]),
            "t": np.random.randint(temperature[0], temperature[1]),
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
        )

        if metric == "avg" and avg < best_performance:
            best_performance = avg
            best_parm = param
        elif metric == "best" and best < best_performance:
            best_performance = best
            best_parm = param
        elif metric == "worst" and worst < best_performance:
            best_performance = worst
            best_parm = param

    print(
        f"""The best setting is 
    epoch_len={best_param['el']},  
    alpha={best_param['al']},
    stopcriterion={best_param['sc']},
    temperature={best_param['t']}"""
    )


if __name__ == "__main__":
    fire.Fire(search)
