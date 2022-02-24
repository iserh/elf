from collections import deque
from datetime import datetime
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import psutil

output_dir = Path("data") / "performance"
output_dir.mkdir(parents=True, exist_ok=True)

SYNC_INTERVAL = 1
PLOT_INTERVAL = 5
LOG_INTERVAL = 60 * 20


def plot(x, y, fpath: Path):
    _, ax = plt.subplots(1, 1, figsize=(15, 5), tight_layout=True)
    ax.set(
        xlabel="time [sec]",
        ylabel="usage [%]",
        ylim=(0, 100),
        xlim=(0, LOG_INTERVAL),
    )
    ax.fill_between(range(0, len(perc), SYNC_INTERVAL), perc, label="memory")
    ax.legend()
    ax.grid(True)
    plt.savefig(fpath)
    plt.close()


perc = deque([0] * (LOG_INTERVAL+1), maxlen=LOG_INTERVAL+1)
k = 0
while True:
    k += 1
    i = 0
    for j in range(LOG_INTERVAL):
        i += 1
        perc.append(psutil.virtual_memory().percent)

        sleep(SYNC_INTERVAL)

        if i % PLOT_INTERVAL == 0:
            plot(range(0, len(perc), SYNC_INTERVAL), perc, output_dir / "live.pdf")

    str_time = datetime.strftime(datetime.now(), "%y-%m-%d--%H:%M:%S")
    plot(range(0, len(perc), SYNC_INTERVAL), perc, output_dir / (str_time + ".pdf"))
