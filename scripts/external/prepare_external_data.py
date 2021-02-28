from multiprocessing import Process
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

from hpa.data import load_channels


class DataTypeChanger:
    def __init__(self, data_dir, pid):
        self.data_dir = data_dir
        self.pid = pid

    def __call__(self, ids):
        bad_ids = []
        if self.pid == 0:
            generator = tqdm(ids, desc='Process 0')
        else:
            generator = ids
        for img_id in generator:
            try:
                channels = load_channels(img_id, self.data_dir)
                for color, channel in channels.items():
                    img = Image.fromarray(channel)
                    output_path = os.path.join(self.data_dir, f'{img_id}_{color}.png')
                    img.save(output_path)
            except:
                bad_ids.append(img_id)
        with open(f'/home/mchobanyan/bad_ids_{self.pid}.txt', 'w') as file:
            for i in bad_ids:
                file.write(str(i) + ',')


if __name__ == '__main__':

    NUM_JOBS = 2
    DATA_INDEX = '/home/mchobanyan/data/kaggle/hpa-single-cell/full-train-index.csv'
    DATA_DIR = '/home/mchobanyan/data/kaggle/hpa-single-cell/misc/public-hpa/data2/'

    data_idx = pd.read_csv(DATA_INDEX)
    data_idx = data_idx.loc[data_idx['Source'] == 'external']
    data_idx = data_idx.reset_index(drop=True)

    chunk_size = int(len(data_idx) / NUM_JOBS) + 1
    chunks = [data_idx['ID'].iloc[i:i + chunk_size].values.tolist() for i in range(0, len(data_idx), chunk_size)]

    processes = []
    for pid, chunk in enumerate(chunks):
        p = Process(target=DataTypeChanger(DATA_DIR, pid), args=(chunk,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
