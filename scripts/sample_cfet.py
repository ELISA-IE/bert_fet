import os
import json
from argparse import ArgumentParser
from collections import Counter
import random


def count_labels(input_file):
    label_count = Counter()
    with open(input_file) as r:
        for line in r:
            inst = json.loads(line)
            for annotation in inst['annotations']:
                label_count.update(annotation['labels'])
    for label, count in label_count.most_common(len(label_count)):
        print('{}: {}'.format(label, count))
    return label_count


def downsample(input_file: str,
               output_file: str,
               threshold: int = 10000
               ):
    total_num = sample_num = 0
    
    print('Counting labels')
    label_count = count_labels(input_file)

    print('Sampling')
    with open(input_file) as r, open(output_file, 'w') as w:
        for line in r:
            total_num += 1
            inst = json.loads(line)
            inst_label_count = [(label, label_count[label])
                                for annotation in inst['annotations']
                                for label in annotation['labels']]
            inst_label_count.sort(key=lambda x: x[1])
            min_label_count = inst_label_count[0][1]
            if min_label_count < threshold:
                w.write(line)
                sample_num += 1
            else:
                ratio = threshold + (min_label_count - threshold) ** .55
                if random.uniform(0, 1) < ratio / min_label_count:
                    w.write(line)
                    sample_num += 1
    print('#Total: {}'.format(total_num))
    print('#Sample: {}'.format(sample_num))


def split(input_file: str,
          train_file: str,
          dev_file: str,
          threshold: int = 20,
          rate: float = .01):
    label_count = Counter()
    print('Count labels')
    with open(input_file) as r:
        for line in r:
            inst = json.loads(line)
            annotations = inst['annotations']
            for annotation in annotations:
                label_count.update(annotation['labels'])

    infreq_labels = {label for label, count in label_count.items()
                     if count < threshold}

    print('Sampling')
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(train_file, 'w', encoding='utf-8') as wt, \
        open(dev_file, 'w', encoding='utf-8') as wd:
        for line in r:
            inst = json.loads(line)
            infreq = any(label in infreq_labels
                         for annotation in inst['annotations']
                         for label in annotation['labels'])
            if infreq or random.uniform(0, 1) > rate:
                wt.write(line)
            else:
                wd.write(line)


if __name__ == '__main__':
    input_file = '/shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/en/en.aida+kairos.cfet.json'
    output_file = '/shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/en/en.aida+kairos.cfet.ds3.json'
    downsample(input_file, output_file, 50000)
    # count_labels(output_file)
    
    train_file = '/shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/en/en.aida+kairos.cfet.ds3.train.json'
    dev_file = '/shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/en/en.aida+kairos.cfet.ds3.dev.json'
    split(output_file, train_file, dev_file, 200, 0.005)
