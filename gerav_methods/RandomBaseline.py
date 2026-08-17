"""Random baseline for authorship verification.

Predicts same/different with a uniform random score in [0, 1), independent
of the input texts. Used as a naive floor alongside AdHominem, CharNGram,
PPM, and SiameseBert in the central results CSV.
"""
import argparse
import os
import random
import time

import numpy as np

from valla.dsets.loaders import get_av_dataset_json
from valla.utils.eval_metrics import av_metrics
from valla.utils.resource_metrics import sample_cpu_memory_gb, sample_gpu_memory_gb, summarize_memory_samples
from valla.utils.central_results import update_central_results, DATASETS


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    test_dataset = get_av_dataset_json(args.test_path)
    labels = [label for label, _, _ in test_dataset]

    probas, eval_times, cpu_memories, gpu_memories = [], [], [], []
    for _ in labels:
        start = time.time()
        proba = random.random()
        eval_times.append(time.time() - start)
        cpu_memories.append(sample_cpu_memory_gb())
        gpu_memories.append(sample_gpu_memory_gb())
        probas.append(proba)

    avg_pred_time_sec = float(np.mean(eval_times))
    mem_stats = summarize_memory_samples(cpu_memories, gpu_memories)
    res = av_metrics(labels, probas=probas, threshold=0.5)

    os.makedirs(args.out_path, exist_ok=True)
    results_txt = os.path.join(args.out_path, 'results.txt')
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write('===AV results===\n')
        for k, v in res.items():
            f.write(f'{k}: {v:.4f}\n')
        f.write('\n===Efficiency===\n')
        f.write(f'avg_prediction_time_sec: {avg_pred_time_sec:.6g}\n')
        for k, v in mem_stats.items():
            f.write(f'{k}: {v:.4f}\n')

    if args.central_results_path:
        update_central_results(
            args.central_results_path,
            method='random',
            dataset=args.dataset,
            f1_score=res['f1'],
            average_time=avg_pred_time_sec,
            average_memory=mem_stats['avg_cpu_memory_gb'],
        )

    print('AV results:', res)
    print('Efficiency:', {'avg_prediction_time_sec': avg_pred_time_sec, **mem_stats})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random baseline for authorship verification')
    parser.add_argument('--train_path', type=str, default=None,
                        help='unused, accepted for CLI parity with the other baselines')
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=DATASETS, default='gerav',
                        help='which benchmark dataset this run evaluates, used to key the central results CSV')
    parser.add_argument('--central_results_path', type=str, default=None,
                        help="if set, upsert this run's metrics into the central results CSV at this path")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
