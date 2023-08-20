# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim
import codecs

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import *
from regularizers import *
import sys
import os
import time
parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'TComplEx', 'TNTComplEx', 'TLT_KGE_Quaternion', 'TLT_KGE_Complex'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

parser.add_argument(
    '--res_path', type=str, default='default'
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--cycle', default=365, type=int,
    help="time range for sharing"
)

parser.add_argument(
    '--gpu', default=0, type=int,
    help="Use CUDA for training"
)

args = parser.parse_args()

save_path = "expe_log/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.model, args.rank, args.learning_rate, args.emb_reg, args.time_reg, args.cycle, int(time.time()))
if not os.path.exists(save_path):
    os.makedirs(save_path) 
dataset = TemporalDataset(args.dataset)
fw = codecs.open("{}/log.txt".format(save_path), 'w')
sizes = dataset.get_shape()
model = {
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb, is_cuda=True if args.gpu == 1 else False),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb, is_cuda=True if args.gpu == 1 else False),
    'TLT_KGE_Complex':TLT_KGE_Complex(sizes, args.rank, cycle=args.cycle, is_cuda=True if args.gpu == 1 else False),
    'TLT_KGE_Quaternion':TLT_KGE_Quaternion(sizes, args.rank, cycle=args.cycle, is_cuda=True if args.gpu == 1 else False),
}[args.model]
# in case a user want to train on a non-cuda machine
if args.gpu == 0:
    model = model.to('cpu')
else:
    model = model.cuda()

best_hits1 = 0
best_res_test = {}

opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)
emb_reg = N3(args.emb_reg)
if args.model in ['TComplEx', 'TNTComplEx']:
    time_reg = Lambda3(args.time_reg)
else:
    time_reg = Lambda3_two(args.time_reg)

add_reg = None
for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size, add_reg = add_reg, is_cuda = True if args.gpu == 1 else False
        )
        optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=args.batch_size, add_reg = add_reg, is_cuda=True if args.gpu == 1 else False
        )
        optimizer.epoch(examples)

    def avg_both(mr, mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor], added):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        mr = (mr['lhs'] + mr['rhs']) / 2.
        return {'MR':mr, 'MRR': m, 'hits@[1,3,10]': h, 'add':added}
    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        if dataset.has_intervals():
            valid, test = [
                dataset.eval(model, split, -1 if split != 'train' else 50000)
                for split in ['valid', 'test']
            ]
            print("valid: ", valid)
            print("test: ", test)
        else:
            valid, test = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test']
            ]
            print("valid: ", epoch, valid['MR'], valid['MRR'], valid['hits@[1,3,10]'])
            print("test: ", epoch, test['MR'], test['MRR'], test['hits@[1,3,10]'])
            fw.write("valid: epoch:{}, MR:{}, MRR:{}, Hist:{}\n".format(epoch, valid['MR'], valid['MRR'], valid['hits@[1,3,10]']))
            fw.write("test: epoch:{}, MR:{}, MRR:{}, Hist:{}\n".format(epoch, test['MR'], test['MRR'], test['hits@[1,3,10]']))
        if valid['hits@[1,3,10]'][0] > best_hits1:
            torch.save({'MRR':test['MRR'], 'Hist':test['hits@[1,3,10]'], 'MR':test['MR'], 'param':model.state_dict()}, '{}/best.pth'.format(save_path, args.model, args.dataset))
            print('best')
            best_hits1 = valid['hits@[1,3,10]'][0]
            best_res_test = [test['MR'], test['MRR'], test['hits@[1,3,10]']]

fw.write("{}\t{}\t{}\t{}\t{}\n".format(best_res_test[0], best_res_test[1], best_res_test[2][0], best_res_test[2][1], best_res_test[2][2]))
