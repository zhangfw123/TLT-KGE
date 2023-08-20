# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        # import ipdb
        # ipdb.set_trace()
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    
                    these_queries = queries[b_begin:b_begin + batch_size]
                    if self.model_name in ['TNTComplEx' , 'TComplEx']:
                        q = self.get_queries(these_queries)
                        scores = q@rhs
                    elif self.model_name == 'TLT_KGE_Quaternion':
                        (a,b), time_ent = self.get_queries(these_queries)
                        scores = a@rhs + torch.sum(b*time_ent,dim=1).unsqueeze(1) + b@rhs + torch.sum(a*time_ent,dim=1).unsqueeze(1)
                    else:
                        q, tmp = self.get_queries(these_queries)
                        scores = q[0]@rhs + torch.sum(q[1]*tmp, dim=1).unsqueeze(1)
                    targets = self.score(these_queries)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"
                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    all_ts_ids = torch.arange(0, scores.shape[1]).to('cpu')[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            import ipdb
            # ipdb.set_trace()
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores = q @ rhs
                # scores = q
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks

class TLT_KGE_Quaternion(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3, cycle=365, is_cuda:bool=False
    ):
        super(TLT_KGE_Quaternion, self).__init__()
        self.model_name = "TLT_KGE_Quaternion"
        self.cycle = cycle
        self.sizes = sizes
        self.rank = rank
        self.pi = 3.14159265358979323846
        self.lam = 1
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2*rank, sparse=True),
            nn.Embedding(sizes[1], 2*rank, sparse=True),
            nn.Embedding(sizes[3] + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3] + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3] + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3]//self.cycle + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3]//self.cycle + 1, 2*rank, sparse=True),
        ])
        self.is_cuda = is_cuda
        
        if rank % 2 != 0:
            raise "rank need to be devided by 2.."
        for i in range(len(self.embeddings)):
            torch.nn.init.xavier_uniform_(self.embeddings[i].weight.data)
    @staticmethod
    def has_time():
        return False
    def quaternion_mul(self, emb1, emb2):
        a,b,c,d = torch.chunk(emb1, 4, dim=1)
        e,f,g,h = torch.chunk(emb2, 4, dim=1)
        return torch.cat([(a*e - b*f - c*g - d*h), (b*e + a*f + c*h - d*g), (c*e + a*g + d*f - b*h), (d*e + a*h + b*g - c*f)], dim=1)
    def forward_over_time(self, x):
        raise NotImplementedError("no.")
    def complex_mul(self, emb1, emb2):
        a,b = torch.chunk(emb1, 2, dim=1)
        c,d = torch.chunk(emb2, 2, dim=1)
        return torch.cat(((a*c + b*d), (a*d - b*c)), dim=1) 
    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        base_ent = self.embeddings[6](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        base = self.embeddings[5](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        comp_time = self.embeddings[4](x[:, 3])  
        time = self.embeddings[2](x[:, 3]) + base
        time_ent = self.embeddings[3](x[:, 3]) +base_ent
        rel_ = self.complex_mul(rel, comp_time)
        rel_ = rel + rel_
        lhs_rel = self.quaternion_mul(torch.cat([lhs, time_ent], dim=1), torch.cat([rel_, time], dim=1))
        a,b = lhs_rel[:, :self.rank*2], lhs_rel[:, self.rank*2:]
        return torch.sum(
            a*rhs + b*time_ent + a*time_ent + b*rhs,
            1, keepdim=True
        )
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        
        base_ent = self.embeddings[6](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        base = self.embeddings[5](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        comp_time = self.embeddings[4](x[:, 3])  
        time = self.embeddings[2](x[:, 3]) + base
        time_ent = self.embeddings[3](x[:, 3]) +base_ent
        rel_ = self.complex_mul(rel, comp_time)
        rel_ = rel + rel_
        lhs_rel = self.quaternion_mul(torch.cat([lhs, time_ent], dim=1), torch.cat([rel_, time], dim=1))
        right = self.embeddings[0].weight
        div = 2
        lhs_de = torch.chunk(lhs, div, dim=1)
        rel_de = torch.chunk(rel, div, dim=1)
        rhs_de = torch.chunk(rhs, div, dim=1)
        l_n = 2
        lhs_r = lhs_de[0] ** l_n
        rel_r = rel_de[0] ** l_n
        rhs_r = rhs_de[0] ** l_n
        a,b = lhs_rel[:, :self.rank*2], lhs_rel[:, self.rank*2:]
        for i in range(1, div):
            lhs_r = lhs_r + lhs_de[i] ** l_n
            rhs_r = rhs_r + rhs_de[i] ** l_n
            rel_r = rel_r + rel_de[i] ** l_n
        return (
                    a@right.transpose(0, 1) + torch.sum(b*time_ent,dim=1).unsqueeze(1) + b@right.transpose(0,1) + torch.sum(a*time_ent,dim=1).unsqueeze(1)
               ), (
                   torch.sqrt(lhs_r),
                   torch.sqrt(rel_r),
                   torch.sqrt(rhs_r),
               ), (self.embeddings[2].weight[:-1],
               self.embeddings[3].weight[:-1],
               self.embeddings[4].weight[:-1],
               self.embeddings[5].weight[:-1],
               self.embeddings[6].weight[:-1],
               )
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)
    def get_queries(self, queries: torch.Tensor):

        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        
        base_ent = self.embeddings[6](torch.div(queries[:, 3], self.cycle, rounding_mode='floor'))
        base = self.embeddings[5](torch.div(queries[:, 3], self.cycle, rounding_mode='floor'))
        comp_time = self.embeddings[4](queries[:, 3])  
        time = self.embeddings[2](queries[:, 3]) + base
        time_ent = self.embeddings[3](queries[:, 3]) +base_ent
        rel_ = self.complex_mul(rel, comp_time)
        rel_ = rel + rel_
        lhs_rel = self.quaternion_mul(torch.cat([lhs, time_ent], dim=1), torch.cat([rel_, time], dim=1))
        lhs_rel = lhs_rel[:, :self.rank*2], lhs_rel[:, self.rank*2:]
        return lhs_rel, time_ent

class TLT_KGE_Complex(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3, cycle=365, is_cuda = False
    ):
        super(TLT_KGE_Complex, self).__init__()
        self.cycle = cycle
        self.model_name = "TLT_KGE_Complex"
        self.sizes = sizes
        self.rank = rank
        self.pi = 3.14159265358979323846
        self.linear = nn.Linear(rank, rank//2)
        self.sig = nn.Sigmoid()
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2*rank, sparse=True),
            nn.Embedding(sizes[1], 2*rank, sparse=True),
            nn.Embedding(sizes[0], 2*rank, sparse=True),
            nn.Embedding(sizes[3] + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3] + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3] + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3]//self.cycle + 1, 2*rank, sparse=True),
            nn.Embedding(sizes[3]//self.cycle + 1, 2*rank, sparse=True),
        ])
        self.is_cuda = is_cuda
        if rank % 2 != 0:
            raise "rank need to be devided by 2.."
        for i in range(len(self.embeddings)):
            torch.nn.init.xavier_uniform_(self.embeddings[i].weight.data)
    @staticmethod
    def has_time():
        return False
    def forward_over_time(self, x):
        raise NotImplementedError("no.")
    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        base_ent = self.embeddings[6](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        base = self.embeddings[7](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        time = self.embeddings[4](x[:, 3]) +base
        time_ent = self.embeddings[3](x[:, 3]) +base_ent
        comp_time = self.embeddings[5](x[:, 3])  
        rel_ = rel*comp_time 
        rel_ = rel + rel_
        lhs_rel_1 = self.opt(lhs, rel_) - self.opt(time_ent, time)  +  (self.opt(lhs, time) + self.opt(time_ent, rel_)) 
        lhs_rel_2 = -self.opt(lhs, rel_) + self.opt(time_ent, time)  +  (self.opt(lhs, time) + self.opt(time_ent, rel_)) 
        return torch.sum(
            lhs_rel_1*(rhs) + lhs_rel_2*(time_ent ),
            1, keepdim=True
        )
    def opt(self, emb1, emb2):
        return emb1*emb2
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        base_ent = self.embeddings[6](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        base = self.embeddings[7](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        time = self.embeddings[4](x[:, 3]) +base
        time_ent = self.embeddings[3](x[:, 3]) +base_ent
        comp_time = self.embeddings[5](x[:, 3])  
        rel_ = rel*comp_time
        rel_ = rel + rel_
        lhs_rel_1 = self.opt(lhs, rel_) - self.opt(time_ent, time)  +  (self.opt(lhs, time) + self.opt(time_ent, rel_)) 
        lhs_rel_2 = -self.opt(lhs, rel_) + self.opt(time_ent, time)  +  (self.opt(lhs, time) + self.opt(time_ent, rel_)) 
        right = self.embeddings[0].weight
        div = 4
        lhs_de = torch.chunk(lhs, div, dim=1)
        rel_de = torch.chunk(rel, div, dim=1)
        rhs_de = torch.chunk(rhs, div, dim=1)
        l_n = 2
        lhs_r = lhs_de[0] ** l_n
        rel_r = rel_de[0] ** l_n
        rhs_r = rhs_de[0] ** l_n
        for i in range(1, div):
            lhs_r = lhs_r + lhs_de[i] ** l_n
            rhs_r = rhs_r + rhs_de[i] ** l_n
            rel_r = rel_r + rel_de[i] ** l_n
        return (
                    (lhs_rel_1@((right).transpose(0, 1))) 
                    + torch.sum(lhs_rel_2*((time_ent )), dim=1).unsqueeze(1)
               ), (
                   torch.sqrt(lhs_r),
                   torch.sqrt(rel_r),
                   torch.sqrt(rhs_r),
               ), (self.embeddings[3].weight[:-1],
               self.embeddings[4].weight[:-1],
               self.embeddings[5].weight[:-1],
               self.embeddings[6].weight[:-1],
               self.embeddings[7].weight[:-1],
               )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)
    def complex_mul(self, emb1, emb2):
        a,b = torch.chunk(emb1, 2, dim=1)
        c,d = torch.chunk(emb2, 2, dim=1)
        return torch.cat(((a*c - b*d), (a*d + b*c)), dim=1)
    def rotate(self, emb_re, emb_im, rot_re, rot_im):
        return torch.cat((emb_re*rot_re + emb_im * rot_im, emb_im*rot_re - emb_re*rot_im), dim=-1)

    def get_queries(self, queries: torch.Tensor):

        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        base_ent = self.embeddings[6](torch.div(queries[:, 3], self.cycle, rounding_mode='floor'))
        base = self.embeddings[7](torch.div(queries[:, 3], self.cycle, rounding_mode='floor'))
        time = self.embeddings[4](queries[:, 3]) +base
        time_ent = self.embeddings[3](queries[:, 3]) +base_ent
        comp_time = self.embeddings[5](queries[:, 3])  
        rel_ = rel*comp_time
        rel_ = rel + rel_
        lhs_rel_1 = self.opt(lhs, rel_) - self.opt(time_ent, time)  +  (self.opt(lhs, time) + self.opt(time_ent, rel_))
        lhs_rel_2 = -self.opt(lhs, rel_) + self.opt(time_ent, time)  +  (self.opt(lhs, time) + self.opt(time_ent, rel_))
        return (lhs_rel_1, lhs_rel_2), time_ent 

class TComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, is_cuda:bool=False
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank    
        self.model_name = "TComplEx"

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.is_cuda=is_cuda

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)


class TNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, is_cuda:bool=False
    ):
        super(TNTComplEx, self).__init__()
        self.model_name = "TNTComplEx"
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.is_cuda=is_cuda

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)