#!/usr/bin/env python
# -*- coding: utf-8 -*-
# measure.py

import math
import numpy as np

from operator import itemgetter


class Measure(object):

    def __init__(self):
        pass

    @staticmethod
    def rating_measure(res):
        measure = []
        mae = Measure.MAE(res)
        measure.append('MAE:' + str(mae) + '\n')
        rmse = Measure.RMSE(res)
        measure.append('RMSE:' + str(rmse) + '\n')

        return measure

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = origin[user].keys()
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def ranking_measure(origin, res, N):
        measure = []
        for n in N:
            predicted = {}
            for user in origin.keys():
                predicted[user] = res[user][:n]
            indicators = []
            if len(origin) != len(predicted):
                print("The Lengths of test set and predicted set are not match!")
                exit(-1)
            hits = Measure.hits(origin, predicted)
            prec = Measure.precision(hits, n)
            indicators.append("Precision:" + str(prec) + "\n")
            recall = Measure.recall(hits, origin)
            indicators.append("Recall:" + str(recall) + "\n")
            F1 = Measure.F1(prec, recall)
            indicators.append("F1:" + str(F1) + "\n")
            MAP = Measure.MAP(origin, predicted, n)
            indicators.append("MAP:" + str(MAP) + "\n")
            NDCG = Measure.NDCG(origin, predicted, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            measure.append("Top " + str(n) + "\n")
            measure += indicators
        return measure

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return float(prec) / (len(hits) * N)

    @staticmethod
    def MAP(origin, res, N):
        sum_prec = 0
        for user in res:
            hits = 0
            precision = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user].keys():
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / (min(len(origin[user]), N) + 0.0)
        return sum_prec / (len(res))

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / (len(res))

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[-2] - entry[-1])
            count += 1
        if count == 0:
            return error
        return error / count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[-2] - entry[-1]) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(error / count)
