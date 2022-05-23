import os
import os.path
import json
import numpy as np

rel2id = json.load(open('meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}


def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(
        truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(
        os.path.join(path, "train_annotated.json"), truth_dir)
    fact_in_train_distant = gen_train_facts(
        os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, "dev.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    correct_re_ = [0, 0, 0, 0, 0]
    answer_ = [0, 0, 0, 0, 0]
    fact_ = [0, 0, 0, 0, 0]

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])
            mention_pairs_cnt = len(vertexSet[h_idx]) * len(vertexSet[t_idx])
            if mention_pairs_cnt == 1:
                fact_[0] += 1
            elif mention_pairs_cnt == 2:
                fact_[1] += 1
            elif mention_pairs_cnt == 3:
                fact_[2] += 1
            elif mention_pairs_cnt == 4:
                fact_[3] += 1
            else:
                fact_[4] += 1

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0

    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0

    correct_in_train_annotated_ = [0, 0, 0, 0, 0]
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        mention_pairs_cnt = len(vertexSet[h_idx]) * len(vertexSet[t_idx])
        if mention_pairs_cnt == 1:
            answer_[0] += 1
        elif mention_pairs_cnt == 2:
            answer_[1] += 1
        elif mention_pairs_cnt == 3:
            answer_[2] += 1
        elif mention_pairs_cnt == 4:
            answer_[3] += 1
        else:
            answer_[4] += 1

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1

            if mention_pairs_cnt == 1:
                correct_re_[0] += 1
            elif mention_pairs_cnt == 2:
                correct_re_[1] += 1
            elif mention_pairs_cnt == 3:
                correct_re_[2] += 1
            elif mention_pairs_cnt == 4:
                correct_re_[3] += 1
            else:
                correct_re_[4] += 1

            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
                if mention_pairs_cnt == 1:
                    correct_in_train_annotated_[0] += 1
                elif mention_pairs_cnt == 2:
                    correct_in_train_annotated_[1] += 1
                elif mention_pairs_cnt == 3:
                    correct_in_train_annotated_[2] += 1
                elif mention_pairs_cnt == 4:
                    correct_in_train_annotated_[3] += 1
                else:
                    correct_in_train_annotated_[4] += 1

            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    # calculate   f1 for different mentions pairs cnt
    re_f1_ = []
    for i in range(5):
        p = 1.0 * correct_re_[i] / answer_[i]
        r = 1.0 * correct_re_[i] / fact_[i]
        if p + r == 0:
            re_f1_.append(0.)
        else:
            re_f1_.append(2.0 * p * r / (p + r))

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    # calculate  ign_f1 for different mentions pairs cnt
    re_f1_ignore_train_annotated_ = []
    for i in range(5):
        p = 1.0 * (correct_re_[i] - correct_in_train_annotated_[i]) / (answer_[i] - correct_in_train_annotated_[i] + 1e-5)
        r = 1.0 * correct_re_[i] / fact_[i]
        if p + r == 0:
            re_f1_ignore_train_annotated_.append(0.)
        else:
            re_f1_ignore_train_annotated_.append(2.0 * p * r / (p + r))

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_f1_[0], re_f1_[1], re_f1_[2], re_f1_[3], re_f1_[4], re_f1_ignore_train_annotated_[0], re_f1_ignore_train_annotated_[1], re_f1_ignore_train_annotated_[2], re_f1_ignore_train_annotated_[3], re_f1_ignore_train_annotated_[4]
