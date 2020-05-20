import os, io
import argparse
import subprocess
from time import strftime, localtime
import time
import pandas as pd
import numpy as np
import random, pickle
from tqdm import tqdm
import torch
import modeling
import Data
from pyNTCIREVAL import Labeler
from pyNTCIREVAL.metrics import MSnDCG, nERR, nDCG, AP, RR
import collections

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

MODEL_MAP = {
    'crossbert' : modeling.CrossBert,
}


def main(model, dataset, train_pairs, qrels, valid_run, test_run, model_out_dir, qrelDict, modelName, fold,
         metricKeys, MAX_EPOCH, data, args):
    LR = 0.001
    BERT_LR = 2e-5

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)
    # optimizer = torch.optim.Adam([non_bert_params], lr=LR)

    top_valid_score = None
    bestResults = {}
    bestPredictions = []
    bestQids = []

    print("Fold: %d" % fold)

    if args.model in ["unsup"]:

        test_qids, test_results, test_predictions = validate(model, dataset, test_run, qrelDict, 0,
                                                             model_out_dir, data, args, "test")

        print(test_results["ndcg@15"])
        txt = 'new top validation score, %.4f' % np.mean(test_results["ndcg@10"])
        print2file(args.out_dir, modelName, ".txt", txt, fold)

        bestResults = test_results
        bestPredictions = test_predictions
        bestQids = test_qids
        pass
    else:
        for epoch in range(MAX_EPOCH):
            t2 = time.time()
            loss = train_iteration(model, optimizer, dataset, train_pairs, qrels, data, args)
            txt = f'train epoch={epoch} loss={loss}'
            print2file(args.out_dir, modelName, ".txt", txt, fold)

            valid_qids, valid_results, valid_predictions = validate(model, dataset, valid_run, qrelDict, epoch,
                                                                    model_out_dir, data, args, "valid")

            # valid_score = np.mean(valid_results["rp"])
            valid_score = np.mean(valid_results["ndcg@10"])
            elapsed_time = time.time() - t2
            txt = f'validation epoch={epoch} score={valid_score} : {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
            print2file(args.out_dir, modelName, ".txt", txt, fold)
            if top_valid_score is None or valid_score > top_valid_score:
                top_valid_score = valid_score
                # model.save(os.path.join(model_out_dir, 'weights.p'))
                test_qids, test_results, test_predictions = validate(model, dataset, test_run, qrelDict, epoch,
                                                                     model_out_dir, data, args, "test")

                # print(test_results["ndcg@15"])
                txt = 'new top validation score, %.4f' % np.mean(test_results["ndcg@10"])
                print2file(args.out_dir, modelName, ".txt", txt, fold)

                bestResults = test_results
                bestPredictions = test_predictions
                bestQids = test_qids

            # elif args.earlystop and epoch >=4:
            elif args.earlystop:
                break


    #   save outputs to files

    for k in metricKeys:
        result2file(args.out_dir, modelName, "." + k, bestResults[k], bestQids, fold)

    prediction2file(args.out_dir, modelName, ".out", bestPredictions, fold)

    print2file(args.out_dir, modelName, ".txt", txt, fold)
    return bestResults


def train_iteration(model, optimizer, dataset, train_pairs, qrels, data, args):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 32 if "eai" in args.data else 256
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in Data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE, data, args):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'],
                           record['wiki_tok'],
                           record['wiki_mask'],
                           record['question_tok'],
                           record['question_mask'])

            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])  # pariwse softmax
            loss.backward()

            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss
            # break


def validate(model, dataset, run, qrel, epoch, model_out_dir, data, args, desc):
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    return run_model(model, dataset, run, runf, qrel, data, args, desc)


def run_model(model, dataset, run, runf, qrels, data, args, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in Data.iter_valid_records(model, dataset, run, BATCH_SIZE, data, args):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'],
                           records['wiki_tok'],
                           records['wiki_mask'],
                           records['question_tok'],
                           records['question_mask'])

            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
            # break

    res = {"%s@%d" % (i, j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10, 15, 20]}
    res['map'] = []
    res['mrr'] = []
    res['rp'] = []
    predictions = []
    qids = []

    for qid in rerank_run:
        ranked_list_scores = sorted(rerank_run[qid].items(), key=lambda x: x[1], reverse=True)
        ranked_list = [i[0] for i in ranked_list_scores]
        for (pid, score) in ranked_list_scores:
            predictions.append((qid, pid, score))
        result = eval(qrels[qid], ranked_list)
        for key in res:
            res[key].append(result[key])
        qids.append(qid)
    return qids, res, predictions


def eval(qrels, ranked_list):
    grades = [1, 2, 3, 4]  # a grade for relevance levels 1 and 2 (Note that level 0 is excluded)
    labeler = Labeler(qrels)
    labeled_ranked_list = labeler.label(ranked_list)
    rel_level_num = 5
    xrelnum = labeler.compute_per_level_doc_num(rel_level_num)
    result = {}

    for i in [5, 10, 15, 20]:
        metric = MSnDCG(xrelnum, grades, cutoff=i)
        result["ndcg@%d" % i] = metric.compute(labeled_ranked_list)

        nerr = nERR(xrelnum, grades, cutoff=i)
        result["nerr@%d" % i] = nerr.compute(labeled_ranked_list)

        _ranked_list = ranked_list[:i]
        result["p@%d" % i] = len(set.intersection(set(qrels.keys()), set(_ranked_list))) / len(_ranked_list)
        result["r@%d" % i] = len(set.intersection(set(qrels.keys()), set(_ranked_list))) / len(qrels)

    result["rp"] = len(set.intersection(set(qrels.keys()), set(ranked_list[:len(qrels)]))) / len(qrels)
    metric = MSnDCG(xrelnum, grades, cutoff=i)

    map = AP(xrelnum, grades)
    result["map"] = map.compute(labeled_ranked_list)
    mrr = RR()
    result["mrr"] = mrr.compute(labeled_ranked_list)

    return result


def write2file(path, name, format, output):
    print(output)
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    thefile.write("%s\n" % output)
    thefile.close()


def prediction2file(path, name, format, preds, fold):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    for (qid, pid, score) in preds:
        thefile.write("%d\t%s\t%s\t%f\n" % (fold, qid, pid, score))
    thefile.close()

def print2file(path, name, format, printout, fold):
    print(printout)
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    thefile.write("%d-%s\n" % (fold, printout))
    thefile.close()

def result2file(path, name, format, res, qids, fold):
    if not os.path.exists(path):
        os.makedirs(path)
    thefile = open(path + name + format, 'a')
    for q, r in zip(qids, res):
        thefile.write("%d\t%s\t%f\n" % (fold, q, r))
    thefile.close()

def main_cli():
    # argument
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='crossbert')
    parser.add_argument('--data', default='akgg')
    parser.add_argument('--path', default="data/")
    parser.add_argument('--wikifile', default="wikihow")
    parser.add_argument('--questionfile', default="question-qq")
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir', default="models/vbert")
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--out_dir', default="out/")
    parser.add_argument('--evalMode', default="all")
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--maxlen', type=int, default=16)
    parser.add_argument('--earlystop', type=int, default=1)

    args = parser.parse_args()

    args.queryfile = io.TextIOWrapper(io.open("%s%s-query.tsv" % (args.path, args.data.split("-")[0]),'rb'), 'UTF-8')
    args.docfile = io.TextIOWrapper(io.open("%s%s-doc.tsv" % (args.path, args.data.split("-")[0]),'rb'), 'UTF-8')
    args.wikifile = io.TextIOWrapper(io.open("%s%s-%s.tsv" % (args.path, args.data.split("-")[0], args.wikifile),'rb'), 'UTF-8')
    args.questionfile = io.TextIOWrapper(io.open("%s%s-%s.tsv" % (args.path, args.data.split("-")[0], args.questionfile),'rb'), 'UTF-8')

    args.train_pairs = "%s%s-train" % (args.path, args.data)
    args.valid_run = "%s%s-valid" % (args.path, args.data)
    args.test_run = "%s%s-test" % (args.path, args.data)

    args.qrels = io.TextIOWrapper(io.open("%s%s-qrel.tsv" % (args.path, args.data.split("-")[0]),'rb'), 'UTF-8')

    dataset = Data.read_datafiles([args.queryfile, args.docfile, args.wikifile,
                                   args.questionfile])
    args.dataset = dataset
    model = MODEL_MAP[args.model](args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[args.model](args)


    # if args.model == "cedr_pacrr":
    #     args.maxlen = 16 if args.mode == 1 else args.maxlen * args.mode
    #     model = MODEL_MAP[args.model](args).cuda() if Data.device.type == 'cuda' else MODEL_MAP[args.model](
    #         args)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    qrels = Data.read_qrels_dict(args.qrels)

    MAX_EPOCH = args.epoch

    train_pairs = []
    valid_run = []
    test_run = []

    foldNum = args.fold
    for fold in range(foldNum):
        f = open(args.train_pairs + "%d.tsv" % fold, "r")
        train_pairs.append(Data.read_pairs_dict(f))
        f = open(args.valid_run + "%d.tsv" % fold, "r")
        valid_run.append(Data.read_run_dict(f))
        f = open(args.test_run + "%d.tsv" % fold, "r")
        test_run.append(Data.read_run_dict(f))

    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    timestamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    if "birch" in args.model:
        wikiName = args.wikifile.name.split("/")[-1].replace(".tsv", "")
        questionName = args.questionfile.name.split("/")[-1].replace(".tsv", "")
        additionName = []
        if args.mode in [1, 3, 5, 6]:
            additionName.append(wikiName)
        if args.mode in [2, 4, 5, 6]:
            additionName.append(questionName)

        modelName = "%s_m%d_%s_%s_%s_e%d_es%d_%s" % (
            args.model, args.mode, args.data, "_".join(additionName), args.evalMode, args.epoch, args.earlystop, timestamp)
    else:
        wikipediaFile = args.wikifile.name.split("/")[-1].replace(".tsv", "")
        questionFile = args.questionfile.name.split("/")[-1].replace(".tsv", "")
        modelName = "%s_%s_m%d_ml%d_%s_%s_%s_e%d_es%d_%s" % (args.data, args.model, args.mode, args.maxlen, wikipediaFile, questionFile, args.evalMode, args.epoch, args.earlystop, timestamp)

    print(modelName)

    df = pd.read_csv("%s%s-qrel.tsv" % (args.path, args.data.split("-")[0]), sep="\t", names=["qid", "empty", "pid", "rele_label", "etype"])
    qrelDict = collections.defaultdict(dict)
    type2pids = collections.defaultdict(set)
    for qid, prop, label, etype in df[['qid', 'pid', 'rele_label', 'etype']].values:
        qrelDict[str(qid)][str(prop)] = int(label)
        type2pids[str(etype)].add(prop)
    args.type2pids = type2pids


    metricKeys = {"%s@%d" % (i, j): [] for i in ["p", "r", "ndcg", "nerr"] for j in [5, 10, 15, 20]}
    metricKeys["rp"] = []
    metricKeys["mrr"] = []
    metricKeys["map"] = []

    results = []

    t1 = time.time()


    args.isUnsupervised = True if args.model in ["sen_emb"] else False


    for fold in range(len(train_pairs)):
        results.append(
            main(model, dataset, train_pairs[fold], qrels, valid_run[fold], test_run[fold], args.model_out_dir,
                 qrelDict, modelName, fold, metricKeys, MAX_EPOCH, Data, args))
    elapsed_time = time.time() - t1
    txt = f'total : {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    print2file(args.out_dir, modelName, ".txt", txt, fold)


    #   average results across 5 folds
    output = []
    for k in metricKeys:
        tmp = []
        for fold in range(foldNum):
            tmp.extend(results[fold][k])
        _res = np.mean(tmp)
        output.append("%.4f" % _res)
    write2file(args.out_dir, modelName, ".res", ",".join(output))


if __name__ == '__main__':
    main_cli()
