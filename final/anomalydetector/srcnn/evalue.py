"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation ("Microsoft") grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
from srcnn.competition_metric import get_variance, evaluate_for_all_series
import time
import json
import argparse
from msanomalydetector.spectral_residual import SpectralResidual
from srcnn.utils import *


def auto():
    path_auto = os.getcwd() + '/auto.json'
    with open(path_auto, 'r+') as f:
        store = json.load(f)
    window = store['window']
    epoch = store['epoch']
    return window, epoch


def getfid(path):
    return path.split('/')[-1]


def get_path(data_source):
    if data_source == 'kpi':
        dir_ = root + '/Test/'
        trainfiles = [dir_ + _ for _ in os.listdir(dir_)]
        files = trainfiles
    else:
        dir_ = root + '/' + data_source + '/'
        files = [dir_ + _ for _ in os.listdir(dir_)]
    return files


def get_score(data_source, files, thres, option):
    total_time = 0
    results = []
    savedscore = []
    for f in files:
        print('reading', f)
        if data_source == 'kpi' or data_source == 'test_kpi' or f.lower().endswith('.csv'):
            in_timestamp, in_value, in_label = read_csv_kpi(f)
        else:
            tmp_data = read_pkl(f)
            in_timestamp, in_value, in_label = tmp_data['timestamp'], tmp_data['value'], tmp_data['label']
        length = len(in_timestamp)
        if model == 'sr_cnn' and len(in_value) < window:
            print("length is shorter than win_size", len(in_value), window)
            continue
        time_start = time.time()
        timestamp, label, pre, scores = models[model](in_timestamp, in_value, in_label, window, net, option, thres)
        time_end = time.time()
        total_time += time_end - time_start
        results.append([timestamp, label, pre, f])
        savedscore.append([label, scores, f, timestamp])
    return total_time, results, savedscore


def metric_dict(f1, pre, rec, TP, FP, TN, FN):
    return {
        'f1': float(f1),
        'precision': float(pre),
        'recall': float(rec),
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
    }


def evaluate_scores_at_threshold(savedscore, threshold, delay):
    newresults = []
    for flabel, cnnscores, path, ftimestamp in savedscore:
        pre = [1 if item > threshold else 0 for item in cnnscores]
        newresults.append([ftimestamp, flabel, pre, path])
    return evaluate_for_all_series(newresults, delay, prt=False), newresults


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='snapshot', help='model path')
    parser.add_argument('--delay', type=int, default=3, help='delay')
    parser.add_argument('--thres', type=float, default=0.95, help='initial threshold of SR')
    parser.add_argument('--auto', type=bool, default=False, help='Automatic filling parameters')
    parser.add_argument('--model', type=str, default='sr_cnn', help='model')
    parser.add_argument('--missing_option', type=str, default='anomaly',
                        help='missing data option, anomaly means treat missing data as anomaly')
    parser.add_argument('--result_dir', type=str, default='results', help='directory to save evaluation outputs')

    args = parser.parse_args()
    if args.auto:
        window, epoch = auto()
    else:
        window = args.window
        epoch = args.epoch
    data_source = args.data
    delay = args.delay
    model = args.model
    root = os.getcwd()
    print(data_source, window, epoch)
    models = {
        'sr_cnn': sr_cnn_eval,
    }

    model_path = root + '/' + args.model_path + '/srcnn_retry' + str(epoch) + '_' + str(window) + '.bin'
    srcnn_model = Anomaly(window)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_model(srcnn_model, model_path).to(device)
    files = get_path(data_source)
    total_time, results, savedscore = get_score(data_source, files, args.thres, args.missing_option)
    print('\n***********************************************')
    print('data source:', data_source, '     model:', model)
    print('-------------------------------')
    total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(results, delay)
    result_dir = root + '/' + args.result_dir + '/'
    os.makedirs(result_dir, exist_ok=True)
    saved_scores_path = result_dir + data_source + '_saved_scores.json'
    with open(saved_scores_path, 'w') as f:
        json.dump(savedscore, f)
    print('time used for making predictions:', total_time, 'seconds')


    best = 0.
    bestthre = 0.
    print('delay :', delay)
    if data_source == 'yahoo':
        sru = {}
        rf = open(data_source + 'sr3.json', 'r')
        srres = json.load(rf)
        for (srtime, srl, srpre, srf) in srres:
            sru[getfid(srf)] = [srtime, srl, srpre]
        for i in range(98):
            newresults = []
            threshold = 0.01 + i * 0.01
            for f, (srtt, srlt, srpret, srft), (flabel, cnnscores, cnnf, cnnt) in zip(files, srres, savedscore):
                fid = getfid(cnnf)
                srtime = sru[fid][0]
                srl = sru[fid][1]
                srpre = sru[fid][2]
                srtime = [(srtime[0] - 3600 * (64 - j)) for j in range(64)] + srtime
                srl = [0] * 64 + srl
                srpre = [0] * 64 + srpre
                print(len(srl), len(flabel), '!!')
                assert (len(srl) == len(flabel))
                pre = [1 if item > threshold else 0 for item in cnnscores]
                newresults.append([srtime, srpre, pre, f])
            total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(newresults, delay, prt=False)
            if total_fscore > best:
                best = total_fscore
                bestthre = threshold
        results = []
        threshold = bestthre
        print('guided threshold :', threshold)
        for f, (flabel, cnnscores, _, ftimestamp) in zip(files, savedscore):
            pre = [1 if item > threshold else 0 for item in cnnscores]
            results.append([ftimestamp, flabel, pre, f])
        print('score\n')
        total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(results, delay)
        print(total_fscore)
    best = 0.
    for i in range(98):
        threshold = 0.01 + i * 0.01
        threshold_metrics, newresults = evaluate_scores_at_threshold(savedscore, threshold, delay)
        if threshold_metrics[0] > best:
            best = threshold_metrics[0]
            bestthre = threshold
            print('tem best', best, threshold)
    threshold = bestthre
    best_metrics, best_results = evaluate_scores_at_threshold(savedscore, threshold, delay)
    print('best overall threshold :', threshold, 'best score :', best)
    print('best threshold metrics')
    print('precision', best_metrics[1])
    print('recall', best_metrics[2])
    print('f1', best_metrics[0])

    summary = {
        'data_source': data_source,
        'model': model,
        'window': window,
        'epoch': epoch,
        'delay': delay,
        'model_path': model_path,
        'files': files,
        'prediction_time_seconds': total_time,
        'initial_threshold': args.thres,
        'initial_metrics': metric_dict(total_fscore, pre, rec, TP, FP, TN, FN),
        'best_threshold': threshold,
        'best_metrics': metric_dict(*best_metrics),
        'saved_scores_path': saved_scores_path,
    }
    summary_path = result_dir + data_source + '_eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('saved evaluation summary:', summary_path)
    print('saved prediction scores:', saved_scores_path)
