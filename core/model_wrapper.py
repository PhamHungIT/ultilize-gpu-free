import logging
import os
from unittest import result

import numpy as np
import torch
import random
import traceback
from tqdm import tqdm
from tqdm.contrib import tzip
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from core import model
from core.utils import utils
from core.argument import train_with_argument, test_with_argument

logger = logging.getLogger()

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def run_train(dataset_file, model_path, config, use_gpu=False):
    samples, labels, answers, origins = load_dataset(dataset_file)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2407)

    for i, (train_index, test_index) in enumerate(skf.split(samples, labels), start=1):
        train_samples, test_samples = samples[train_index], samples[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        f_out = open('data/train_fold_' + str(i) +
                     '.txt', 'w', encoding='utf-8')
        for line in train_samples:
            f_out.write(line.strip() + '\n')

        idm = model.IntentDetector(hidden_size=int(config['hidden_size']), batch_size=int(config['batch_size']),
                                   dropout=float(config['dropout']), use_gpu=use_gpu, lr=float(config['lr']),
                                   num_epochs=int(config['num_epochs']))

        idm.run_train(train_samples, train_labels, answers, origins, model_path)

        idm.save_model(model_path)
        # print('Loading pre-trained model...')
        # test_idm = IntentDetector(config)
        # test_idm.load_model()
        # golds, preds = evaluation(test_idm, test_samples, test_labels)

        preds, golds = [], []
        for sample, label in zip(test_samples, test_labels):
            pred, results = idm.prediction(sample)
            preds.append(pred)
            golds.append(idm.label2index[label])

        accuracy = metrics.accuracy_score(golds, preds)
        p, r, f1, s = metrics.precision_recall_fscore_support(golds, preds,
                                                              average='weighted',
                                                              warn_for=[])

        accuracies.append(accuracy)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

        print('Testing fold: {}'.format(i))
        print('--- Fold %d: Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' %
              (i, accuracy, p, r, f1))
        break
    acc = np.mean(accuracies)
    p = np.mean(precisions)
    r = np.mean(recalls)
    f1 = np.mean(f1_scores)

    print('------- Final results: %0.3f, %0.3f, %0.3f, %0.3f' % (acc, p, r, f1))


def run_only_train(dataset_file, model_path, config, use_gpu=False):
    train_samples, train_labels, answers, origins = load_dataset(dataset_file)
    f_out = open('data/train.txt', 'w', encoding='utf-8')
    for line in train_samples:
        f_out.write(line.strip() + '\n')

    idm = model.IntentDetector(hidden_size=int(config['hidden_size']), batch_size=int(config['batch_size']),
                               dropout=float(config['dropout']), use_gpu=use_gpu, lr=float(config['lr']),
                               num_epochs=int(config['num_epochs']))

    idm.run_train(train_samples, train_labels, answers, origins, model_path)
    idm.save_model(model_path)


def run_train_on_json_data(
        json_data,
        model_path,
        config,
        use_gpu=False,
        progress_handler=None,
        version_id=None,
        job_id=None,
        use_semhash=False,
        remove_accents=True,
        training_with_augument=False,
        word2sent_path='',
        unknown_intent='UNKNOWN'
):
    questions, answers, origins = load_dataset_from_json(json_data)
    train_samples, train_labels, answers, origins = load_dataset(questions, answers, origins, use_semhash, remove_accents)
    idm = model.IntentDetector(use_semhash=use_semhash, hidden_size=int(config['hidden_size']), batch_size=int(config['batch_size']),
                               dropout=float(config['dropout']), use_gpu=use_gpu, lr=float(config['lr']),
                               num_epochs=int(config['num_epochs']))

    model_out = do_train(idm, train_samples, train_labels, answers,
                         origins, model_path, progress_handler, version_id, job_id)
    if model_out and model_out.is_stop:
        return model_out
    if training_with_augument:
        return train_with_argument(word2sent_path=word2sent_path,
                                   train_samples=train_samples,
                                   train_labels=train_labels,
                                   answers=answers,
                                   origins=origins,
                                   unknown_intent=unknown_intent,
                                   use_semhash=use_semhash,
                                   remove_accents=remove_accents,
                                   model_path=model_path,
                                   progress_handler=None,
                                   version_id=version_id,
                                   job_id=job_id)
    return model_out


def run_test_on_json_data(json_data, model_path, config, use_gpu=False, progress_handler=None, version_id=None,
                          job_id=None, remove_accents=True, training_with_augument=False, word2sent_path='',
                          unknown_intent='UNKNOWN', use_semhash=False):
    questions, answers, origins = load_dataset_from_json(json_data)
    samples, labels, answers, origins = load_dataset(questions, answers, origins, use_semhash, remove_accents)

    pos_accuracies, pos_precisions, pos_recalls, pos_f1_scores = [], [], [], []
    neg_accuracies, neg_precisions, neg_recalls, neg_f1_scores = [], [], [], []
    pos_aug_accuracies, pos_aug_precisions, pos_aug_recalls, pos_aug_f1_scores = [], [], [], []
    neg_aug_accuracies, neg_aug_precisions, neg_aug_recalls, neg_aug_f1_scores = [], [], [], []
    wrong_pos_samples_5fold, wrong_neg_samples_5fold, wrong_aug_pos_samples_5fold, wrong_aug_neg_samples_5fold = set(), set(), set(), set()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2407)

    for i, (train_index, test_index) in enumerate(skf.split(samples, labels), start=1):
        train_samples, test_samples = samples[train_index], samples[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        idm = model.IntentDetector(
            use_semhash=use_semhash,
            hidden_size=int(config['hidden_size']),
            batch_size=int(config['batch_size']),
            dropout=float(config['dropout']),
            use_gpu=use_gpu,
            lr=float(config['lr']),
            num_epochs=int(config['num_epochs'])
        )

        model_out = do_train(idm, train_samples, train_labels, answers,
                             origins, model_path, progress_handler, version_id, job_id)

        assert set(train_labels) == set(test_labels)
        # Test no augument
        pos_accuracy, pos_p, pos_r, pos_f1, pos_report, wrong_pos_samples = do_test(model_out, test_samples, test_labels)
        wrong_pos_samples_5fold.update(wrong_pos_samples)
        pos_accuracies.append(pos_accuracy)
        pos_precisions.append(pos_p)
        pos_recalls.append(pos_r)
        pos_f1_scores.append(pos_f1)
        print('Testing fold no augument with positive sample: {}'.format(i))
        print('--- Fold %d: Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (i, pos_accuracy, pos_p, pos_r, pos_f1))
        print(wrong_pos_samples)
        print('----------------------------------------------------------------------')

        if training_with_augument:
            train_samples, train_labels, test_aug_samples, test_aug_labels \
                = test_with_argument(word2sent_path=word2sent_path,
                                     train_samples=train_samples,
                                     train_labels=train_labels,
                                     unknown_intent=unknown_intent,
                                     remove_accents=remove_accents,
                                     model_path=model_path)

            neg_accuracy, neg_p, neg_r, neg_f1, neg_report, wrong_neg_samples = do_test(model_out, test_aug_samples, test_aug_labels)
            neg_accuracies.append(neg_accuracy)
            neg_precisions.append(neg_p)
            neg_recalls.append(neg_r)
            neg_f1_scores.append(neg_f1)
            wrong_neg_samples_5fold.update(wrong_neg_samples)
            print('Testing fold no augument with negative sample: {}'.format(i))
            print('--- Fold %d: Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (i, neg_accuracy, neg_p, neg_r, neg_f1))
            print('----------------------------------------------------------------------')

            model_out = do_train(idm, train_samples, train_labels, answers, origins, model_path, progress_handler, version_id, job_id)

            pos_aug_accuracy, pos_aug_p, pos_aug_r, pos_aug_f1, pos_aug_report, wrong_aug_pos_samples = do_test(model_out, test_samples, test_labels)
            pos_aug_accuracies.append(pos_aug_accuracy)
            pos_aug_precisions.append(pos_aug_p)
            pos_aug_recalls.append(pos_aug_r)
            pos_aug_f1_scores.append(pos_aug_f1)
            wrong_aug_pos_samples_5fold.update(wrong_aug_pos_samples)
            print('Testing fold with augument with positive sample: {}'.format(i))
            print('--- Fold %d: Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (i, pos_aug_accuracy, pos_aug_p, pos_aug_r, pos_aug_f1))
            print('----------------------------------------------------------------------')

            neg_aug_accuracy, neg_aug_p, neg_aug_r, neg_aug_f1, neg_aug_report, wrong_aug_neg_samples = do_test(model_out, test_aug_samples, test_aug_labels)
            neg_aug_accuracies.append(neg_aug_accuracy)
            neg_aug_precisions.append(neg_aug_p)
            neg_aug_recalls.append(neg_aug_r)
            neg_aug_f1_scores.append(neg_aug_f1)
            wrong_aug_neg_samples_5fold.update(wrong_aug_neg_samples)
            print('Testing fold with augument with negative sample: {}'.format(i))
            print('--- Fold %d: Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (i, neg_aug_accuracy, neg_aug_p, neg_aug_r, neg_aug_f1))
            print('**********************************************************************')

    pos_acc = np.mean(pos_accuracies)
    pos_p = np.mean(pos_precisions)
    pos_r = np.mean(pos_recalls)
    pos_f1 = np.mean(pos_f1_scores)
    print('------- Final results no augument with positive sample: %0.3f, %0.3f, %0.3f, %0.3f' % (pos_acc, pos_p, pos_r, pos_f1))

    if training_with_augument:
        neg_acc = np.mean(neg_accuracies)
        neg_p = np.mean(neg_precisions)
        neg_r = np.mean(neg_recalls)
        neg_f1 = np.mean(neg_f1_scores)

        pos_aug_acc = np.mean(pos_aug_accuracies)
        pos_aug_p = np.mean(pos_aug_precisions)
        pos_aug_r = np.mean(pos_aug_recalls)
        pos_aug_f1 = np.mean(pos_aug_f1_scores)

        neg_aug_acc = np.mean(neg_aug_accuracies)
        neg_aug_p = np.mean(neg_aug_precisions)
        neg_aug_r = np.mean(neg_aug_recalls)
        neg_aug_f1 = np.mean(neg_aug_f1_scores)

        print('------- Final results no augument with negative sample: %0.3f, %0.3f, %0.3f, %0.3f' % (neg_acc, neg_p, neg_r, neg_f1))
        print('------- Final results with augument with positive sample: %0.3f, %0.3f, %0.3f, %0.3f' % (pos_aug_acc, pos_aug_p, pos_aug_r, pos_aug_f1))
        print('------- Final results with augument with negative sample: %0.3f, %0.3f, %0.3f, %0.3f' % (neg_aug_acc, neg_aug_p, neg_aug_r, neg_aug_f1))

    if training_with_augument:
        return (pos_acc, pos_p, pos_r, pos_f1, pos_report, wrong_pos_samples_5fold), \
               (neg_acc, neg_p, neg_r, neg_f1, neg_report, wrong_neg_samples_5fold), \
               (pos_aug_acc, pos_aug_p, pos_aug_r, pos_aug_f1, pos_aug_report, wrong_aug_pos_samples_5fold), \
               (neg_aug_acc, neg_aug_p, neg_aug_r, neg_aug_f1, neg_aug_report, wrong_aug_neg_samples_5fold)
    else:
        return (pos_acc, pos_p, pos_r, pos_f1, pos_report, wrong_pos_samples_5fold), None, None, None


def do_train(idm:model.IntentDetector, train_samples, train_labels, answers, origins, model_path, progress_handler=None, version_id=None, job_id=None):
    if len(train_samples) == 0:
        idm.save_model(model_path, {})
        return
    idm.run_train(train_samples, train_labels, answers, origins,
                  model_path, progress_handler=progress_handler, version_id=version_id, job_id=job_id)
    idm.save_model(model_path)

    return idm


def do_test(model, samples, labels):
    preds, golds = [], []
    wrong_samples = []
    for sample, label in zip(samples, labels):
        pred, results = model.prediction(sample)
        preds.append(pred)
        golds.append(model.label2index[label])
        if pred != model.label2index[label]:
            wrong_samples.append((sample, model.index2label[pred], label))

    accuracy = metrics.accuracy_score(golds, preds)
    p, r, f1, s = metrics.precision_recall_fscore_support(
        golds, preds, average='micro', warn_for=[])

    print('Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (accuracy, p, r, f1))

    report = metrics.classification_report(golds, preds, labels=list(set(preds)), target_names=list(model.index2label.values()))

    return accuracy, p, r, f1, report, wrong_samples


def extract_ambiguous_samples(intent_check, samples_check, samples2label, limit=10, thresh=80):
    ambiguous_samples = []
    for sample in tqdm(samples_check, total=len(samples_check), desc='loop samples_check...'):
        for similarity_sample, score in process.extract(sample, list(samples2label.keys()), limit=limit, scorer=fuzz.ratio):
            if samples2label[similarity_sample] != intent_check and score >= thresh:
                ambiguous_samples.append((sample, intent_check, similarity_sample, samples2label[similarity_sample]))

    return ambiguous_samples


def run_test(dataset_file, model):
    samples, labels, answers, origins = load_dataset(dataset_file)

    preds, golds = [], []
    for sample, label in zip(samples, labels):
        pred, results = model.prediction(sample)
        preds.append(pred)
        golds.append(model.label2index[label])

    accuracy = metrics.accuracy_score(golds, preds)
    p, r, f1, s = metrics.precision_recall_fscore_support(
        golds, preds, average='weighted', warn_for=[])

    print('Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (accuracy, p, r, f1))

    return accuracy, p, r, f1

def infer_batch(model: model.IntentDetector, samples):
    return model.predict_batch(samples)

def run_test_v2(samples, labels, model, unknown_intent='UNKNOWN'):

    preds, golds = [], []
    out_of_intent = set()
    count = {}
    for sample, label in tzip(samples, labels):
        if label != 'unknown':
            try:
                sample = sample.lower()
            except:
                out_of_intent.add(label)
                continue
            pred, results = model.prediction(sample)
            # print(f'n_pred: {len(preds)}')
            # print(f'n_gold: {len(golds)}')
            # assert len(preds) == len(golds)

            try:
                golds.append(model.label2index[label])
                preds.append(pred)
                # if results[0][1] > 0.7:
                #     preds.append(pred)
                #     pred_label = model.index2label[pred]
                #     if pred_label in count:
                #         count[pred_label] += 1
                #     else:
                #         count[pred_label] = 1
                # else:
                #     preds.append(model.label2index[unknown_intent])
                #     if unknown_intent in count:
                #         count[unknown_intent] += 1
                #     else:
                #         count[unknown_intent] = 1
            except Exception:
                if len(preds) > len(golds):
                    preds = preds[:-1]
                out_of_intent.add(label)
                traceback.print_exc()
    print(f'out_of_intent: {out_of_intent}')
    print(f'pred: {len(preds)}')
    print(f'golds: {len(golds)}')
    print(f'count: {count}')

    accuracy = metrics.accuracy_score(golds, preds)
    p, r, f1, s = metrics.precision_recall_fscore_support(golds, preds, average='weighted', warn_for=[])

    print('Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (accuracy, p, r, f1))
    # del model.index2label[model.label2index['UNKNOWN']]
    label_ids = [id for id in list(model.index2label.keys())]
    label_names = [label for label in list(model.index2label.values())]
    report = metrics.classification_report(golds, preds, labels=label_ids, target_names=label_names, zero_division=1)
    # report = metrics.classification_report(golds, preds, zero_division=1, labels=[35])
    print(report)
    # matrix = confusion_matrix(golds, preds)
    # plot_confusion_matrix(matrix, list(set(model.label2index.keys())))
    with open('report.txt', 'w') as fw:
        fw.write(report)

    return accuracy, p, r, f1


def infer(model: model.IntentDetector, sample):
    pred, results = model.prediction(sample, topk=2)
    return model.index2label[pred], results[0][1]


def load_dataset(questions, answers, origins, use_semhash, remove_accents=True):
    samples, labels = [], []

    for intent, list_of_samples in tqdm(questions.items(), total=len(questions), desc='looping data'):
        for sample in list_of_samples:
            tokenized_sample = utils.text_tokenizer(sample)
            hashed_sample = utils.semhash_tokenizer(tokenized_sample, use_semhash)
            samples.append(hashed_sample)
            labels.append(intent)

            if remove_accents:
                # remove accents
                unaccented_sample = utils.rm_accents(tokenized_sample)
                if unaccented_sample != tokenized_sample:
                    unaccented_hash_sample = utils.semhash_tokenizer(unaccented_sample, use_semhash)
                    samples.append(unaccented_hash_sample)
                    labels.append(intent)
    samples = np.asarray(samples)
    labels = np.asarray(labels)

    return samples, labels, answers, origins


def load_dataset_from_json(data):
    questions, answers, origins = data.get('questions', []), data.get(
        'answers', []), data.get("origins", [])

    return questions, answers, origins


def load_dataset_from_file(dataset_file, remove_accents=True):
    questions, answers, origins = utils.load_json_data(dataset_file)
    return load_dataset(questions, answers, origins, remove_accents)


def load_model(model_path, config, use_gpu, use_semhash):
    idm = model.IntentDetector(use_semhash, hidden_size=int(config['hidden_size']), batch_size=int(config['batch_size']),
                               dropout=float(config['dropout']), use_gpu=use_gpu, lr=float(config['lr']),
                               num_epochs=int(config['num_epochs']))

    model_loaded = idm.load_model(model_path)
    if model_loaded is None:
        return None
    return idm

