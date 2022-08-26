import time
import pickle
# import endpoint
import traceback
import itertools
import pandas as pd
from tqdm import tqdm
from pyvi import ViTokenizer
from core import model_wrapper
from collections import OrderedDict
from core.utils.utils import join_semhash


import configparser
MODEL_CONFIG = 'config/trainer/model.cfg'
USE_GPU = True
model_config = configparser.ConfigParser()
model_config.read(MODEL_CONFIG)


def build_sample_by_intent(samples, intents, use_semhash):
    data = dict()
    for sample, intent in zip(samples, intents):
        if intent in data:
            if use_semhash:
                data[intent].append(join_semhash(sample))
            else:
                data[intent].append(sample)
        else:
            if use_semhash:
                data[intent] = [join_semhash(sample)]
            else:
                data[intent] = [sample]

    return data


def statistic_len(length, len_dict):
    if length <= 5:
        len_dict[5] += 1
    elif 5 < length <= 10:
        len_dict[10] += 1
    elif 10 < length <= 20:
        len_dict[20] += 1
    else:
        len_dict[50] += 1

    return len_dict


def count_word(samples, tokenize=True, lower=True):
    word2count, len_dict = {}, OrderedDict({5: 0, 10: 0, 20: 0, 50: 0})
    for sample in tqdm(samples, total=len(samples), desc='count_word_frequence'):
        if tokenize:
            sample = ViTokenizer.tokenize(sample)
        sample = sample.lower() if lower else sample
        toks = sample.split()
        len_dict = statistic_len(len(toks), len_dict)
        for tok in toks:
            tok = tok.replace('_', ' ')
            if tok in word2count:
                word2count[tok] += 1
            else:
                word2count[tok] = 1

    return word2count, len_dict


def get_top_negative_word(word2count, negative_word_ratio):
    word2count_order = OrderedDict()
    for word, count in sorted(word2count.items(), key=lambda item: item[1], reverse=True):
        word2count_order[word] = count

    filter_word_size = int(negative_word_ratio * len(word2count_order))  # Lấy top các từ xuất hiện nhiều nhất
    word2count_order = dict(itertools.islice(word2count_order.items(), 0, filter_word_size))

    return word2count_order


def filter_sent_by_len(sents, all_aug_sents, len_dict):
    count_5, count_10, count_20, count_50 = 0, 0, 0, 0
    filter_sents = []
    for sent in sents:
        length = len(sent.split())
        if (length <= 5 and count_5 < len_dict[5]) \
                or (5 < length <= 10 and count_10 < len_dict[10]) \
                or (10 < length <= 20 and count_20 < len_dict[20]) \
                or (length > 20 and count_50 < len_dict[50]):
            if sent not in all_aug_sents:  # check duplicate
                filter_sents.append(sent)
                if length <= 5: count_5 += 1
                if 5 < length <= 10: count_10 += 1
                if 10 < length <= 20: count_20 += 1
                if length > 20: count_50 += 1

    return filter_sents


def get_sent_by_negative_word(intent, word2count, filter_sent_by_intent, all_aug_sents, word2sent, len_dict,
                              use_length_distribute=True):
    for word, freq in word2count.items():
        try:
            sents = word2sent[word]
            # if use_length_distribute:
            #     sents = filter_sent_by_len(sents, all_aug_sents, len_dict)

            if len(sents) < freq:
                print(f'Found {len(sents)}/{freq} of word \'{word}\'')
            else:
                print(f'Found {len(sents)}/{freq} of word \'{word}\'')
                sents = list(sents)[:freq]

            all_aug_sents.extend(sents)
            if intent in filter_sent_by_intent:
                filter_sent_by_intent[intent + '@' + word].extend(sents)
            else:
                filter_sent_by_intent[intent + '@' + word] = sents

        except:
            print(f'Could not find sentence for word \'{word}\'')
            traceback.print_exc()

    return filter_sent_by_intent, all_aug_sents


def augument_negative_sample(questions, word2sent, tokenize=True, negative_word_ratio=0.05,
                             lower=True, use_length_distribute=True, unknown_intent='UNKNOWN'):

    filter_sent_by_intent = {}
    all_aug_sents = []
    for intent, list_of_samples in questions.items():
        if intent != unknown_intent:
            word2count, len_dict = count_word(list_of_samples, tokenize, lower)
            word2count = get_top_negative_word(word2count, negative_word_ratio)
            filter_sent_by_intent, all_aug_sents = \
                get_sent_by_negative_word(intent, word2count, filter_sent_by_intent,
                                          all_aug_sents, word2sent, len_dict,
                                          use_length_distribute=use_length_distribute)

    return all_aug_sents


def do_augument(corpus_db, questions, tokenize=True, negative_word_ratio=0.05, lower=True,
                use_length_distribute=True, unknown_intent='UNKNOWN'):
    start = time.time()
    word2sent = pickle.load(open(corpus_db, 'rb'))
    print(f'Finish loading corpus in : {time.time() - start}')

    start = time.time()
    augumented_samples = augument_negative_sample(questions, word2sent, tokenize=tokenize,
                                                  negative_word_ratio=negative_word_ratio,
                                                  lower=lower, use_length_distribute=use_length_distribute,
                                                  unknown_intent=unknown_intent)
    print(f'Time for augument: {time.time() - start}')

    return augumented_samples


def do_filter(augumented_samples, idm, unknown_intent='UNKNOWN_INTENT', filter_thresh=0.5):
    filtered_samples = set()

    for sample in tqdm(augumented_samples, total=len(augumented_samples), desc='Filter aug...'):
        pred, results = idm.prediction(sample)
        label, prob = idm.index2label[pred], results[0][1]

        if label != unknown_intent and prob < filter_thresh:
            filtered_samples.add(sample)

    return list(filtered_samples)


def train_with_argument(word2sent_path, train_samples, train_labels, answers, origins, unknown_intent='UNKNOWN', filter_thresh=0.5,
                        use_semhash=False, remove_accents=True, model_path=None, progress_handler=None, version_id=None, job_id=None):
    questions = build_sample_by_intent(train_samples, train_labels, use_semhash)
    # augument data
    augumented_samples = do_augument(word2sent_path, questions, unknown_intent=unknown_intent)

    # load intent model for filter
    # idm = model_wrapper.load_model(model_path, endpoint.model_cfg.model_config['intent'], use_gpu=endpoint.model_cfg.use_gpu)
    idm = model_wrapper.load_model(model_path, model_config['intent'], use_gpu=USE_GPU, use_semhash=use_semhash)

    # filter by model
    augumented_samples = do_filter(augumented_samples, idm, unknown_intent=unknown_intent, filter_thresh=filter_thresh)

    with open('augumented_sample.txt', 'w') as fw:
        for sample in augumented_samples:
            fw.write(sample.strip() + '\n')

    if unknown_intent in questions:
        questions[unknown_intent].extend(augumented_samples)
    else:
        questions[unknown_intent] = augumented_samples

    train_samples, train_labels, answers, origins = model_wrapper.load_dataset(questions, [], [], use_semhash, remove_accents)

    return model_wrapper.do_train(idm, train_samples, train_labels, answers, origins, model_path, progress_handler, version_id, job_id)


def test_with_argument(word2sent_path, train_samples, train_labels, unknown_intent='UNKNOWN', filter_thresh=0.5,
                       use_semhash=False, remove_accents=True, model_path=None):
    questions = build_sample_by_intent(train_samples, train_labels, use_semhash)
    # augument data
    augumented_samples = do_augument(word2sent_path, questions, unknown_intent=unknown_intent)

    # load intent model for filter
    idm = model_wrapper.load_model(model_path, model_config['intent'], use_gpu=USE_GPU, use_semhash=use_semhash)

    # filter by model
    augumented_samples = do_filter(augumented_samples, idm, unknown_intent=unknown_intent, filter_thresh=filter_thresh)

    train_aug_size = int(0.8 * len(augumented_samples))
    train_aug_samples, train_aug_labels = augumented_samples[:train_aug_size], [unknown_intent] * train_aug_size
    test_aug_samples, test_aug_labels = augumented_samples[train_aug_size:], [unknown_intent] * (len(augumented_samples) - train_aug_size)

    if unknown_intent in questions:
        questions[unknown_intent].extend(train_aug_samples)
    else:
        questions[unknown_intent] = train_aug_samples

    train_samples, train_labels, answers, origins = model_wrapper.load_dataset(questions, [], [], use_semhash, remove_accents)

    return train_samples, train_labels, test_aug_samples, test_aug_labels


if __name__ == '__main__':
    train_file = 'data/vinfast_vi/raw/train.csv'
    df = pd.read_csv(train_file, delimiter='\t')
    samples = df['sample'].values.tolist()
    intents = df['label'].values.tolist()
    questions = build_sample_by_intent(samples, intents)

    word2sent_path = 'word2sent.pkl'

    train_with_argument(word2sent_path, samples, intents, [], [], unknown_intent='UNKNOWN', filter_thresh=0.5,
                        remove_accents=True, model_path=None, progress_handler=None, version_id=None)


    # do_split_train_test()
    # do_merge_negative()

    # preprocess_test_neg_aug('data/data_vinfast_vi/train_negative_by_intent_v1_test_negative.txt')
    # preprocess_neg_aug('data/data_vinfast_vi/train_negative_by_intent_v1.txt')
