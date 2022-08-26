from numpy import fromstring, dtype
from torch.autograd import Variable
from nltk import word_tokenize
from tqdm import tqdm
from operator import itemgetter
from collections import OrderedDict
from pyvi import ViTokenizer

import numpy as np
import codecs
import random
import torch
import json
import unicodedata
import unidecode
import re


import sys
# import ipdb


def log(message):
    print('OPPO_{0}'.format(message), file=sys.stderr)


def find_ngrams(input_list, n_gram=3):
    return zip(*[input_list[i:] for i in range(n_gram)])


def semhash_tokenizer(text, use_semhash, n_gram=3):
    tokens = text.split(" ")
    if not use_semhash:
        final_text = " ".join(map(str, tokens))
    else:
        final_tokens = []
        for unhashed_token in tokens:
            hashed_token = "#{}#".format(unhashed_token)
            final_tokens += [''.join(gram)
                             for gram in list(find_ngrams(list(hashed_token), n_gram))]

        final_text = " ".join(map(str, final_tokens))

    return final_text


def join_semhash(sample):
    tokens = []
    token = ''
    for semhash in sample.split():
        if semhash.startswith('#'):
            token = semhash.replace('#', '')
        elif semhash.endswith('#'):
            tokens.append(token)
            token = ''
        else:
            token += semhash[-1]

    return ' '.join(tokens)


def remove_accents(text):
    nkfd_form = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nkfd_form if not unicodedata.combining(c)])


def rm_accents(text):
    return unidecode.unidecode(text)


def is_punct(tk):
    """Filter punctuations in a sentence
    """
    filter = frozenset([c for c in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'])
    newtk = ''.join([c for c in tk if c not in filter])
    if newtk == '':
        return True
    else:
        return False


def text_tokenizer(text):
    tokens = [tk.lower() for tk in word_tokenize(text) if not is_punct(tk)]

    if len(tokens) > 100:
        tokens = tokens[:100]

    return ' '.join(tokens)


def build_vocab(samples, labels):
    vocab = dict()
    vocab['#@UNK@#'] = 0
    count_word = {}
    for sample, label in tqdm(zip(samples, labels), total=len(samples), desc='build_vocab'):
        for hash in sample.split(" "):
            if hash not in count_word:
                count_word[hash] = 1
            else:
                count_word[hash] += 1

    sorted_count_word = OrderedDict(sorted(count_word.items(), key=itemgetter(1), reverse=True))
    sorted_count_word_key = list(sorted_count_word.keys())[:5000]
    sorted_count_word = {key:sorted_count_word[key] for key in sorted_count_word_key}
    for word, count in sorted_count_word.items():
        vocab[word] = len(vocab)

    print(f'vocab_size: {len(vocab)}')
    return vocab


# def hash2index(samples, labels):
#     data = list()
#     label2index = dict()
#     vocab = build_vocab(samples, labels)
#     for sample, label in zip(samples, labels):
#         sample_idx = []
#         for hash in sample.split(" "):
#             if hash in vocab:
#                 sample_idx.append(vocab[hash])
#             else:
#                 sample_idx.append(vocab['#@UNK@#'])
#
#         if label not in label2index:
#             label_idx = len(label2index)
#             label2index[label] = label_idx
#
#         data.append((sample_idx, label2index[label]))
#
#     return data, vocab, label2index

# with open('../../config/trainer/stopword_vn.txt', 'r') as fi:
with open('config/trainer/stopword_vn.txt', 'r') as fi:
    stop_words = fi.readlines()
    stop_words = [word.rstrip() for word in stop_words]

def clean_vietnamese_text(txt: str, stop_word=stop_words) -> str:
    txt = txt.lower()
    txt = re.sub('[^A-Za-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]+', ' ', txt)
    txt = txt.strip()
    for w in stop_word:
        while w in txt.split():
            txt = re.sub(w, '', txt)
    tokens = ViTokenizer.tokenize(txt)
    return tokens

def hash2index(samples, labels, preprocess=True):
    data = list()
    vocab = dict()
    label2index = dict()
    vocab['#@UNK@#'] = 0
    for sample, label in zip(samples, labels):
        sample_idx = []
        if preprocess:
            sample = clean_vietnamese_text(sample)
        for hash in sample.split(" "):
            if hash not in vocab:
                vocab[hash] = len(vocab)
            sample_idx.append(vocab[hash])

        if label not in label2index:
            label_idx = len(label2index)
            label2index[label] = label_idx

        data.append((sample_idx, label2index[label]))

    return data, vocab, label2index


def index2matrix(sample_indices, vocab_size=10000):
    data_size = len(sample_indices)
    mat = torch.zeros(data_size, vocab_size)

    for idx,sample_idx in enumerate(sample_indices):
        index = torch.LongTensor(sample_idx)
        mat[idx].index_add_(0, index, torch.ones(index.size()))

    mat_var = Variable(mat)

    return mat_var


def prepare_sequences(batch_data, vocab_size=10000):
    labels = []
    sample_indices = []

    for (sample, label) in batch_data:
        sample_indices.append(sample)
        labels.append(label)

    labels = Variable(torch.LongTensor(np.asarray(labels)))
    seq_tensor = index2matrix(sample_indices, vocab_size)

    return seq_tensor, labels


def preprocess_one_sentence(sentence, word2index, use_semhash, vocab_size=10000):
    sentence_idx = []
    sentence = clean_vietnamese_text(sentence)
    tokenized_sentence = text_tokenizer(sentence)
    hashed_sentence = semhash_tokenizer(tokenized_sentence, use_semhash)
    for word in hashed_sentence.strip().split(' '):
        if word in word2index:
            sentence_idx.append(word2index[word])
        else:
            sentence_idx.append(word2index['#@UNK@#'])

    sent_tensor = index2matrix([sentence_idx], vocab_size)

    return sent_tensor


def preprocess_batch(samples, word2index, use_semhash, vocab_size=10000):
    sentence_ids = []
    check = True
    for sentence in samples:
        # if sentence == sentence:
        #     sentence = sentence.replace('_', ' ')
        sentence_idx = []
        sentence = clean_vietnamese_text(sentence)
        tokenized_sentence = text_tokenizer(sentence)
        hashed_sentence = semhash_tokenizer(tokenized_sentence, use_semhash)
        for word in hashed_sentence.strip().split(' '):
            if word in word2index:
                sentence_idx.append(word2index[word])
            else:
                sentence_idx.append(word2index['#@UNK@#'])
        sentence_ids.append(sentence_idx)

    sent_tensor = index2matrix(sentence_ids, vocab_size)

    return sent_tensor


def getBatch(data, batch_size):
    random.shuffle(data)
    start_idx = 0
    batches = []
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        batch = data[start_idx:start_idx + batch_size]
        start_idx = start_idx + batch_size
        batches.append(batch)

    if start_idx < len(data):
        batch = data[start_idx:len(data)]
        batches.append(batch)

    return batches


def load_json_data(filepath):
    # samples = {}
    # labels = {}
    try:
        with open(filepath, 'r') as f:
            json_str = f.read()
        data = json.loads(json_str)
        questions, answers, origins = data['questions'], data['answers'], data["origins"]

        return questions, answers, origins

    except Exception as e:
        print(str(e))


def prepare_sequences_eval(sentences, labels, word2index, label2index, max_seq_len=None, min_seq_len=5):
    sentences_idx = []
    golds_idx = []

    if max_seq_len is None:
        max_seq_len = max([len(sent) for sent in sentences])

    if max_seq_len < min_seq_len: max_seq_len = min_seq_len

    for sentence, label in zip(sentences, labels):
        sentence = text_tokenizer(sentence)
        sentence_idx = []
        for word in sentence.strip().split(' '):
            if word in word2index:
                sentence_idx.append(word2index[word])
            else:
                sentence_idx.append(word2index['#UNK#'])
        if len(sentence_idx) < max_seq_len:
            while len(sentence_idx) < max_seq_len:
                sentence_idx.append(word2index['#PAD#'])

        sentences_idx.append(sentence_idx)
        golds_idx.append(label2index[label])

    return sentences_idx, golds_idx


def smart_open(fname, mode='rb'):
    if fname.endswith('.gz'):
        import gzip
        return gzip.open(fname, mode)
    elif fname.endswith('.bz2'):
        import bz2
        return bz2.BZ2File(fname, mode)
    elif fname.endswith('.bin'):
        return open(fname, mode)
    else:
        return codecs.open(fname, mode, 'utf-8')


# def load_pretrain_embeddings(filepath, binary=1):
#     vecs = []
#     vocab = []
#     if binary == 1:
#         with smart_open(filepath, 'rb') as f:
#             # header = f.readline().encode('utf-8')
#             header = f.readline()
#             vocab_size, vector_size = map(int, header.split())
#             binary_len = dtype(np.float32).itemsize * vector_size
#             for _ in range(vocab_size):
#                 word = []
#                 while True:
#                     ch = f.read(1)
#                     if ch == b' ':
#                         break
#                     if ch != b'\n':
#                         word.append(ch)
#                 word = to_unicode(b''.join(word))
#                 # word = b''.join(word)
#                 vocab.append(word)
#                 vec = fromstring(f.read(binary_len), dtype=np.float32)
#                 vecs.append(vec)
#     else:
#         with smart_open(filepath, 'rb') as f:
#             header = f.readline()
#             if len(header.split()) == 2:
#                 vocab_size, vector_size = map(int, header.split())
#             elif len(header.split()) > 2:
#                 parts = header.rstrip().split(" ")
#                 word, vec = parts[0], list(map(np.float32, parts[1:]))
#                 vocab.append(word)
#                 vecs.append(vec)
#             for _, line in enumerate(f):
#                 parts = line.rstrip().split(" ")
#                 word, vec = parts[0], list(map(np.float32, parts[1:]))
#                 vocab.append(word)
#                 vecs.append(vec)
#
#     embs_dim = len(vecs[1])
#     UNKNOWN_VEC = np.random.uniform(-0.25, 0.25, embs_dim)
#     PAD_VEC = np.random.uniform(-0.25, 0.25, embs_dim)
#     vecs = np.vstack((UNKNOWN_VEC, vecs))
#     vecs = np.vstack((PAD_VEC, vecs))
#     vocab = ['#PAD#'] + ['#UNK#'] + vocab
#
#     # word2index = {word:idx for idx,word in enumerate(vocab)}
#
#     return vocab, vecs


def load_pretrain_embeddings(filepath, vocab=None, binary=1):
    vecs = []
    use_vocab = True
    if vocab is None:
        use_vocab = False
        vocab = []
    if binary == 1:
        with smart_open(filepath, 'rb') as f:
            # header = f.readline().encode('utf-8')
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            binary_len = dtype(np.float32).itemsize * vector_size
            for _ in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        word.append(ch)
                # word = to_unicode(b''.join(word))
                word = b''.join(word)
                vec = fromstring(f.read(binary_len), dtype=np.float32)
                if use_vocab and word in vocab:
                    vecs.append(vec)
                elif not use_vocab:
                    vocab.append(word)
                    vecs.append(vec)
    else:
        with smart_open(filepath, 'rb') as f:
            header = f.readline()
            if len(header.split()) == 2:
                vocab_size, vector_size = map(int, header.split())
            elif len(header.split()) > 2:
                parts = header.rstrip().split(" ")
                word, vec = parts[0], list(map(np.float32, parts[1:]))
                if use_vocab and word in vocab:
                    vecs.append(vec)
                elif not use_vocab:
                    vocab.append(word)
                    vecs.append(vec)
            count_word = 1
            for _, line in enumerate(f):
                count_word += 1
                print('\r count word = %d' % count_word, end='\r')
                parts = line.rstrip().split(" ")
                word, vec = parts[0], list(map(np.float32, parts[1:]))
                if use_vocab and word in vocab:
                    vecs.append(vec)
                elif not use_vocab:
                    vocab.append(word)
                    vecs.append(vec)

    embs_dim = len(vecs[0])
    UNKNOWN_VEC = np.random.uniform(-0.25, 0.25, embs_dim)
    PAD_VEC = np.random.uniform(-0.25, 0.25, embs_dim)

    if len(vecs) != len(vocab):
        for _ in range(len(vocab) - len(vecs)):
            vecs = np.vstack((UNKNOWN_VEC, vecs))

    vecs = np.vstack((UNKNOWN_VEC, vecs))
    vecs = np.vstack((PAD_VEC, vecs))
    vocab = ['#PAD#'] + ['#UNK#'] + vocab

    # word2index = {word:idx for idx,word in enumerate(vocab)}

    return vocab, vecs


def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    return str(text, encoding=encoding, errors=errors)


if __name__ == '__main__':
    # embedding_path = '/home/anhnk/oppo-backend/data/fptai_unseg_50d.word2vec.bin'

    # vocab, vecs = load_pretrain_embeddings(embedding_path)
    # ipdb.set_trace()

    samples = ["this is the first doc, and the first paragraph",
               "this is the second doc, and the second paragraph"]
    labels = ["book", "book"]

    data, vocab, label2index = hash2index(samples, labels)

    print(data)
    print(vocab)
    print(label2index)

    data_size = len(data)
    vocab_size = len(vocab)

    mat = torch.zeros(data_size, vocab_size)
    for idx, (inds, label) in enumerate(data):
        index = torch.LongTensor(inds)
        mat[idx].index_add_(0, index, torch.ones(index.size()))

    print(mat)