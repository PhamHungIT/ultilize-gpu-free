from optparse import Values
import os
import configparser
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from core import model_wrapper
from utils import write_csv, write_csv_predicted, write_csv_tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from argparse import ArgumentParser

MODEL_CONFIG = 'config/trainer/model.cfg'
USE_GPU = True
model_config = configparser.ConfigParser()
model_config.read(MODEL_CONFIG)

remove_accents = False
word2sent_path = 'db/word2sent.pkl'
# word2sent_path='db/word2sent_semhash.pkl'


def build_data(samples, intents):
    intent_data = dict()
    for sample, intent in zip(samples, intents):
        if str(sample) != 'nan':
            if intent not in intent_data:
                intent_data[intent] = [sample]
            else:
                intent_data[intent].append(sample)

    answers = "this is answer"
    origins = "this is origin question"
    data = {'questions': intent_data, 'answers': answers, 'origins': origins}

    return data


def prepare_data(csv_file):
    df = pd.read_csv(csv_file, delimiter='\t')
    samples = df['sample'].values.tolist()
    intents = df['label'].values.tolist()

    data = build_data(samples, intents)

    return data


def filter_sample(samples, labels):

    intent2label = {}
    count = 0
    for sample, label in tqdm(zip(samples, labels), total=len(samples), desc='filter sample'):
        count += 1
        if label in intent2label:
            intent2label[label].add(str(sample).lower())
        else:
            intent2label[label] = {str(sample).lower()}

    filter_sample, filter_label = [], []
    for intent, samples in intent2label.items():
        if len(samples) > 2:
            filter_sample.extend(list(samples))
            filter_label.extend([intent] * len(samples))

    return filter_sample, filter_label


def split_data(csv_file):
    df = pd.read_csv(csv_file, delimiter='\t', error_bad_lines=False)
    samples = df['sample'].values.tolist()
    intents = df['label'].values.tolist()

    samples, intents = filter_sample(samples, intents)

    X_train, X_test, y_train, y_test = train_test_split(samples, intents, stratify=intents, test_size=0.5)

    train_data = build_data(X_train, y_train)
    test_data = build_data(X_test, y_test)

    return train_data, test_data, X_train, X_test, y_train, y_test


def do_split_file(folder, csv_file, train_file, test_file):
    train_data, test_data, X_train, X_test, y_train, y_test = split_data(os.path.join(folder, csv_file))

    write_csv(X_train, y_train, os.path.join(folder, train_file))
    write_csv(X_test, y_test, os.path.join(folder, test_file))


def do_train_test_5fold(folder, train_file, model_name, unknown_intent, use_semhash):
    model_path = os.path.join(folder, model_name)

    train_data = prepare_data(os.path.join(folder, train_file))
    return model_wrapper.run_test_on_json_data(train_data, model_path, model_config['intent'],
                                               use_gpu=USE_GPU,
                                               use_semhash=use_semhash,
                                               remove_accents=remove_accents,
                                               training_with_augument=training_with_augument,
                                               word2sent_path=word2sent_path,
                                               unknown_intent=unknown_intent)


def do_train(folder, train_file, model_name, unknown_intent, use_semhash):
#     model_path = os.path.join(folder, model_name)
    model_path = "/content/drive/MyDrive/Colab Notebooks/category_classify/shopee/model/" + model_name

    train_data = prepare_data(os.path.join(folder, train_file))
    model_wrapper.run_train_on_json_data(train_data, model_path, model_config['intent'],
                                         use_gpu=USE_GPU,
                                         use_semhash=use_semhash,
                                         remove_accents=remove_accents,
                                         training_with_augument=training_with_augument,
                                         word2sent_path=word2sent_path,
                                         unknown_intent=unknown_intent)


def do_test(folder, test_file, model_name, unknown_intent, use_semhash):
    model_path = os.path.join(folder, model_name)

    model = model_wrapper.load_model(model_path, model_config['intent'], USE_GPU, use_semhash)

    df = pd.read_csv(os.path.join(folder, test_file), delimiter='\t')
    samples = df['sample'].values.tolist()
    labels = df['label'].values.tolist()
    model_wrapper.run_test_v2(samples, labels, model, unknown_intent)

def do_test_batch(folder, test_file, model_name, unknown_intent, use_semhash):
    model_path = os.path.join(folder, model_name)

    model = model_wrapper.load_model(model_path, model_config['intent'], USE_GPU, use_semhash)

    df = pd.read_csv(os.path.join(folder, test_file), delimiter='\t')
    df.dropna(inplace=True)
    samples = df['sample'].values.tolist()
    labels = df['label'].values.tolist()
    result = model_wrapper.infer_batch(model, samples)
    predict_label = [list(pre.keys())[0] for pre in result]
    print(classification_report(labels, predict_label))

def do_infer_batch(folder, test_file, model_name, unknown_intent, use_semhash):
    model_path = os.path.join(folder, model_name)

    model = model_wrapper.load_model(model_path, model_config['intent'], USE_GPU, use_semhash)

    df = pd.read_csv(os.path.join(folder, test_file), delimiter='\t')
    samples = df['sample'].values.tolist()
    labels = df['label'].values.tolist()
    result = model_wrapper.infer_batch(model, samples)
    predict_label = [list(pre.keys())[0] for pre in result]
    predict_probs = [list(pre.values())[0] for pre in result]
    df.insert(loc=len(df.columns), column="predicted_label", value=predict_label)
    df.insert(loc=len(df.columns), column="predicted_probs", value=predict_probs)
    out_file = os.path.join(folder, test_file.replace('.csv', '_predicted.csv'))
    df.to_csv(out_file, sep='\t', index=False)

def do_test_domain_flow(folder, test_file, domain_model_name, indomain_model_name, domain, unknown_intent):

    domain_model = model_wrapper.load_model(os.path.join(folder, domain_model_name), model_config['intent'], USE_GPU)
    indomain_model = model_wrapper.load_model(os.path.join(folder, indomain_model_name), model_config['intent'], USE_GPU)

    df = pd.read_csv(os.path.join(folder, test_file), delimiter='\t')
    samples = df['sample'].values.tolist()
    labels = df['label'].values.tolist()

    preds, golds = [], []
    out_of_intent = set()
    count = {}
    for sample, label in zip(samples, labels):
        sample = sample.lower()

        predicted_label, prob = model_wrapper.infer(domain_model, sample)
        print(f'{predicted_label}: {sample}')

        if predicted_label == domain:
            pred, _ = indomain_model.prediction(sample)
            preds.append(pred)
            golds.append(indomain_model.label2index[label])
        else:
            preds.append(indomain_model.label2index[unknown_intent])

    print(f'out_of_intent: {out_of_intent}')
    print(f'pred: {len(preds)}')
    print(f'golds: {len(golds)}')
    print(f'count: {count}')

    accuracy = metrics.accuracy_score(golds, preds)
    p, r, f1, s = metrics.precision_recall_fscore_support(golds, preds, average='weighted', warn_for=[])

    print('Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' % (accuracy, p, r, f1))
    report = metrics.classification_report(golds, preds, labels=list(set(preds)), zero_division=1)
    # report = metrics.classification_report(golds, preds, labels=list(set(preds)), target_names=list(model.index2label.values()), zero_division=1)
    # report = metrics.classification_report(golds, preds, zero_division=1, labels=[35])
    print(report)
    # matrix = confusion_matrix(golds, preds)

    return accuracy, p, r, f1


def do_infer(folder, model_name):
    model_path = os.path.join(folder, model_name)

    model = model_wrapper.load_model(model_path, model_config['intent'], USE_GPU, use_semhash)
    print(f'n_vocab: {model.vocab_size}')
    print(f'n_label: {model.label_size}')
    print(f'n_params: {model.count_parameters()}')
    while True:
        sample = input('Sample: ')
        predicted_label, prob = model_wrapper.infer(model, sample)
        print(f'{predicted_label}: {prob}')


def do_infer_from_file(folder, csv_file, model_name):
    model_path = os.path.join(folder, model_name)
    model = model_wrapper.load_model(model_path, model_config['intent'], USE_GPU, use_semhash)

    df = pd.read_csv(os.path.join(folder, csv_file), delimiter='\t')
    samples = df['sample'].values.tolist()
    labels = df['label'].values.tolist()

    predicted_labels, predicted_probs = [], []

    for sample, label in tqdm(zip(samples, labels), total=len(samples)):
        predicted_label, prob = model_wrapper.infer(model, sample)
        predicted_labels.append(predicted_label)
        predicted_probs.append(prob)

    predicted_file = csv_file.replace('.csv', '_predicted.csv')
    write_csv_predicted(samples, labels, predicted_labels, predicted_probs, os.path.join(folder, predicted_file))


def do_infer_domain_flow(domain_model_name, incar_model_name, phonecall_model_name, media_model_name, navi_model_name, general_model_name):
    domain_model = model_wrapper.load_model(os.path.join(folder, domain_model_name), model_config['intent'], USE_GPU)
    incar_model = model_wrapper.load_model(os.path.join(folder, idata/all_category/data_run/_1M_shopee/test_set/additional/test_concat.csvr_model_name), model_config['intent'], USE_GPU)
    phonecall_model = model_wrapper.load_model(os.path.join(folder, phonecall_model_name), model_config['intent'], USE_GPU)
    media_model = model_wrapper.load_model(os.path.join(folder, media_model_name), model_config['intent'], USE_GPU)
    navi_model = model_wrapper.load_model(os.path.join(folder, navi_model_name), model_config['intent'], USE_GPU)
    general_model = model_wrapper.load_model(os.path.join(folder, general_model_name), model_config['intent'], USE_GPU)

    while True:
        sample = input('Sample: ')
        predicted_label, prob = model_wrapper.infer(domain_model, sample)
        print(f'{predicted_label}: {prob}')
        if predicted_label == 'INCAR':
            intent_predicted_label, prob = model_wrapper.infer(incar_model, sample)
        elif predicted_label == 'PHONECALL':
            intent_predicted_label, prob = model_wrapper.infer(phonecall_model, sample)
        elif predicted_label == 'MEDIA':
            intent_predicted_label, prob = model_wrapper.infer(media_model, sample)
        elif predicted_label == 'NAVI':
            intent_predicted_label, prob = model_wrapper.infer(navi_model, sample)
        else:
            intent_predicted_label, prob = model_wrapper.infer(general_model, sample)

        print(f'{intent_predicted_label}: {prob}')


def do_experiment(folder, csv_file, model_name, unknown_intent, use_semhash, mode):
    train_file = csv_file.replace('.csv', '_train.csv')
    test_file = csv_file.replace('.csv', '_test.csv')
    # train_file = 'train_merge_random_processed.csv'
    # test_file = 'test_merge_random_processed.csv'
    # train_file = csv_file.replace('.csv', '_100UNK_train.csv')
    # test_file = csv_file.replace('.csv', '_100UNK_test.csv')
    # train_file = 'Lux_intent_2021_07_27_processed_merged_Lux_qna_2021_07_27_processed_domain_train_with_kbqa_negative.csv'

    if mode == 'revised':
        # do_train_test_5fold(folder, train_file, model_name, unknown_intent, use_semhash)
        do_revise_samples(folder, csv_file, model_name)
    if mode == 'split':
        do_split_file(folder, csv_file, train_file, test_file)

    if mode == 'train':
        do_train(folder, csv_file, model_name, unknown_intent, use_semhash)

    if mode == 'test':
        # test_file = 'test-vf-chatbot-intent-revised.csv'
        # test_file = 'test-vf-chatbot-intent-02.csv'
        # test_file = 'test.csv'
        # test_file = 'test_merge_random_processed.csv'
        do_test(folder, csv_file, model_name, unknown_intent, use_semhash)
    if mode == 'test-batch':
        do_test_batch(folder, csv_file, model_name, unknown_intent, use_semhash)
    if mode == 'infer-batch':
        do_infer_batch(folder, csv_file, model_name, unknown_intent, use_semhash)

    if mode == 'train-test':
        # train_file = 'vinfast_chatbot_intent.csv'
        # test_file = 'test-vf-chatbot-intent-revised.csv'
        # test_file = 'test-vf-chatbot-intent-02.csv'
        # train_file = 'ict_train.csv'
        # test_file = 'ict_test.csv'
        do_train(folder, train_file, model_name, unknown_intent, use_semhash)
        do_test(folder, test_file, model_name, unknown_intent, use_semhash)

    if mode == 'infer':
        do_infer(folder, model_name)

    if mode == 'infer_file':
        # test_file = 'test-vf-chatbot-intent-revised.csv'
        # test_file = 'all_shopee.csv'
        do_infer_from_file(folder, csv_file, model_name)

    # # do_test_domain_flow(folder, test_file, 'dwhile re.findall(r' [A-z] |^[A-z] | [A-z]$', txt):
    #     txt = re.sub(r' [A-z] |^[A-z] | [A-z]$', ' ', txt)omain.pt', 'incar.pt', 'INCAR', unknown_intent)
    # # do_infer_domain_flow('domain.pt', 'incar.pt', 'phonecall.pt', 'media.pt', 'navi.pt', 'general.pt')


def resaved_model(model_path):
    model = model_wrapper.load_model(model_path, model_config['intent'], USE_GPU, use_semhash)
    new_model_path = model_path.replace('.pt', '_saved.pt')
    model.save_model(new_model_path)


def do_revise_samples(folder, train_file, model_name):
    pos, neg, pos_aug, neg_aug = do_train_test_5fold(folder, train_file, model_name, unknown_intent, use_semhash)
    if training_with_augument:
        wrong_pos_samples_5fold = pos[-1]
        wrong_neg_samples_5fold = neg[-1]
        wrong_aug_pos_samples_5fold = pos_aug[-1]
        wrong_aug_neg_samples_5fold = neg_aug[-1]

        write_csv_tuple(wrong_pos_samples_5fold, os.path.join(folder, train_file.replace('.csv', '_pos_wrong.csv')))
        write_csv_tuple(wrong_neg_samples_5fold, os.path.join(folder, train_file.replace('.csv', '_neg_wrong.csv')))
        write_csv_tuple(wrong_aug_pos_samples_5fold, os.path.join(folder, train_file.replace('.csv', '_pos_aug_wrong.csv')))
        write_csv_tuple(wrong_aug_neg_samples_5fold, os.path.join(folder, train_file.replace('.csv', '_neg_aug_wrong.csv')))
    else:
        wrong_pos_samples_5fold = pos[-1]
        write_csv_tuple(wrong_pos_samples_5fold, os.path.join(folder, train_file.replace('_train.csv', '_pos_wrong.csv')))


if __name__ == '__main__':

    folder = 'data/all_category/data_run/_1M_shopee/colab'
    parser = ArgumentParser(add_help=False)

    parser.add_argument('-t', '--train_name', type=str)
    parser.add_argument('-m', '--model_name', type=str)
    args = parser.parse_args()
    
    csv_file = args.train_name

    use_semhash = False
    training_with_augument = False

    # if use_semhash:
    #     model_name = csv_file.replace('.csv', '_semhash_5k.pt')
    #     # model_name = csv_file.replace('.csv', '_semhash_full_vocab.pt')
    # else:
    #     model_name = csv_file.replace('.csv', '_preprocessed.pt')
    #     # model_name = csv_file.replace('.csv', '_token_full_vocab.pt')
    
    # model_name = "model/official_model/shopee_train_token_5k_ep1.pt"
    # model_name = 'all_cate_data_preprocessed.pt'
    model_name = args.model_name
    unknown_intent = 'UNKNOWN'

    # mode = 'split'
    mode = 'train'
    # mode = 'train-test'
    # mode = 'test-batch'
    # mode = "test"
    # mode = 'infer-batch'
    # mode = 'infer_file' 
    do_experiment(folder=folder,
                  csv_file=csv_file,
                  model_name=model_name,
                  unknown_intent=unknown_intent,
                  use_semhash=use_semhash,
                  mode=mode)
