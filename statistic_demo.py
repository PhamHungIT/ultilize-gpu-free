import os
import json
import collections
import configparser
import pandas as pd
from tqdm import tqdm
import streamlit as st
from pyvi import ViTokenizer
from fuzzywuzzy import fuzz
from core import model_wrapper


@st.cache(allow_output_mutation=True)
def read_data(csv_file):
    data = dict()
    df = pd.read_csv(csv_file, delimiter='\t', error_bad_lines=False)
    samples = df['sample'].values.tolist()
    try:
        intents = df['intent'].values.tolist()
    except:
        intents = df['label'].values.tolist()
    for sample, intent in zip(samples, intents):
        if intent not in data:
            data[intent] = [sample]
        else:
            data[intent].append(sample)

    # kb_samples = load_kb_sample('local/intent_by_domain/resources/questions_annotated_no_dup_aug_with_remove_ques_word.txt')
    # data['GENERAL_KNOWNLEDGE.txt'] = kb_samples

    return data


@st.cache(allow_output_mutation=True)
def statistic_intent(csv_file):
    data = read_data(csv_file)
    count_intent = dict()
    for intent in data:
        count_intent[intent] = len(data[intent])

    count_intent = {k: v for k, v in sorted(count_intent.items(), key=lambda item: item[1], reverse=True)}
    total = 0
    n_intent = 0
    print(len(count_intent))
    for intent, count in count_intent.items():
        print(f'{intent}: {count}')
        if intent != 'UNKNOWN':
            total += count
            n_intent += 1
    print(f'Total: {total}')
    print(f'n_intent: {n_intent}')

    return count_intent


@st.cache(allow_output_mutation=True)
def statistic_token(csv_file):
    data = read_data(csv_file)

    count_token_by_intent = {}
    for intent, samples in data.items():
        intent = intent.upper()
        count_token_by_intent[intent] = {}
        for sample in tqdm(samples, total=len(samples)):
            sample = ViTokenizer.tokenize(sample)
            # pos_tag = ViPosTagger.postagging(ViTokenizer.tokenize(sample))
            # pos_tag_sample = list(zip(pos_tag[0], pos_tag[1]))
            # sample = ''
            # for item in pos_tag_sample:
            #     sample += item[0] + '#' + item[1] + ' '
            for token in sample.strip().split():
                if token.replace('#', ' | ') not in count_token_by_intent[intent]:
                    count_token_by_intent[intent][token.replace('#', ' | ')] = 1
                else:
                    count_token_by_intent[intent][token.replace('#', ' | ')] += 1

    count_token_by_intent = collections.OrderedDict(sorted(count_token_by_intent.items()))
    for intent in count_token_by_intent:
        count_token_by_intent[intent] = dict(sorted(count_token_by_intent[intent].items(), key=lambda item: item[1], reverse=True)[:20])

    json.dump(count_token_by_intent, open(os.path.join(folder, 'count_token_by_intent.json'), 'w'), ensure_ascii=False)
    print(len(count_token_by_intent))

    return count_token_by_intent


@st.cache(allow_output_mutation=True)
def statistic_sample_by_word(csv_file, word_search):
    data = read_data(csv_file)

    count = {}
    for intent, samples in data.items():
        for sample in samples:
            sample = sample.lower()
            if word_search in sample:
                if intent in count:
                    count[intent] += 1
                else:
                    count[intent] = 1

    count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
    print(count)

    return count


def init_gui():
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
        color: green;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title('MY INTENT')


def count_sample_by_intent():
    table_html = """
                | <b>INTENT</b> | <b>COUNT</b> |
                | --- | --- |
            """

    for intent, count in sample_by_intents.items():
        table_html += '|<p style="color:crimson;"><b>' + intent.strip() + '</b></p>|<p style="color:green;"><b>' + str(count) + '</b></p>'
        table_html += '\n'

    st.markdown(table_html, unsafe_allow_html=True)


def count_word_by_intent():
    intent_search = st.text_input("Intent search")

    if intent_search != '':
        if intent_search.strip().upper() in token_by_intents:
            st.write({intent_search.strip().upper(): token_by_intents[intent_search.strip().upper()]})
        else:
            for intent in token_by_intents:
                if fuzz.ratio(intent.lower(), intent_search.lower()) > 80 \
                        or ' '.join(intent_search.lower().split('_')) in ' '.join(intent.lower().split('_')):
                    print(intent_search)
                    print(intent)
                    print('------------------------------------------------------------')
                    st.write({intent.upper(): token_by_intents[intent.strip().upper()]})


def get_intent_by_word():
    word_search = st.text_input("Word search")

    if word_search != '':
        count_sample_by_word = statistic_sample_by_word(os.path.join(folder, csv_file), word_search)
        # count_sample_by_word = statistic_sample_by_word(os.path.join(folder, 'domain.csv'), word_search)

        table_html = """
                        | <b>INTENT</b> | <b>COUNT</b> |
                        | --- | --- |
                    """
        for intent, count in count_sample_by_word.items():
            table_html += '| <p style="color:crimson;"><b>' + intent + '</b></p> | <p style="color:green;"><b>' + str(
                count) + '</b></p>'
            table_html += '\n'
        st.markdown(table_html, unsafe_allow_html=True)


def demo():
    def init_gui():
        st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
            color: green;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title('INTENT DEMO')

    folder = "/media/nghind/DATA/projects/chatbot/backup/intent-services/local/intent_by_domain/resources/30112021"

    MODEL_CONFIG = 'config/trainer/model.cfg'
    model_config = configparser.ConfigParser()
    model_config.read(MODEL_CONFIG)

    model_active = {
        "token_domain": True,
        "token_asr_domain": False,
        "token_intent": False,
        "semhash_domain": False,
        "semhash_asr_domain": False,
        "semhash_intent": False,
    }

    init_gui()
    user_input = st.text_input("")

    if user_input != '':

        # """ Token level """
        if model_active['token_domain']:
            # token_domain_model_path = os.path.join(folder, "domain_token_limit_4k_full.pt")
            token_domain_model_path = os.path.join(folder, "domain_token_limit_4k_full_aug.pt")
            # token_domain_model_path = os.path.join('local/vf_chatbot', "data_token_limit_4k_full_aug.pt")
            token_domain_model = model_wrapper.load_model(token_domain_model_path, model_config['intent'], False, False)
            token_domain_predicted_label = model_wrapper.infer(token_domain_model, user_input)
        else:
            token_domain_predicted_label = ('', 0)

        if model_active['token_asr_domain']:
            # token_asr_domain_model_path = os.path.join(folder, "asr_domain_aug_first_pass_KHANHLINH_HOAIANH_289k_token_limit_4k_full.pt")
            token_asr_domain_model_path = os.path.join(folder, "asr_domain_aug_first_pass_KHANHLINH_HOAIANH_289k_token_limit_4k_full_aug.pt")
            token_asr_domain_model = model_wrapper.load_model(token_asr_domain_model_path, model_config['intent'],
                                                              False, False)
            token_asr_domain_predicted_label = model_wrapper.infer(token_asr_domain_model, user_input)
        else:
            token_asr_domain_predicted_label = ('', 0)

        if model_active['token_intent']:
            # token_intent_model_path = os.path.join(folder, "intent_token_limit_4k_full.pt")
            folder = "/media/nghind/DATA/projects/chatbot/backup/intent-services/local/vf_chatbot"
            token_intent_model_path = os.path.join(folder, "VF_Config_Data_Bot_24112021_delimiter_token_full_vocab.pt")
            # token_intent_model_path = os.path.join(folder, "vfe34_sample_intent_domain_intent_filter_aug_token_limit_4k_full_aug.pt")
            token_intent_model = model_wrapper.load_model(token_intent_model_path, model_config['intent'], False, False)
            token_intent_predicted_label = model_wrapper.infer(token_intent_model, user_input)
        else:
            token_intent_predicted_label = ('', 0)

        # """ Semhash level """
        if model_active['semhash_domain']:
            semhash_domain_model_path = os.path.join(folder, "domain_semhash_limit_4k_full.pt")
            semhash_domain_model = model_wrapper.load_model(semhash_domain_model_path, model_config['intent'], False,
                                                            False)
            semhash_domain_predicted_label = model_wrapper.infer(semhash_domain_model, user_input)
        else:
            semhash_domain_predicted_label = ('', 0)

        if model_active['semhash_asr_domain']:
            semhash_asr_domain_model_path = os.path.join(folder, "asr_domain_semhash_limit_4k_full.pt")
            semhash_asr_domain_model = model_wrapper.load_model(semhash_asr_domain_model_path, model_config['intent'],
                                                                False, False)
            semhash_asr_domain_predicted_label = model_wrapper.infer(semhash_asr_domain_model, user_input)
        else:
            semhash_asr_domain_predicted_label = ('', 0)

        if model_active['semhash_intent']:
            semhash_intent_model_path = os.path.join(folder, "intent_semhash_limit_4k_full.pt")
            semhash_intent_model = model_wrapper.load_model(semhash_intent_model_path, model_config['intent'], False,
                                                            False)
            semhash_intent_predicted_label = model_wrapper.infer(semhash_intent_model, user_input)
        else:
            semhash_intent_predicted_label = ('', 0)

        # """ GUI """

        st.markdown(
            """
        | <b>MODEL</b> | <b>LABEL</b> | <b>PROB</b> |
        | --- | --- | --- | --- |
        | <b>TOKEN DOMAIN</b> | <p style="color:crimson;"><b>%s</b></p> | <p style="color:green;">%.4f</p> |
        | <b>TOKEN ASR DOMAIN</b> | <p style="color:crimson;"><b>%s</b></p> | <p style="color:green;">%.4f</p> |
        | <b>TOKEN INTENT</b> | <p style="color:crimson;"><b>%s</b></p> | <p style="color:green;">%.4f</p> |
        | <b>SEMHASH DOMAIN</b> | <p style="color:crimson;"><b>%s</b></p> | <p style="color:green;">%.4f</p> |
        | <b>SEMHASH ASR DOMAIN</b> | <p style="color:crimson;"><b>%s</b></p> | <p style="color:green;">%.4f</p> |
        | <b>SEMHASH INTENT</b> | <p style="color:crimson;"><b>%s</b></p> | <p style="color:green;">%.4f</p> |
        """ % (token_domain_predicted_label[0].upper(), token_domain_predicted_label[1],
               token_asr_domain_predicted_label[0].upper(), token_asr_domain_predicted_label[1],
               token_intent_predicted_label[0].upper(), token_intent_predicted_label[1],
               semhash_domain_predicted_label[0].upper(), semhash_domain_predicted_label[1],
               semhash_asr_domain_predicted_label[0].upper(), semhash_asr_domain_predicted_label[1],
               semhash_intent_predicted_label[0].upper(), semhash_intent_predicted_label[1])
            , unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_kb_sample(txt_file):
    with open(txt_file, 'r') as fi:
        lines = [line.strip() for line in fi.readlines()]

    return lines


if __name__ == '__main__':

    folder = '/home/nghind/workspace/data/category_classification'
    csv_file = 'ict_category_kpro_websosanh_samples.csv'

    sample_by_intents = statistic_intent(os.path.join(folder, csv_file))
    token_by_intents = statistic_token(os.path.join(folder, csv_file))

    feature = st.sidebar.radio("Feature", ('Count sample by intent', 'Count word by intent', 'Get intent by word', 'Demo'), index=0)

    if feature == 'Count sample by intent':
        init_gui()
        count_sample_by_intent()

    elif feature == 'Count word by intent':
        init_gui()
        count_word_by_intent()

    elif feature == 'Get intent by word':
        init_gui()
        get_intent_by_word()

    elif feature == 'Demo':
        demo()