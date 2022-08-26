import math
import logging

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

from core.utils import utils
from common.utils.profiling import timeit
from common.utils.constants import (TRAINING_STATUS_STOPPED, TRAINING_STATUS_STOPPING, TRAINING_STATUS_TRAINING)

torch.backends.cudnn.enabled = True

logger = logging.getLogger()


class Encoder(nn.Module):
    def __init__(self, label_size, hidden_size, max_words, dropout=0.3):
        super(Encoder, self).__init__()
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.max_words = max_words

        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Parameter(torch.FloatTensor(hidden_size, max_words))
        self.b1 = nn.Parameter(torch.FloatTensor(hidden_size))
        self.layer2 = nn.Parameter(torch.FloatTensor(label_size, hidden_size))
        self.b2 = nn.Parameter(torch.FloatTensor(label_size))

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters
        :return:
        """
        stdv1 = 1.0 / math.sqrt(self.hidden_size)
        self.layer1.data.uniform_(-stdv1, stdv1)
        self.b1.data.uniform_(-stdv1, stdv1)

        stdv2 = 1.0 / math.sqrt(self.label_size)
        self.layer2.data.uniform_(-stdv2, stdv2)
        self.b2.data.uniform_(-stdv2, stdv2)

    def forward(self, inps):
        layer1_output = F.relu(F.linear(inps, self.layer1, self.b1))
        layer1_output = self.dropout(layer1_output)
        logit = F.linear(layer1_output, self.layer2, self.b2)

        return logit


class IntentDetector(nn.Module):
    def __init__(self, use_semhash, hidden_size, batch_size=8, dropout=0.3, use_gpu=False,
                 lr=0.001, num_epochs=50, save_model_to='./checkpoint/rt-polarity-intent.model'):

        super(IntentDetector, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_model_to = save_model_to
        self.tot_batch = 0
        self.is_stop = False
        self.use_semhash = use_semhash

    def run_train(self, train_samples, train_labels, answers, origins, model_path, progress_handler=None, version_id=None, job_id=None):
        """
        :param train_samples: list of sentences
        :param train_labels: list of labels
        :return: pre-trained model
        """
        self.answers = answers
        self.origins = origins
        self.save_model_to = model_path
        train_data, self.vocab, self.label2index = utils.hash2index(samples=train_samples, labels=train_labels, preprocess=False)

        self.vocab_size = len(self.vocab)
        self.label_size = len(self.label2index)
        self.encoder = Encoder(label_size=self.label_size, hidden_size=self.hidden_size,
                               max_words=self.vocab_size, dropout=self.dropout)

        self.index2label = {v: k for k, v in self.label2index.items()}
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.loss_function = nn.CrossEntropyLoss()

        if self.use_gpu:
            self.encoder.cuda()
        print("Checking...")
        for epoch in range(self.num_epochs):
            '''
            if (progress_handler != None):
                try:
                    progress_handler.update_batch(
                        'training_intent_model', epoch * 1. / self.num_epochs)
                except Exception as ex:
                    pass
            '''
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            self.encoder.train()
            losses = []
            batches = utils.getBatch(train_data, self.batch_size)
            for i in tqdm(range(0, len(batches)), total=len(batches), desc='   + Process'):
                sents_tensor, golds_tensor = utils.prepare_sequences(batches[i], vocab_size=self.vocab_size)

                if self.use_gpu:
                    sents_tensor = sents_tensor.cuda()
                    golds_tensor = golds_tensor.cuda()

                self.encoder.zero_grad()
                preds = self.encoder(sents_tensor)
                loss = self.loss_function(preds, golds_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
                self.optimizer.step()

                losses.append(loss.data.cpu().numpy()
                              if self.use_gpu else loss.data.numpy())
                # if i % 50 == 0:
                #    print('epoch: ', epoch, 'loss: ', numpy.mean(losses))
                # print('epoch: ', epoch, 'iter: ', i+1)
                self.tot_batch += 1
                batch_progress = self.tot_batch / \
                    (self.num_epochs * len(batches))
                # self.redis_db.set('intent_classifier' + self.save_model_to, batch_progress)

                if (progress_handler != None):
                    try:
                        response = progress_handler.update_batch(
                            version_id, job_id, batch_progress, status=TRAINING_STATUS_TRAINING)
                        if response is not None:
                            status = response.get(
                                'data', {}).get('status', None)
                            if status in (TRAINING_STATUS_STOPPED, TRAINING_STATUS_STOPPING):
                                progress_handler.update_batch(
                                    version_id, job_id, batch_progress, status=TRAINING_STATUS_STOPPED)
                                self.is_stop = True
                                return
                    except Exception as ex:
                        pass
            print("   + Loss: %.6f"%(np.mean(np.array(losses))))
            
            # Save checkpoint
            # if epoch % 3 == 0:
            self.save_model(model_path.replace('.pt', f'_ep{epoch+1}.pt'))

                
        #
        # if len(train_samples) <= 500:
        #     test_samples=train_samples
        #     test_labels=train_labels
        # else:
        #     test_size=500
        #     if test_size < len(set(train_labels)):
        #         test_size=len(set(train_labels))
        #     _, test_samples, _, test_labels=train_test_split(train_samples, train_labels, test_size=test_size,
        #                                                        stratify=train_labels)
        #
        # golds, preds=self.evaluation(test_samples, test_labels)
        #
        # accuracy=metrics.accuracy_score(golds, preds)
        # p, r, f1, s=metrics.precision_recall_fscore_support(golds, preds,
        #                                                       average='weighted',
        #                                                       warn_for=[])
        #
        # print('Training: Acc: %0.3f, P: %0.3f, R: %0.3f, F1: %0.3f' %
        #       (accuracy, p, r, f1))

    def evaluation(self, test_samples, test_labels):
        preds, golds=[], []
        for sample, label in zip(test_samples, test_labels):
            pred, results=self.prediction(sample)
            preds.append(pred)
            golds.append(self.label2index[label])

        return golds, preds

    def predict_batch(self, samples):
        def get_batch(data, batch_size=1024):
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
        self.encoder.eval()
        label2prob = []
        with torch.no_grad():
            batches = get_batch(samples)
            for batch in tqdm(batches, total=len(batches), desc='predict...'):
                sents_tensor = utils.preprocess_batch(batch, self.vocab, self.use_semhash, self.vocab_size)
                if self.use_gpu:
                    sents_tensor = sents_tensor.cuda()
                output = self.encoder(sents_tensor)
                output = torch.softmax(output, dim=1)

                probs, preds = torch.max(output, 1)
                for prob, pred in zip(probs, preds):
                    lbl = self.index2label[pred.item()]
                    label2prob.append(dict([(lbl, prob.item())]))

            return label2prob

    @ timeit
    def prediction(self, test_sample, topk=1):
        """
        :param test_samples: list of sentences
        :param test_labels: list of labels
        :return:
            predictions: list of predictions
            golds: list of labels
            resutls: dictionary of labels and probabilities
        """
        self.encoder.eval()
        label2prob = []
        with torch.no_grad():
            sent_tensor = utils.preprocess_one_sentence(
                test_sample, self.vocab, self.use_semhash, self.vocab_size)
            if self.use_gpu:
                sent_tensor = sent_tensor.cuda()

            output = self.encoder(sent_tensor)
            output = torch.softmax(output, dim=1)

            if topk == -1:
                k = output.shape[1]
            else:
                k = topk

            topk_probs, topk_preds = torch.topk(output, k=k)
            topk_probs = topk_probs.squeeze(0).cpu().tolist() if self.use_gpu else topk_probs.squeeze(0).tolist()
            topk_preds = topk_preds.squeeze(0).cpu().tolist() if self.use_gpu else topk_preds.squeeze(0).tolist()

            for prob, pred in zip(topk_probs, topk_preds):
                lbl = self.index2label[pred]
                label2prob.append((lbl, prob))

            _, pred = torch.max(output, 1)

            # print("label2prob: {}".format(label2prob))
            return pred.cpu().item(), label2prob

    def save_model(self, model_path, data_save=None):
        if self.is_stop:
            return
        # to save the pre-trained model for testing
        if data_save is not None:
            state = data_save
        else:
            state = {
                'state_dict': self.encoder.state_dict(),
                'vocab': self.vocab,
                'label2index': self.label2index,
                'hidden_size': self.hidden_size,
                'answers': self.answers,
                'origins': self.origins
            }
        with open(model_path, 'wb') as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)

    def load_model(self, model_file):
        # load the pre-trained model
        with open(model_file, 'rb') as f:
            # If we want to use GPU and CUDA is correctly installed
            if self.use_gpu and torch.cuda.is_available():
                state = torch.load(f)
            else:
                # Load all tensors onto the CPU
                state = torch.load(f, map_location='cpu')
        logger.debug(f'model state = {str(state)}')
        if not state:
            return None
        self.vocab = state['vocab']
        self.label2index = state['label2index']
        self.hidden_size = state['hidden_size']
        self.vocab_size = len(self.vocab)
        self.label_size = len(self.label2index)
        self.answers = state['answers']
        self.origins = state['origins']

        self.index2label = {v: k for k, v in self.label2index.items()}

        self.encoder = Encoder(label_size=self.label_size, hidden_size=self.hidden_size,
                               max_words=self.vocab_size, dropout=self.dropout)
        self.encoder.load_state_dict(state['state_dict'])

        if self.use_gpu:
            self.encoder.cuda()
        return True

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
