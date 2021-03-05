""" Finetuning the bert like models for NER"""

import argparse
import glob
import logging
import os
import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.layers import ConditionalRandomField
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import (
    open as custom_open,
    ViterbiDecoder,
    to_array
)
from tokenizer import CustomTokenizer as Tokenizer
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import datetime

logger = logging.getLogger(__name__)

today = datetime.date.today().strftime("%Y%m%d")

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split('\t')
                except:
                    print(c, l)
                    raise
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < args.max_length:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NamedEntityRecognizer(ViterbiDecoder):
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)

        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
#         print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(os.path.join(args.output_dir, f'best_model_weights_{today}.h5'))
        logger.info(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        if args.do_test:
            f1, precision, recall = evaluate(test_data)
            logger.info(
                'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
                (f1, precision, recall)
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--config_file",
        default="bert_config.json",
        type=str,
        help="bert config file name",
    )
    parser.add_argument(
        "--checkpoint_file",
        default="bert_model.ckpt",
        type=str,
        help="bert ckpt file name",
    )
    parser.add_argument(
        "--vocab_file",
        default="vocab.txt",
        type=str,
        help="bert vocab file name",
    )
    parser.add_argument(
        "--pretrained_weights",
        default=None,
        type=str,
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .txt files for the task."
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="Name of the training data."
    )
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        required=True,
        help="Name of the val data."
    )
    parser.add_argument(
        "--labels_path",
        default=None,
        type=str,
        required=True,
        help="path od labels.txt"
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="Name of the testing data."
    )
    parser.add_argument(
        "--max_length",
        default=256,
        type=int,
        help="The maximum number of tokens for input.",
    )
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per for training.")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--bert_layers", default=12, type=float, help="Number of bert layers")
    parser.add_argument("--crf_lr_multiplier", default=100, type=float, help="multiplier for crf learning rate")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # load data
    train_data = load_data(os.path.join(args.data_dir, args.train_file))
    valid_data = load_data(os.path.join(args.data_dir, args.val_file))

    logger.info(f"Train size: {len(train_data)}")

    if args.do_test:
        test_data = load_data(os.path.join(args.data_dir, args.test_file))

    # labels
    labels = []
    with open(args.labels_path, "r") as f:
        for label in f:
            labels.append(label.strip("\n"))

    id2label = dict(enumerate(labels))
    label2id = {j: i for i, j in id2label.items()}
    num_labels = len(labels) * 2 + 1

    # load tokenizer
    tokenizer = Tokenizer(os.path.join(args.model_path, args.vocab_file), do_lower_case=True)

    # load model
    model = build_transformer_model(
        config_path=os.path.join(args.model_path, args.config_file),
        checkpoint_path=os.path.join(args.model_path, args.checkpoint_file) if args.checkpoint_file is not None else None,
    )

    output_layer = 'Transformer-%s-FeedForward-Norm' % (args.bert_layers - 1)
    output = model.get_layer(output_layer).output
    output = Dense(num_labels)(output)
    CRF = ConditionalRandomField(lr_multiplier=args.crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)

    if args.pretrained_weights is not None:
        model.load_weights(args.pretrained_weights)

    model.summary()
    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(args.learning_rate),
        metrics=[CRF.sparse_accuracy]
    )

    NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

    evaluator = Evaluator()
    train_generator = data_generator(train_data, args.batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        callbacks=[evaluator]
    )
    logger.info(f"Done! Trained weights are saved in {args.output_dir}")
