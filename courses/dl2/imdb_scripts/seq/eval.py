"""
Evaluation utility methods.
"""
import numpy as np
import torch

from sklearn.metrics import accuracy_score
from create_toks_conll import PAD, BOS_LABEL

from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.data.vocabulary import Vocabulary


def get_acc(preds1, preds2, gold, weight=0.5):
    preds = np.exp(preds1) * weight + np.exp(preds2) * (1 - weight)
    preds = np.array([np.argmax(p) for p in preds])
    return accuracy_score(gold, preds)


def eval_ner(learner, id2label, is_test=False):
    # set up AllenNLP evaluation metric
    mode = 'Test' if is_test else 'Validation'
    id2label = [f'B-{l}' if l in [PAD, BOS_LABEL] else l for l in id2label]
    namespace = 'ner_labels'
    label_vocab = Vocabulary(
        non_padded_namespaces=(namespace,),
        tokens_to_add={namespace: id2label})  # create the tag vocabulary
    f1_metric = SpanBasedF1Measure(label_vocab,
                                   tag_namespace=namespace,
                                   ignore_classes=[PAD, BOS_LABEL])
    preds, y = learner.predict_with_targs(is_test=is_test)
    # convert to tensors, add a batch dimension
    preds_tensor = torch.from_numpy(preds).unsqueeze(0)
    y_tensor = torch.from_numpy(y).unsqueeze(0)
    f1_metric(preds_tensor, y_tensor)
    all_metrics = f1_metric.get_metric(reset=True)
    print(f'{mode} f1 measure overall:', all_metrics['f1-measure-overall'])
    print(all_metrics)
    preds_fwd_ids = [np.argmax(p) for p in preds]
    acc_fwd = accuracy_score(y, preds_fwd_ids)
    print(f'{mode} token-level accuracy of NER model: %.4f.' % acc_fwd)
