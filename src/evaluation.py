"""
this script is for PyTorch but by a little change is works for Tensorflow as well
changes are:
    AutoModelForQuestionAnswering -> TFAutoModelForQuestionAnswering
    AnswerPredictor -> TFAnswerPredictor
"""

# local imports
from utils import AnswerPredictor
from load_ds import read_qa, c2dict

# official imports
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_metric
from collections import Counter
import re

model_name = # your model name
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# loading testset
test_ds = c2dict(read_qa('pqa_test.json'))
questions, contexts, answers = test_ds['question'], test_ds['context'], test_ds['answers']

# creating predictions
predictor = AnswerPredictor(model, tokenizer, device='cuda')
preds = predictor(questions, contexts, batch_size=12)

# cleaner function
def cleaner(text):
    return re.sub('\u200c', " ", text).strip()
  
# -------------------------------------------------------------------- Method One (datasets.load_metric)
# SQuAD2.0 HuggingFace metrics 
metric = load_metric("squad_v2") # the dataset is like SQuAD2.0

formatted_preds = [{"id": str(k), 
                    "prediction_text": cleaner(v['text']),
                    "no_answer_probability": 0.0} 
                    for k, v in preds.items()]

references = [{"id": str(i), 
               "answers": {'answer_start': a['answer_start'], 
                          'text': map(cleaner, a['text'])}}
              for i, a in enumerate(answers)]

print(metric.compute(predictions=formatted_preds, references=references))

# ------------------------------------------------------------------- Method Two (offical SQuADv2)
# offical SQuAD2.0 evaluation script. Modifed slightly for this dataset
def f1_score(prediction, ground_truth):
    prediction_tokens = cleaner(prediction)
    ground_truth_tokens = cleaner(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (cleaner(prediction) == cleaner(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0
    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

  
y_hat = [v['text'] for v in preds.values()]
y = [v['text'] if len(v['text'])>0 else [''] for v in answers]

print(evaluate(y, y_hat))
