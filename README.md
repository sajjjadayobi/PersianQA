<span align="center">
    <a href="https://www.kaggle.com/"><img alt="Kaggle" src="https://img.shields.io/static/v1?label=Kaggle&message=Click&logo=Kaggle&color=20BEFF"/></a>
    <a href="https://huggingface.co/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Click&color=yellow"></a>
    <a href="https://paperswithcode.com/"><img src="https://img.shields.io/static/v1?label=%F0%9F%93%8E%20Papers%20With%20Code&message=Click&color=21cbce"></a>
    <a href="https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Demo&logo=Google%20Colab&color=f9ab00"></a>
</span>

# PersianQA: a dataset for Persian Question Answering

Persian Question Answering (PersianQA) Dataset is a reading comprehension
dataset on [Persian Wikipedia](https://fa.wikipedia.org/). The crowd-sourced
dataset consists of more than 9,000 entries. Each entry can be either an
_impossible to answer_ or a question with one or more answers spanning in the
passage (the _context_) from which the questioner proposed the question.
Much like the SQuAD2.0 dataset, the impossible or _unanswerable_ questions can be
utilized to create a system which "knows that it doesn't know the answer".

On top of that, the dataset has 900 test data available.
Moreover, the first models trained on the dataset, Transformers, are available.

All the crowdworkers of the dataset are native Persian speakers. Also, it worth
mentioning that the contexts are collected from all categories of the Wiki
(Historical, Religious, Geography, Science, etc.)

At the moment, each context has 7 pairs of questions with one answer and 3
impossible questions.

As mentioned before, the dataset is inspired by the famous SQuAD2.0 dataset and is
compatible with and can be merged into it. But that's not all, the dataset here
has some relative advantages to the original SQuAD, some of which are listed below:

- Lengthier contexts
- Increased number of articles (despite having less data)
- More questions per contexts (7 comparing to 5)
- Including _informal ("Mohaaverei")_ entries
- More varied answers (names, locations, dates and more)

You can check out an online [iPython Demo Notebook on Google Colab ](https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb).

## Dataset Information

- Description
- Access
- Example
- Statistic

| Split | # of instances | # of unanswerables | avg. question length | avg. paragraph length | avg. answer length |
| :---: | :------------: | :----------------: | :------------------: | :-------------------: | :----------------: |
| Train |     9,000      |       2,700        |         8.39         |        224.58         |        9.61        |
| Test  |      938       |        280         |         8.02         |        220.18         |        5.99        |

The lengths are on token level.

## Models

Currently, two models (baseline) on [HuggingFace🤗](https://huggingface.co/SajjadAyoubi/) model hub are using the dataset.
The models are listed in the table below.

|                                          Name                                          | Params |              Training              |
| :------------------------------------------------------------------------------------: | :----: | :--------------------------------: |
| [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa) |  558M  | fine-tuned on SQuAD2.0 + PersianQA |
|         [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa)         |  162M  |      fine-tuned on PersianQA       |

You can try out our existing models and study examples. For more information
on the examples, visit [this page]().

**In case you have trained any model on the dataset, we'd be more than glad to
hear the details. Please, make a pull request for that regards.**

### How to use

All the examples are based on the Bert version but you can use other versions as well.

#### Requirements

Transformers require `transformers` and `sentencepiece`, both of which can be
installed using `pip`.

```sh
pip install transformers sentencepiece
```

#### Pipelines 🚀

In case you are not familiar with Transformers, you can use pipelines instead.

Note that, pipelines can't have _no answer_ for the questions.

```python
from transformers import pipeline

model_name = "SajjadAyoubi/bert-base-fa-qa"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

text = "سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم "
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

for question in questions:
    print(qa_pipeline({"context": text, "question": question}))

# >>> {'score': 0.4839823544025421, 'start': 8, 'end': 18, 'answer': 'سجاد ایوبی'}
# >>> {'score': 0.3747948706150055, 'start': 24, 'end': 32, 'answer': '۲۰ سالمه'}
# >>> {'score': 0.5945395827293396, 'start': 38, 'end': 55, 'answer': 'پردازش زبان طبیعی'}
```

#### Manual approach 🔥

Using the Manual approach, it is possible to have _no answer_ with even better
performance.

- PyTorch

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "SajjadAyoubi/bert-base-fa-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

text = "سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم "
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

# this class is from src/utils.py and you can read more about it
predictor = AnswerPredictor(model, tokenizer, device="cpu", n_best=10)
preds = predictor(questions, [text] * 3, batch_size=3)

for k, v in preds.items():
    print(v)
```

Produces an output such below:

```
100%|██████████| 1/1 [00:00<00:00,  3.56it/s]
{'score': 8.040637016296387, 'text': 'سجاد ایوبی'}
{'score': 9.901972770690918, 'text': '۲۰'}
{'score': 12.117212295532227, 'text': 'پردازش زبان طبیعی'}
```

- TensorFlow 2.X

```python
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

model_name = "SajjadAyoubi/bert-base-fa-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

text = "سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم "
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

# this class is from src/utils.py, you can read more about it
predictor = TFAnswerPredictor(model, tokenizer, n_best=10)
preds = predictor(questions, [text] * 3, batch_size=3)

for k, v in preds.items():
    print(v)
```

Produces an output such below:

```text
100%|██████████| 1/1 [00:00<00:00,  3.56it/s]
{'score': 8.040637016296387, 'text': 'سجاد ایوبی'}
{'score': 9.901972770690918, 'text': '۲۰'}
{'score': 12.117212295532227, 'text': 'پردازش زبان طبیعی'}
```

Or you can access the whole demonstration using [HowToUse iPython Notebook on
Google
Colab](https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/HowToUse.ipynb)

### Evaluation

To evaluate your models, you can use the provided [evaluation script](https://github.com/sajjjadayobi/PersianQA/blob/main/src/evaluation.py).

#### Results

<!-- TODO: Explain what are these metrics -->

Although, the GLEU metrics are not the best measures to evaluate the model on,
the results are as shown below.
Best baseline scores are indicated as bold

##### On [ParsiNLU](https://github.com/persiannlp/parsinlu)

- it contuns 570 question without (unanswerable questions)

|         Model         | F1 Score  | Exact Match | Params |
| :-------------------: | :-------: | :---------: | :----: |
|         Human         |   86.2%   |      -      |   -    |
| Our XLM-Roberta-Large | **78.6%** |   52.10%    |  558M  |
|     Our ParsBERT      |   62.6%   |   35.43%    |  162M  |
| ParsiNLU's mT5-small  |   28.6%   |      -      |  300M  |
|  ParsiNLU's mT5-base  |   43.0%   |      -      |  582M  |
| ParsiNLU's mT5-large  |   60.1%   |      -      |  1.2B  |
|   ParsiNLU's mT5-XL   |   65.5%   |      -      |   -    |

##### On PersianQA testset

|         Model         |  F1 Score  | Exact Match | Params |
| :-------------------: | :--------: | :---------: | :----: |
| Our XLM-Roberta-Large | **84.81%** |   70.40%    |  558M  |
|     Our ParsBERT      |   70.06%   |   53.55%    |  162M  |

# Citation

Yet, we didn't publish any papers on the work.
However, if you did, please cite us properly with an entry like one below.

```bibtex
@misc{PersianQA,
  author          = {Ayoubi, Sajjad \& Davoodeh, Mohammad Yasin},
  title           = {PersianQA: a dataset for Persian Question Answering},
  year            = 2021,
  publisher       = {GitHub},
  journal         = {GitHub repository},
  howpublished    = {\url{https://github.com/SajjjadAyobi/PersianQA}},
}
```

## Acknowledgment

- Thanks to _Navid Kanani_ and _Abbas Ayoubi_
- Thanks to Google’s Colab😄 and HuggingFace🤗 for making this work easier 
