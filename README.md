<a href="https://www.kaggle.com/"><img alt="Kaggle" src="https://img.shields.io/static/v1?label=Kaggle&message=Click&logo=Kaggle&color=20BEFF"/></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Click&color=yellow"></a>
<a href="https://paperswithcode.com/"><img src="https://img.shields.io/static/v1?label=%F0%9F%93%8E%20Papers%20With%20Code&message=Click&color=21cbce"></a>

# PersianQA: a dataset for Persian Question Answering

The dataset provided here has more than 9,000 training data and about 900 test data available.
Moreover, the first models trained on the dataset, Transformers, are available.

## Data
- Description
- access
- Example
- Stats

## Models

Currently, two models on [HuggingFace🤗](https://huggingface.co/SajjadAyoubi/) are using the dataset.

- [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa)
  -  fine-tuned with PersianQA
- [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa)
  -  fine-tuned with SQuAD v2 + PersianQA

If you trained any model on the dataset, we'd be more than glad to hear the
details. Please, make a pull request for that regards.

- Installation

Transformers require `transformers` and `sentencepiece`, which can be installed using `pip`.
```sh
pip install transformers sentencepiece
```

- All the examples are based on the Bert version but you can use other versions as well

#### Pipelines 🚀

In case you are not familiar with Transformers, you can use pipelines instead.
  - pipelines can't have no answer for questions

```python
from transformers import pipeline

model_name = "SajjadAyoubi/bert-base-fa-qa"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

text = r"""سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم """
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

for question in questions:
    print(qa_pipeline({"context": text, "question": question}))

# >>> {'score': 0.4839823544025421, 'start': 8, 'end': 18, 'answer': 'سجاد ایوبی'}
# >>> {'score': 0.3747948706150055, 'start': 24, 'end': 32, 'answer': '۲۰ سالمه'}
# >>> {'score': 0.5945395827293396, 'start': 38, 'end': 55, 'answer': 'پردازش زبان طبیعی'}
```

#### Manual approach 🔥
using Manual approach you can have no answer and better performance

- Pytorch
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "SajjadAyoubi/bert-base-fa-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

text = r"""سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم """
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

# this class is from src/utils.py and you can read more about it
predictor = AnswerPredictor(model, tokenizer, device='cpu', n_best=10)
preds = predictor(questions, [text]*3, batch_size=3)

for k, v in preds.items():
    print(v)
```
  - the output is
  ```sh
  100%|██████████| 1/1 [00:00<00:00,  3.56it/s]
  {'score': 8.040637016296387, 'text': 'سجاد ایوبی'}
  {'score': 9.901972770690918, 'text': '۲۰ سالمه'}
  {'score': 12.117212295532227, 'text': 'پردازش زبان طبیعی'}
  ```

- TensorFlow 2.X
```python
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

model_name = '"SajjadAyoubi/bert-base-fa-qa"'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, from_pt=True)

text = "سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم "
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

# this class is from src/utils.py, you can read more about it
predictor = TFAnswerPredictor(model, tokenizer, n_best=10)
preds = predictor(questions, [text]*3, batch_size=3)

for k, v in preds.items():
    print(v)
```
  - the output is
  ```sh
  100%|██████████| 1/1 [00:00<00:00,  3.56it/s]
  {'score': 8.040637016296387, 'text': 'سجاد ایوبی'}
  {'score': 9.901972770690918, 'text': '۲۰ سالمه'}
  {'score': 12.117212295532227, 'text': 'پردازش زبان طبیعی'}
  ```

### Evaluation
Although, the GLEU metrics are not the best measures to evaluate the model on,
the results are as shown below.

#### On [ParsiNLU](https://github.com/persiannlp/parsinlu)
- it contuns 570 question without (no answer)

|           Model            | F1 Score | Exact Match |
| :------------------------: | :------: | :---------: |
| Our version of XLM-Roberta |  78.59%  |   52.10%    |
| Our version of ParsBERT    |  62.60%  |   35.43%    |


#### On PersianQA testset
|           Model            | F1 Score | Exact Match |
| :------------------------: | :------: | :---------: |
| Our version of XLM-Roberta |  84.81%  |   70.40%    |
| Our version of ParsBERT    |  70.06%  |   53.55%    |


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
