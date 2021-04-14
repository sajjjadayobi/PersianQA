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
- Statistic

|           Split            | avg. question length | avg. paragraph length | avg. answer length |
| :------------------------: | :------------------: | :-------------------: | :----------------: |
|           Train            |          -           |           -           |         -          |
|           Test             |          -           |           -           |         -          |

## Models

Currently, two models on [HuggingFace🤗](https://huggingface.co/SajjadAyoubi/) are using the dataset.


|           Name             | Params | Training |
| :------------------------: | :------: | :---------: |
| [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa) |  558M  |   fine-tuned on SQuAD v2 + PersianQA   |
| [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa)    |  162M  |  fine-tuned on PersianQA    |


If you trained any model on the dataset, we'd be more than glad to hear the
details. Please, make a pull request for that regards.

- Installation

Transformers require `transformers` and `sentencepiece`, which can be installed using `pip`.
```sh
pip install transformers sentencepiece
```

- All the examples are based on the Bert version but you can use other versions as well

### How to use
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
  {'score': 9.901972770690918, 'text': '۲۰'}
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
  {'score': 9.901972770690918, 'text': '۲۰'}
  {'score': 12.117212295532227, 'text': 'پردازش زبان طبیعی'}
  ```

### Evaluation
Although, the GLEU metrics are not the best measures to evaluate the model on,
the results are as shown below. TODO: Adding more disc

#### Results
##### On [ParsiNLU](https://github.com/persiannlp/parsinlu)
- it contuns 570 question without (unanswerable questions)

|           Model            | F1 Score | Exact Match | Params |
| :------------------------: | :------: | :---------: | :----: |
|           Human            |  86.2%   |     -       |    -     |
|      XLM-Roberta(Ours)     |  **78.6%**   |   52.10%    |  558M |  
|       ParsBERT(Ours)       |  62.6%   |   35.43%    |  162M  |
|    mT5-small (ParsiNLU)    |  28.6%   |     -       |  300M  |
|    mT5-base (ParsiNLU)     |  43.0%   |     -       |  582M  |
|    mT5-large (ParsiNLU)    |  60.1%   |     -       |  1.2B  |
|     mT5-XL (ParsiNLU)      |  65.5%   |     -       |   -    |
 

##### On PersianQA testset
|           Model            | F1 Score | Exact Match | Params |
| :------------------------: | :------: | :---------: | :----: |
| Our version of XLM-Roberta |  84.81%  |   70.40%    |  558M  |
| Our version of ParsBERT    |  70.06%  |   53.55%    |  162M  |


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
