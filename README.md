<a href="https://www.kaggle.com/"><img alt="Kaggle" src="https://img.shields.io/static/v1?label=Kaggle&message=Click&logo=Kaggle&color=20BEFF"/></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Click&color=yellow"></a>
<a href="https://paperswithcode.com/"><img src="https://img.shields.io/static/v1?label=%F0%9F%93%8E%20Papers%20With%20Code&message=Click&color=21cbce"></a>

# PersianQA: a dataset for Persian Question Answering

The dataset provided here has more than 9,000 training data and about 900 test data available.
Moreover, the first models trained on the dataset, Transformers, are available.

## Data
- Discribtions
- access
- Example
- Stats

## Models

Currently, two models on [Hugging Face](https://huggingface.co/SajjadAyoubi/) are using the dataset.

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

#### Pipelines

In case you are not familiar with Transformers, you can use pipelines instead.
  - pipelines can't have no answer for questions

```python
from transformers import pipeline

model_name = "SajjadAyoubi/bert-base-fa-qa"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

text = r"""سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم """
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

for i in questions:
    print(qa_pipeline({"context": text, "question": question}))

# >>> {answer: 'سجاد ایوبی'}
# >>> {}
# >>> {}
```

#### Manual approach (PyTorch)
- using Manual approach you can have no answer and better performance

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "SajjadAyoubi/bert-base-fa-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

text = r"""سلام من سجاد ایوبی هستم ۲۰ سالمه و به پردازش زبان طبیعی علاقه دارم """
questions = ["اسمم چیه؟", "چند سالمه؟", "به چی علاقه دارم؟"]

# this class is from PersianQA/utils and you can read more about it
infer = QAInference(model, tokenizer, device='cuda', n_best=10)
preds = infer(questions, contexts*3, batch_size=3)
print(preds)
```

##### TensorFlow 2.0

```python
import tensorflow as tf
from transformers import AutoTokenizer, AutoTFModelForQuestionAnswering
```

### Evaluation
Although, the GLEU metrics are not the best measures to evaluate the model on,
the results are as shown below.

- On ParsiNLU 
|           Model            | F1 Score | Exact Match |
| :------------------------: | :------: | :---------: |
| Our version of XLM-Roberta |  73.44%  |   50.70%    |
| Our version of ParsBERT    |  61.50%  |   43.70%    |


- On PersianQA testset
|           Model            | F1 Score | Exact Match |
| :------------------------: | :------: | :---------: |
| Our version of XLM-Roberta |  72.88%  |   50.70%    |
| Our version of ParsBERT    |  56.88%  |   43.70%    |


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
