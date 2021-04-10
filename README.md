<a href="https://www.kaggle.com/"><img alt="Kaggle" src="https://img.shields.io/static/v1?label=Kaggle&message=Click&logo=Kaggle&color=20BEFF"/></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Click&color=yellow"></a>
<a href="https://paperswithcode.com/"><img src="https://img.shields.io/static/v1?label=%F0%9F%93%8E%20Papers%20With%20Code&message=Click&color=21cbce"></a>

<!-- markdown-toc start - Don't edit this section manually. -->

**Table of Contents**

- [PersianQA: a dataset for Persian Question Answering](#persianqa-a-dataset-for-persian-question-answering)
  - [Models](#models)
    - [Installation](#installation)
    - [Examples](#examples)
      - [Transformers](#transformers)
        - [TensorFlow 2.0](#tensorflow-20)
        - [PyTorch](#pytorch)
      - [Pipelines](#pipelines)
      - [Manual approach](#manual-approach)
        - [PyTorch](#pytorch-1)
        - [TensorFlow 2.0](#tensorflow-20-1)
    - [Evaluation](#evaluation)
      - [On ParsiNLU](#on-parsinlu)
- [Citation](#citation)

<!-- markdown-toc end -->

# PersianQA: a dataset for Persian Question Answering

The dataset provided here has more than 9,000 training data and about 900 test
data available.

Moreover, the first models trained on the dataset, Transformers, are available.

## Models

Currently, two models on [Hugging Face](https://huggingface.co/SajjadAyoubi/)
are using the dataset.

- [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa)
- [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa)

If you trained any model on the dataset, we'd be more than glad to hear the
details. Please, make a pull request for that regards.

### Installation

Transformers require `transformers`, `sentencepiece` and `tokenizer`, which can
be installed using `pip`.

```sh
pip install transformers sentencepiece tokenizer
```

### Examples

All the examples are based on the Bert version.

#### Transformers

##### TensorFlow 2.0

```python
from transformers import AutoConfig, BertTokenizer, TFBertForQuestionAnswering

model_name = "SajjadAyoubi/bert-base-fa-qa"
model = TFBertForQuestionAnswering.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
```

##### PyTorch

```python
from transformers import AutoConfig, BertTokenizer, BertForQuestionAnswering

model_name = "SajjadAyoubi/bert-base-fa-qa"
model = BertForQuestionAnswering.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
```

#### Pipelines

In case you are not familiar with Transformers, you can use pipelines instead.

```python
from transformers import pipeline

model_name = "SajjadAyoubi/bert-base-fa-qa"
qa_pipeline = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=model_name,
)

context = "من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم"
question = "فامیلی من چیه؟"

qa_pipeline(
    {
        "context": context,
        "question": question,
    }
)

# >>> {answer: "ایوبی"}

```

#### Manual approach

##### PyTorch

```python

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "SajjadAyoubi/bert-base-fa-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()

text = "من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم"
questions = [
    "فامیلی من چیه؟",
    "به چی علاقه دارم؟",
]

for question in questions:
    inputs = tokenizer(
        question,
        text,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits[0][answer_start:]) + answer_start + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    print(f"Question: {question}")
    print(f"Answer: {answer}")
```

##### TensorFlow 2.0

```python
import tensorflow as tf
from transformers import AutoTokenizer, AutoTFModelForQuestionAnswering
```

### Evaluation

#### On ParsiNLU

Although, the GLEU metrics are not the best measures to evaluate the model on,
the results are as shown below.

|           Model            | F1 Score | Exact Match |
| :------------------------: | :------: | :---------: |
| Our version of XLM-Roberta |  72.88%  |   50.70%    |

# Citation

we didn't publish any paper about this work, but! Please cite in your
publication as the following: Yet, we didn't publish any papers on the work.
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
