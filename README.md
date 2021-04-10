# PersianQA: First Models and Dataset For Persian Question Answering 
- PersianQA dataset (9k Train, 900 Test)
- Transformers models for Persian(Farsi) Question Answering

# Dataset:
- it's available on:
  - this github repo
  - kaggle datasets
  - huggingface datasets
  - paperswithcode datasets


# Model
  - [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa)
  - [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa)

## Installation 🤗
- install transformers package for using this as simple as posible
  ```bash 
  !pip install -q transformers
  !pip install -q sentencepiece
  !pip install -q tokenizer
  ```
  
## How to use 
- these examples are base on the Bert Model 

### TensorFlow 2.0 

```python
from transformers import AutoConfig, BertTokenizer, TFBertForQuestionAnswering

model_name = 'SajjadAyoubi/bert-base-fa-qa'
model = TFBertForQuestionAnswering.from_pretrained(model_name).
config = AutoConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
```

### Pytorch

```python
from transformers import AutoConfig, BertTokenizer, BertForQuestionAnswering

model_name = 'SajjadAyoubi/bert-base-fa-qa'
model = BertForQuestionAnswering.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
```

## Examples
- if you are not fimilar with Transformers use pipeline
- if you wanna more access to the model use manually

### Pipeline 
```python
from transformers import pipeline

model_name = 'SajjadAyoubi/bert-base-fa-qa'
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

ccontext = 'من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم '
question = 'فامیلی من چیه؟'

qa_pipeline({'context': context, 'question': question})
>>> {answer: 'ایوبی'}
```

### Manually 

#### Pytorch
```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = 'SajjadAyoubi/bert-base-fa-qa'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()

text = 'من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم'
questions = ["فامیلی من چیه؟",
             "به چی علاقه دارم؟",]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    outputs = model(**inputs)
  
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits[0][answer_start:]) + answer_start + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")
```
#### Tensorflow 2.0
```python
import tensorflow as tf
from transformers import AutoTokenizer, AutoTFModelForQuestionAnswering
```

## Evaluation
### On ParsiNLU
- **Anybody who works in NLP knows that GLEU metrics aren't really well**
- if you not sure about that fact answer the questions and compute your f1 and Exact 😊


  | Model | F1 Score | Exact Match |
  |  :---:  |  :---:  | :---: |
  | Our XLM-Roberta | 72.88% | 50.70% |

- But I believe that the model is better than these numbers


# Cite

we didn't publish any paper about this work, but! Please cite in your publication as the following:

```bibtex
@misc{PersianQA,
  author = {Sajjad Ayoubi, Muhammad Yasin Davoodeh},
  title = {PersianQA: Persian Question Answersing with Dataset & Models},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sajjjadayobi/PersianQA}},
}
```
