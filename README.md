# PersianQA: Without any Datasets 🙄 
- Transformers models for Persian(Farsi) Question Answering
- these models are not actually Persian but I use some tricks to improve them on the Persian Language 

# Model
  - [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa)
    - about 700MB it's ready both TF & Torch
  - [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa)
    - about 2.4GB it's ready both TF & Torch
    
## Online Tester
  - on [Colab](https://colab.research.google.com/drive/1Y7yisfVnhFYtzw7KvE3QFbruzvGe0qgk?usp=sharing)

## Installation 🤗
- install transformers pakcage for using this as simple as posible

  ```bash 
      !pip install -q transformers
  ```
- if you use the xlm-roberta you need to install sentencepiece
  
  ```bash 
      !pip install -q sentencepiece
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

ccontext = 'من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم و چون به نظرم خیلی جزابه هست'
question = 'فامیلی من چیه؟'

qa_pipeline({'context': context, 'question': question})
>>> {answer: 'ایوبی'}
```

### Manually (Pytorch)
```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = 'SajjadAyoubi/bert-base-fa-qa'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()

text = 'من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم و چون به نظرم خیلی جزابه هست'
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
## Evaluation On ParsiNLU
- **Anybody who works in NLP knows that GLEU metrics aren't really well**
- if you not sure about that fact answer the questions and compute your f1 and Exact 😊


  | Model | F1 Score | Exact Match |
  |  :---:  |  :---:  | :---: |
  | Our XLM-Roberta | 71.08% | 47.82% |


- But I believe that my model is better than these numbers
