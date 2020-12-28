# PersianQA: Without any Datasets 🙄 
- Transformers models for Persian(Farsi) Question Answering
- these models are not actually Persian but I use some tricks to improve them on the Persian Language 

# Model
  - SajjadAyoubi/bert-base-fa-qa
    - about 700MB it's ready both TF & Torch
  - SajjadAyoubi/xlm-roberta-large-fa-qa
    - about 2.4GB it's ready both TF & Torch

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

ccontext = 'سلام من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم و چون به نظرم خیلی جزابه هست'
question = 'فامیلی من چیه؟'

qa_pipeline({'context': context, 'question': question})
>>> {answer: 'ایوبی'}
```

### Manually (Pytorch)
```python
import torch
import numpy as np
from transformers import AutoConfig, BertTokenizer, BertForQuestionAnswering

model_name = 'SajjadAyoubi/bert-base-fa-qa'
model = BertForQuestionAnswering.from_pretrained(model_name).eval()
tokenizer = BertTokenizer.from_pretrained(model_name)

# inputs
ccontext = 'سلام من سجاد ایوبی هستم. به پردازش زبان طبیعی علاقه دارم و چون به نظرم خیلی جزابه هست'
question = 'فامیلی من چیه؟'

# tokenization
inputs =  tokenizer.encode_plus(question, context)
# convert to tensor and predicttions 
ids, token_type = torch.tensor([inputs['input_ids']]), torch.tensor([inputs['token_type_ids']])
predicts = model.forward(ids, token_type_ids=token_type)

# Find the tokens with the highest `start` and `end` scores.
start_text = int(np.argwhere(np.array(inputs['input_ids'])==tokenizer.sep_token_id)[0])
answer_start = torch.argmax(predicts['start_logits'][0][start_text:]) + start_text
after_start = torch.squeeze(predicts['end_logits'])[answer_start:]
answer_end = torch.argmax(after_start) + answer_start

# Combine the tokens and print.
answer = ' '.join(tokens[answer_start:answer_end+1])
print('Answer is: "' + answer + '"')
```
## Evaluation On ParsiNLU
- **Anybody who works in NLP knows that GLEU metrics aren't really well**
- if you not sure about that fact answer the questions and compute your f1 and Exact 😊
- <span style='color:green'>I got f1: 0.673, Exact: 0.413</span>
- But I believe that my model is better than these numbers
  - I'll show you some examples
### Some Examples that shows the F1 & Exact aren't good 
- the prediction of model is also correct !! but f1 is zero

![example_one](https://github.com/sajjjadayobi/PersianQA/blob/main/imgs/exam_1.png)
