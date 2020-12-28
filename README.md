# PersianQA
- Transformers models for Persian(Farsi) Question Answering
- this model is not actually Persian but I use some tricks to improve it on the Persian Language 

## Installation
- install transformers pakcage for using this as simple as posible

  ```bash 
      pip install -q transformers
  ```
## How to use 

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

context = 'امروز شنبه 5 آذر تولد من است'
question = 'پنچ آذر چه مناسبتی است؟'

qa_pipeline({'context': context, 'question': question})
>>> {answer: 'تولد من'}
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
context = 'امروز شنبه 5 آذر تولد من است'
question = 'پنچ آذر چه مناسبتی است؟'

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
