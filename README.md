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
model = TFBertForQuestionAnswering.from_pretrained(model_name).eval()
config = AutoConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
```

### Pytorch

```python
from transformers import AutoConfig, BertTokenizer, BertForQuestionAnswering

model_name = 'SajjadAyoubi/bert-base-fa-qa'
model = BertForQuestionAnswering.from_pretrained(model_name).eval()
config = AutoConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
```

## Examples

### Pipeline 
```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

context = 'امروز شنبه 5 آذر تولد من است'
question = 'پنچ آذر چه مناسبتی است؟'

qa_pipeline({'context': context, 'question': question})
>>> {answer: 'تولد من'}
```

### Manually 
