from collections import OrderedDict
from pathlib import Path


def read_ds(path):
    """
    this reads dataset from json files like SQuAD v2
    """
    path = Path(path)
    ds = []
    with open(path, encoding="utf-8") as f:
        squad = json.load(f)
    for example in squad["data"]:
        title = example.get("title", "").strip()
        for paragraph in example["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                question = qa["question"].strip()
                id_ = qa["id"]
                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                answers = [answer["text"].strip() for answer in qa["answers"]]
                ds.append({
                  "title": title,
                  "context": context,
                  "question": question,
                  "id": id_,
                  "answers": {
                      "answer_start": answer_starts,
                      "text": answers},})
    return ds
  

def convert2order(ds):
    """
    convert ds object to an OrderDict 
    """
    return OrderedDict([('answers', [i['answers'] for i in ds]), 
                      ('context', [i['context'] for i in ds]), 
                      ('question', [i['question'] for i in ds])])


def combine(ds1, ds2):
    """
    for combinig PersianQA with SQuAD v2
    """
    return OrderedDict([('answers', ds1['answers']+ds2['answers']), 
                      ('context', ds1['context']+ds2['context']), 
                      ('question', ds1['question']+ds2['question'])])

  
# example  
if __name__ == "__main__":
    train_ds = convert2order(read_ds('PersianQA-train.json'))
    valid_ds = convert2order(read_ds('PersianQA-valid.json'))
    
    # if you want use PersianQA + SQuAD v2
    squad_train_ds = convert2order(read_ds('train-v2.0.json'))
    train_ds = combine(squad_train_ds, train_ds)
