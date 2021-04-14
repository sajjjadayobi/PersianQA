from pathlib import Path
import json

def read_qa(path):
    """
    this reads dataset from JSON files like SQuAD v2
    you can use this function for loading train and test file
    """
    ds = []
    with open(Path(path), encoding="utf-8") as f:
        squad = json.load(f)
    for example in squad["data"]:
        title = example.get("title", "").strip()
        for paragraph in example["paragraphs"]:
            for qa in paragraph["qas"]:
                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                answers = [answer["text"].strip() for answer in qa["answers"]]
                ds.append({
                  "title": title,
                  "context": paragraph["context"].strip(),
                  "question": qa["question"].strip(),
                  "id": qa["id"],
                  "answers": {
                      "answer_start": answer_starts,
                      "text": answers},})
    return ds



if __name__ == "__main__":
    train_ds = read_qa('pqa_train.json')
    test_ds  = read_qa('pqa_test.json')
    
    # Example
    print(train_ds[0])
    # >>> {'title': 'شرکت فولاد مبارکه اصفهان',
    # >>> 'context': 'شرکت فولاد مبارکۀ اصفهان، بزرگ ترین واحد صنعتی خصوصی در ایران و بزرگ ترین مجتمع تولید فولاد در کشور ایران است، که در شرق شهر مبارکه قرار دارد. فولاد مبارکه هم اکنون محرک بسیاری از صنایع بالادستی و پایین دستی است. فولاد مبارکه در ۱۱ دوره جایزۀ ملی تعالی سازمانی و ۶ دوره جایزۀ شرکت دانشی در کشور رتبۀ نخست را بدست آورده است و همچنین این شرکت در سال ۱۳۹۱ برای نخستین بار به عنوان تنها شرکت ایرانی با کسب امتیاز ۶۵۴ تندیس زرین جایزۀ ملی تعالی سازمانی را از آن خود کند. شرکت فولاد مبارکۀ اصفهان در ۲۳ دی ماه ۱۳۷۱ احداث شد و اکنون بزرگ ترین واحدهای صنعتی و بزرگترین مجتمع تولید فولاد در ایران است. این شرکت در زمینی به مساحت ۳۵ کیلومتر مربع در نزدیکی شهر مبارکه و در ۷۵ کیلومتری جنوب غربی شهر اصفهان واقع شده است. مصرف آب این کارخانه در کمترین میزان خود، ۱٫۵٪ از دبی زاینده رود برابر سالانه ۲۳ میلیون متر مکعب در سال است و خود یکی از عوامل کم آبی زاینده رود شناخته می شود.',
    # >>> 'question': 'شرکت فولاد مبارکه در کجا واقع شده است',
    # >>> 'id': 1,
    # >>> 'answers': {'answer_start': [114], 'text': ['در شرق شهر مبارکه']}}
