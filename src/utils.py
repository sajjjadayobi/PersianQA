from tqdm import tqdm
import torch


class AnswerPredictor:
  def __init__(self, model, tokenizer, device='cuda', n_best=10,
               max_length=512, stride=256, no_answer=False):
    
      """
      Question Anwsering Prediction
      Args:
        model: 
        tokenizer:
        device:
        n_best:
        max_length:
        stride: 
        no_answer:
      """
      self.model = model.eval().to(device)
      self.tokenizer = tokenizer
      self.device = device
      self.max_length = max_length
      self.stride = stride
      self.no_answer = no_answer
      self.n_best = n_best


  def model_pred(self, questions, contexts, batch_size=1):
      n = len(contexts)
      if n%batch_size!=0:
          raise Exception("batch_size must be dividble by sample len")

      tokens = self.tokenizer(questions, contexts, add_special_tokens=True, 
                              return_token_type_ids=True, return_tensors="pt", padding=True, 
                              return_offsets_mapping=True, truncation="only_second", 
                              max_length=self.max_length, stride=self.stride)


      start_logits, end_logits = [], []
      for i in tqdm(range(0, n-batch_size+1, batch_size)):
          with torch.no_grad():
              out = self.model(tokens['input_ids'][i:i+batch_size].to(self.device), 
                          tokens['attention_mask'][i:i+batch_size].to(self.device), 
                          tokens['token_type_ids'][i:i+batch_size].to(self.device))

              start_logits.append(out.start_logits)
              end_logits.append(out.end_logits)

      return tokens, torch.stack(start_logits).view(n, -1), torch.stack(end_logits).view(n, -1)



  def __call__(self, questions, contexts, batch_size=1, answer_max_len=100):
      """ create model prediction 
      Args:
        questions:
        contexts:
        batch_size: 
        answer_max_len:
        
      Returns:
        predictions: best model predictions (dict)
            {0: {"text": str, "score": int}}
      """

      tokens, starts, ends = self.model_pred(questions, contexts, batch_size=batch_size)
      print()
      start_indexes = starts.argsort(dim=-1, descending=True)[:, :self.n_best]
      end_indexes = ends.argsort(dim=-1, descending=True)[:, :self.n_best]

      predictions = {}
      for i, (c, q) in enumerate(zip(contexts, questions)):  
          min_null_score = starts[i][0] + ends[i][0] # 0 is CLS Token
          start_context = tokens['input_ids'][i].tolist().index(self.tokenizer.sep_token_id)
          
          offset = tokens['offset_mapping'][i]
          valid_answers = []
          for start_index in start_indexes[i]:
              # Don't consider answers that are in question
              if start_index<start_context:
                  continue
              for end_index in end_indexes[i]:
                  # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                  # to part of the input_ids that are not in the context.
                  if (start_index >= len(offset) or end_index >= len(offset)
                      or offset[start_index] is None or offset[end_index] is None):
                      continue
                  # Don't consider answers with a length that is either < 0 or > max_answer_length.
                  if end_index < start_index or (end_index-start_index+1) > answer_max_len:
                      continue

                  start_char = offset[start_index][0]
                  end_char = offset[end_index][1]
                  valid_answers.append({"score": (starts[i][start_index] + ends[i][end_index]).item(),
                                        "text": c[start_char: end_char]})
                  
          if len(valid_answers) > 0:
              best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
          else:
              best_answer = {"text": "", "score": min_null_score}

          if self.no_answer:
              predictions[i] = best_answer if best_answer["score"] >= min_null_score else {"text": "", "score": min_null_score}
          else:
              predictions[i] = best_answer

      return predictions
    
    

class TFAnswerPredictor:
  pass
