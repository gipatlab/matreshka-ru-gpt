import numpy as np
import torch
from transformers import (GPT2LMHeadModel, GPT2Tokenizer)
from datetime import datetime
from flask import Flask, request, jsonify

device = torch.device("cpu")
model_type = 'gpt2'
model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
tokenizer = tokenizer_class.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
app = Flask(__name__)

def set_seed(seed=datetime.now().microsecond):
  np.random.seed(seed)
  torch.manual_seed(seed)
  print("random seed is " + str(seed))
  if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(seed)


set_seed()
model = model_class.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model.to(device)

def cut_extra_stuff(txt):
  """Вырезает артефакты"""
  extra = txt.find('\n\n\n')
  return txt[0:extra] if extra != -1 else txt

@app.route("/message", methods=["POST"])
def message():
  try:
    message = request.get_json()["message"]
  except:
    return jsonify({'error': "Parameter message is required."})

  encoded_prompt = tokenizer.encode(message, add_special_tokens=False, return_tensors="pt")
  encoded_prompt = encoded_prompt.to(device)

  output_sequences = model.generate(
    input_ids=encoded_prompt,
    max_length=50,
    no_repeat_ngram_size=3,
    repetition_penalty=2.,
  )

  if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()

  stop_token = "</s>"

  res = ""

  for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    generated_sequence = generated_sequence.tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[:text.find(stop_token) if stop_token else None]
    total_sequence = (
      text[len(
        tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):].rsplit(' ', 1)[0])

    total_sequence = cut_extra_stuff(total_sequence)
    res = total_sequence

  return jsonify({'generated': res})


app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)

