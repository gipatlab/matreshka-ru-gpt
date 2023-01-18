import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify



np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cpu")
tok = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
app = Flask(__name__)


def generate(
    model, tok, text,
    do_sample=True,
    max_length=50,
    repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3
    ):
  input_ids = tok.encode(text, return_tensors="pt")
  out = model.generate(
      input_ids,
      max_length=max_length,
      repetition_penalty=repetition_penalty,
      do_sample=do_sample,
      top_k=top_k, top_p=top_p, temperature=temperature,
      num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
      )
  return list(map(tok.decode, out))

@app.route("/message", methods=["POST"])
def message():
  try:
    message = request.get_json()["message"]
  except:
    return jsonify({'error': "Parameter message is required."})

  try:
    do_sample = request.get_json()["do_sample"]
  except:
    do_sample = False

  try:
    max_length = request.get_json()["max_length"]
  except:
    max_length = 50

  try:
    repetition_penalty = request.get_json()["repetition_penalty"]
  except:
    repetition_penalty = 5.0

  try:
    top_k = request.get_json()["top_k"]
  except:
    top_k = 5

  try:
    top_p = request.get_json()["top_p"]
  except:
    top_p = 0.95

  try:
    temperature = request.get_json()["temperature"]
  except:
    temperature = 1

  try:
    num_beams = request.get_json()["num_beams"]
  except:
    num_beams = 10

  try:
    no_repeat_ngram_size = request.get_json()["no_repeat_ngram_size"]
  except:
    no_repeat_ngram_size = 3

  ds = False
  if do_sample == "True":
    ds = True
  else:
    ds = False

  generated = generate(
    model,
    tok,
    message,
    do_sample=ds,
    max_length=max_length,
    repetition_penalty=repetition_penalty,
    top_k=top_k,
    top_p=top_p,
    temperature=temperature,
    num_beams=num_beams,
    no_repeat_ngram_size=no_repeat_ngram_size
  )

  return jsonify({'generated': generated})


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)

