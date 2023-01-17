import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify


app = Flask(__name__)

np.random.seed(42)
torch.manual_seed(42)

def load_tokenizer_and_model(model_name_or_path):
  return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()

def generate(
    model, tok, text,
    do_sample=True, max_length=50, repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3
    ):
  input_ids = tok.encode(text, return_tensors="pt").cuda()
  out = model.generate(
      input_ids.cuda(),
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
    tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3large_based_on_gpt2")
    generated = generate(model, tok, message, num_beams=10)

    jsonify({generated: generated})
  except:
    return jsonify({'error': "Parameter message is required."})


  return jsonify({'message': "res"})

def main():
  app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)


if __name__ == "__main__":
  main()

