import deepspeed
import warnings
import sys
import os
from src.xl_wrapper import RuGPT3XL
import torch
from src import mpu
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")
sys.path.append("ru-gpts/")
os.environ["USE_DEEPSPEED"] = "1"

# seq_len - это максимальная длина последовательности для генерации, используемая в процессе генерации.
# Максимальное доступное значение seq_len равно 2048 (в токенах). Кроме того, для вывода требуется около 10 Гб памяти графического процессора

torch.cuda.is_available()

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5000"

PROMPT = "Какие особенности у одаренных подростков? "

# gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
# model parallel group is not initialized - если не подключена gpu

# Get logits
logits = gpt(PROMPT).logits
type(logits), logits.shape

# Get loss
input_ids = [gpt.tokenizer(PROMPT)['input_ids']]
labels = input_ids

with torch.no_grad():
  loss = gpt(input_ids=input_ids, labels=labels).loss

def filter_resuls(nr):
  return [x[:x.find("<|endoftext|>")] for x in nr]


app = Flask(__name__)

@app.route("/message", methods=["POST"])
def message():
  try:
    message = request.get_json()["message"]
  except:
    return jsonify({'error': "Parameter message is required."})

  # Greedy decoding
  res = filter_resuls(gpt.generate(
    message,
    max_length=50,
    no_repeat_ngram_size=3,
    repetition_penalty=2.,
  ))

  return jsonify({'message': res})

def main():
  #Greedy decoding
  # filter_resuls(gpt.generate(
  #   PROMPT,
  #   max_length=50,
  #   no_repeat_ngram_size=3,
  #   repetition_penalty=2.,
  # ))
  #Sample
  # filter_resuls(gpt.generate(
  #   PROMPT,
  #   do_sample=True,
  #   num_return_sequences=5,
  #   max_length=50,
  #   no_repeat_ngram_size=3,
  #   repetition_penalty=2.,
  # ))

  #Top_k top_p filtering
  # filter_resuls(gpt.generate(
  #   PROMPT,
  #   top_k=5,
  #   top_p=0.95,
  #   temperature=1.2,
  #   num_return_sequences=5,
  #   do_sample=True,
  #   max_length=50,
  #   no_repeat_ngram_size=3,
  #   repetition_penalty=2.,
  # ))

  #Beamsearch
  # filter_resuls(gpt.generate(
  #   text=PROMPT,
  #   max_length=50,
  #   num_beams=10,
  #   no_repeat_ngram_size=3,
  #   repetition_penalty=2.,
  #   num_return_sequences=5,
  # ))
  app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)


if __name__ == "__main__":
  main()