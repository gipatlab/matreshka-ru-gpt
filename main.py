import deepspeed
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("ru-gpts/")
import os
os.environ["USE_DEEPSPEED"] = "1"
from src.xl_wrapper import RuGPT3XL
import torch
torch.cuda.is_available()
from src import mpu

from flask import Flask, request, jsonify

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5000"

gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)


def filter_resuls(nr):
  return [x[:x.find("<|endoftext|>")] for x in nr]

app = Flask(__name__)

@app.route("/message", methods=["POST"])
def message():
  try:
    message = request.get_json()["message"]
    res = filter_resuls(gpt.generate(
      message,
      do_sample=True,
      num_return_sequences=5,
      max_length=50,
      no_repeat_ngram_size=3,
      repetition_penalty=2.,
    ))
    jsonify({message: res})
  except:
    return jsonify({'error': "Parameter message is required."})


  return jsonify({'message': "res"})

def main():
  app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)


if __name__ == "__main__":
  main()

