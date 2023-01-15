import deepspeed.ops.sparse_attention.sparse_attn_op
import deepspeed
import warnings
import sys
import os
from src.xl_wrapper import RuGPT3XL
# Note! seq_len is max sequence length for generation used in generation process. Max avialable seq_len is 2048 (in tokens). Also inference takes around 10 Gb GPU memory.
import torch
from src import mpu


warnings.filterwarnings("ignore")
sys.path.append("ru-gpts/")
os.environ["USE_DEEPSPEED"] = "1"
torch.cuda.is_available()

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5000"

# gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
# model parallel group is not initialized - если не подключена gpu
logits = gpt("Какие особенности у одаренных подростков? ").logits
type(logits), logits.shape
input_ids = [gpt.tokenizer("Какие особенности у одаренных подростков? ")['input_ids']]
labels = input_ids
with torch.no_grad():
    loss = gpt(input_ids=input_ids, labels=labels).loss
    print(loss)

def filter_resuls(nr):
    return [x[:x.find("<|endoftext|>")] for x in nr]

def main():
    filter_resuls(gpt.generate(
        "Какие особенности у одаренных подростков? ",
        max_length=50,
        no_repeat_ngram_size=3,
        repetition_penalty=2.,
    ))

if __name__ == "__main__":
    main()