import torch
import pytorch_lightning as pl
import streamlit as st
import os
from utils import load_model
from transformers import BertTokenizer

model_path='checkpoints/yezi/yezi1.ckpt'
tokenizer_path='zh_tokenizer'

@st.cache()
def load_model_and_tokenizer():
    global model_path
    model=load_model(model_path).cpu()
    tokenizer=BertTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"additional_special_tokens":['[BLANK]']})
    return model, tokenizer

st.write('# Blank Language Models: 椰子评论生成')

decode = st.radio("Decoding", ("Greedy", "Sample")).lower()
device = st.radio("Device", ("CPU", )).lower()

model, tokenizer=load_model_and_tokenizer()

st.write('## Load infilling text')
text_input = st.text_input("Blanked input", value="___ 质感 ___ 不错 ___ 保湿 ___ .")
s = text_input.replace("___", "[BLANK]")
s = tokenizer(s, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)['input_ids']
print(s)
_, full = model.generate(s, decode, device)

print(full)
full = [tokenizer.convert_ids_to_tokens(ids) for ids in full]
for step in full:
    st.write(" ".join(step).replace("<blank>", "\_\_\_"))

if st.button("Rerun"):
  pass