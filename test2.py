from utils import load_model
from transformers import AutoTokenizer

t=AutoTokenizer.from_pretrained('zh_tokenizer')

model_path='checkpoints/yezi/yezi1.ckpt'
model=load_model(model_path).cpu()

t.add_special_tokens({"additional_special_tokens":['[BLANK]']})


context_output = t(['橘子[BLANK]'], return_token_type_ids=False)
choice_output = t(['很甜'], add_special_tokens=False, return_token_type_ids=False)

context = [i[1:-1] for i in context_output['input_ids']]
choice = choice_output['input_ids']

print(model.get_prob(context, choice))


# output = t('橘子很甜', add_special_tokens=False, return_token_type_ids=False, return_tensors='pt')
# seq, n = output['input_ids'], output['attention_mask'].sum(dim=1)

# print(model.nll_mc(seq, n ,1))