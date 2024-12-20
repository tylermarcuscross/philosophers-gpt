# %%
import tokenize, ast
from io import BytesIO
from tiktoken import encoding_for_model
from openai import ChatCompletion,Completion
enc = encoding_for_model("text-davinci-003")
toks = enc.encode("They are splashing")
toks

# %%
[enc.decode_single_token_bytes(o).decode('utf-8') for o in toks]


# %%
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

# %%
completion.choices[0].message

# %%
aussie_sys = "You are an Aussie LLM that uses Aussie slang and analogies whenever possible."

c = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": aussie_sys},
              {"role": "user", "content": "What is money?"}])

# %% [markdown]
# - [Model options](https://platform.openai.com/docs/models)

# %%
c.choices[0].message.content
#c['choices'][0]['message']['content']

# %%
from fastcore.utils import nested_idx

# %%
def response(compl): print(compl.choices[0].message.content)

# %%
response(c)

# %%
print(c.usage)

# %%
0.002 / 1000 * 150 # GPT 3.5

# %%
0.03 / 1000 * 150 # GPT 4

# %%
c = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": aussie_sys},
              {"role": "user", "content": "What is money?"},
              {"role": "assistant", "content": "Well, mate, money is like kangaroos actually."},
              {"role": "user", "content": "Really? In what way?"}])

# %%
response(c)

# %%
def askgpt(user, system=None, model="gpt-3.5-turbo", **kwargs):
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return client.chat.completions.create(model=model, messages=msgs, **kwargs)

# %%
response(askgpt('What is the meaning of life?', system=aussie_sys))

# %% [markdown]
# - [Limits](https://platform.openai.com/docs/guides/rate-limits/what-are-the-rate-limits-for-our-api)
# 
# Created by Bing:

# %%
def call_api(prompt, model="gpt-3.5-turbo"):
    msgs = [{"role": "user", "content": prompt}]
    try: return client.chat.completions.create(model=model, messages=msgs)
    except openai.error.RateLimitError as e:
        retry_after = int(e.headers.get("retry-after", 60))
        print(f"Rate limit exceeded, waiting for {retry_after} seconds...")
        time.sleep(retry_after)
        return call_api(params, model=model)

# %%
call_api("What's the world's funniest joke? Has there ever been any scientific analysis?")

# %%
c = Completion.create(prompt="Australian Jeremy Howard is ",
                      model="gpt-3.5-turbo-instruct", echo=True, logprobs=5)

# %% [markdown]
# ### Create our own code interpreter

# %%
from pydantic import create_model
import inspect, json
from inspect import Parameter

# %%
def sums(a:int, b:int=1):
    "Adds a + b"
    return a + b

# %%
def schema(f):
    kw = {n:(o.annotation, ... if o.default==Parameter.empty else o.default)
          for n,o in inspect.signature(f).parameters.items()}
    s = create_model(f'Input for `{f.__name__}`', **kw).schema()
    return dict(name=f.__name__, description=f.__doc__, parameters=s)

# %%
schema(sums)

# %%
c = askgpt("Use the `sum` function to solve this: What is 6+3?",
           system = "You must use the `sum` function instead of adding yourself.",
           functions=[schema(sums)])

# %%
m = c.choices[0].message
m

# %%
k = m.function_call.arguments
print(k)

# %%
funcs_ok = {'sums', 'python'}

# %%
def call_func(c):
    fc = c.choices[0].message.function_call
    if fc.name not in funcs_ok: return print(f'Not allowed: {fc.name}')
    f = globals()[fc.name]
    return f(**json.loads(fc.arguments))

# %%
call_func(c)

# %%
def run(code):
    tree = ast.parse(code)
    last_node = tree.body[-1] if tree.body else None
    
    # If the last node is an expression, modify the AST to capture the result
    if isinstance(last_node, ast.Expr):
        tgts = [ast.Name(id='_result', ctx=ast.Store())]
        assign = ast.Assign(targets=tgts, value=last_node.value)
        tree.body[-1] = ast.fix_missing_locations(assign)

    ns = {}
    exec(compile(tree, filename='<ast>', mode='exec'), ns)
    return ns.get('_result', None)

# %%
run("""
a=1
b=2
a+b
""")

# %%
def python(code:str):
    "Return result of executing `code` using python. If execution not permitted, returns `#FAIL#`"
    go = input(f'Proceed with execution?\n```\n{code}\n```\n')
    if go.lower()!='y': return '#FAIL#'
    return run(code)

# %%
c = askgpt("What is 12 factorial?",
           system = "Use python for any required computations.",
           functions=[schema(python)])

# %%
call_func(c)

# %%
c = client.chat.completions.create(
    model="gpt-3.5-turbo",
    functions=[schema(python)],
    messages=[{"role": "user", "content": "What is 12 factorial?"},
              {"role": "function", "name": "python", "content": "479001600"}])

# %%
response(c)

# %%
c = askgpt("What is the capital of France?",
           system = "Use python for any required computations.",
           functions=[schema(python)])

# %%
response(c)


# %%
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

# %%
model = AutoModel.from_pretrained("bert-base-cased")

# %% [markdown]
# - [HF leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
# - [fasteval](https://fasteval.github.io/FastEval/)

# %%
mn = "bert-base-cased"

# %%
model = AutoModelForCausalLM.from_pretrained(mn, device_map=0, load_in_8bit=True)

# %%
tokr = AutoTokenizer.from_pretrained(mn)
prompt = "Jeremy Howard is a "
toks = tokr(prompt, return_tensors="pt")

# %%
toks

# %%
tokr.batch_decode(toks['input_ids'])

# %%
%%time
res = model.generate(**toks.to("cuda"), max_new_tokens=15).to('cpu')
res

# %%
tokr.batch_decode(res)

# %%
model = AutoModelForCausalLM.from_pretrained(mn, device_map=0, torch_dtype=torch.bfloat16)

# %%
%%time
res = model.generate(**toks.to("cuda"), max_new_tokens=15).to('cpu')
res

# %%
model = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7b-Chat-GPTQ', device_map=0, torch_dtype=torch.float16)

# %%
%%time
res = model.generate(**toks.to("cuda"), max_new_tokens=15).to('cpu')
res

# %%
mn = 'TheBloke/Llama-2-13B-GPTQ'
model = AutoModelForCausalLM.from_pretrained(mn, device_map=0, torch_dtype=torch.float16)

# %%
%%time
res = model.generate(**toks.to("cuda"), max_new_tokens=15).to('cpu')
res

# %%
def gen(p, maxlen=15, sample=True):
    toks = tokr(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample).to('cpu')
    return tokr.batch_decode(res)

# %%
gen(prompt, 50)

# %% [markdown]
# [StableBeluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B)

# %%
mn = "stabilityai/StableBeluga-7B"
model = AutoModelForCausalLM.from_pretrained(mn, device_map=0, torch_dtype=torch.bfloat16)

# %%
sb_sys = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can.\n\n"

# %%
def mk_prompt(user, syst=sb_sys): return f"{syst}### User: {user}\n\n### Assistant:\n"

# %%
ques = "Who is Jeremy Howard?"

# %%
gen(mk_prompt(ques), 150)

# %% [markdown]
# [OpenOrca/Platypus 2](https://huggingface.co/Open-Orca/OpenOrca-Platypus2-13B)

# %%
mn = 'TheBloke/OpenOrca-Platypus2-13B-GPTQ'
model = AutoModelForCausalLM.from_pretrained(mn, device_map=0, torch_dtype=torch.float16)

# %%
def mk_oo_prompt(user): return f"### Instruction: {user}\n\n### Response:\n"

# %%
gen(mk_oo_prompt(ques), 150)

# %% [markdown]
# ### Retrieval augmented generation

# %%
from wikipediaapi import Wikipedia

# %%
wiki = Wikipedia('JeremyHowardBot/0.0', 'en')
jh_page = wiki.page('Jeremy_Howard_(entrepreneur)').text
jh_page = jh_page.split('\nReferences\n')[0]

# %%
print(jh_page[:500])

# %%
len(jh_page.split())

# %%
ques_ctx = f"""Answer the question with the help of the provided context.

## Context

{jh_page}

## Question

{ques}"""

# %%
res = gen(mk_prompt(ques_ctx), 300)

# %%
print(res[0].split('### Assistant:\n')[1])

# %%
from sentence_transformers import SentenceTransformer

# %%
emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=0)

# %%
jh = jh_page.split('\n\n')[0]
print(jh)

# %%
tb_page = wiki.page('Tony_Blair').text.split('\nReferences\n')[0]

# %%
tb = tb_page.split('\n\n')[0]
print(tb[:380])

# %%
q_emb,jh_emb,tb_emb = emb_model.encode([ques,jh,tb], convert_to_tensor=True)

# %%
tb_emb.shape

# %%
import torch.nn.functional as F

# %%
F.cosine_similarity(q_emb, jh_emb, dim=0)

# %%
F.cosine_similarity(q_emb, tb_emb, dim=0)

# %% [markdown]
# ### Private GPTs

# %% [markdown]
# ## Fine tuning

# %%
import datasets

# %% [markdown]
# [knowrohit07/know_sql](https://huggingface.co/datasets/knowrohit07/know_sql)

# %%
ds = datasets.load_dataset('tylercross/platos_socrates_no_context')

# %%
ds

# %%
trn = ds['train']
trn[3]

# %% [markdown]
# `accelerate launch -m axolotl.cli.train sql.yml`

# %%
tst = dict(**trn[3])
tst['instruction'] = "Respond as Socrates would given the preceeding text"
tst

# %%
fmt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

# %%
def philosopher_prompt(d): return fmt.format(instruction=d["instruction"], input=d["input"])

# %%
print(philosopher_prompt(tst))

# %%
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# %%
ax_model = r'C:\Users\tyler\Desktop\qlora-out'

# %%
tokr = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

# %%
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1',
                                             torch_dtype=torch.bfloat16, device_map=0)
model = PeftModel.from_pretrained(model, ax_model)
model = model.merge_and_unload()
model.save_pretrained('mistral_socrates-model')

# %%
toks = tokr(philosopher_prompt(tst), return_tensors="pt")

# %%
res = model.generate(**toks.to("cuda"), max_new_tokens=250).to('cpu')

# %%
print(tokr.batch_decode(res)[0])

# %% [markdown]
# ## [llama.cpp](https://github.com/abetlen/llama-cpp-python)

# %% [markdown]
# [TheBloke/Llama-2-7b-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF)

# %%
from llama_cpp import Llama

# %%
llm = Llama(model_path="/home/jhoward/git/llamacpp/llama-2-7b-chat.Q4_K_M.gguf")

# %%
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)

# %%
print(output['choices'])

# %% [markdown]
# ## [MLC](https://mlc.ai/mlc-llm/docs/get_started/try_out.html#get-started)


