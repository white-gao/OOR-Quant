from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# model_name = "OpenOneRec/OneRec-1.7B" # 用model ID会卡死，原因不明
model_name = "/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b/"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

# prepare the model input
# case - prompt with itemic tokens
prompt = "这是一个视频：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，帮我总结一下这个视频讲述了什么内容"
# prompt = "请给我推荐2000块钱价位的手机，并说明理由。"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
# Note: In our experience, default decoding settings may be unstable for small models.
# For 1.7B, we suggest: top_p=0.95, top_k=20, temperature=0.75 (during 0.6 to 0.8)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    top_p=0.95,
    top_k=20,
    temperature=0.75
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)