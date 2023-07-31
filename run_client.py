from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from gradio.components import Textbox, Number, Checkbox
import gradio as gr

model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "gptq_model-4bit-128g"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.75,
    top_p=0.95,
    repetition_penalty=1.15
)


def chat(prompt, max_new_tokens):
    response = pipe(prompt, max_new_tokens=max_new_tokens)[0]['generated_text']
    return response

iface = gr.Interface(fn=chat, 
                    inputs=[Textbox(lines=5, label='Prompt'),
                            Number(value=4000, label="max_new_tokens")],
                    outputs=Textbox(label='Response'))

iface.launch(share=True)
