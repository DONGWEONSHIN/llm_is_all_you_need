from flask import Flask, render_template, request, jsonify
import markdown


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Gemini, LangChain 설정
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# PaLM2 설정
# from langchain.embeddings import VertexAIEmbeddings

# microsoft/DialoGPT-medium 설정
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Gemini 설정
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name = "gemini-pro")

# LangChain 설정
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# LangChain 설정 - PaLM2
# embeddings = VertexAIEmbeddings()


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    
    # LangChain 설정
    result = llm.invoke(input)
    llm_result = result.content
    
    
    # PaLM2 설정
    # text_embedding = embeddings.embed_query(input)
    # llm_result = text_embedding
    
    return markdown.markdown(llm_result, extensions=['extra'])
    # return get_Chat_response(input)


def get_Chat_response(text):

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


if __name__ == '__main__':
    app.run()
