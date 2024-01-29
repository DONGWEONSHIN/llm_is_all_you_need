# OS : Linux-5.15.0-91-generic-x86_64-with-glibc2.31
# Python : 3.9.18 / 3.10.13
# Flask : 3.0.0
# google-generativeai : 0.3.2
# langchain : 0.1.0
# langchain-google-genai : 0.0.6
# FileName : app.py
# Base LLM : Vertex AI Palm2 / Google AI Gemini
# Created: Jan. 29. 2024
# Author: D.W. SHIN


import os

from flask import Flask, jsonify, render_template, request
from markdown import markdown

from model_Gemini import Gemini
from model_Palm2 import Palm2

# pdf 저장폴더
PDF_DN_FOLDER = "./PDF_DN_FOLDER"


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/chatMuseum")
def chatMuseum():
    return render_template("museum.html")


@app.route("/chatDiffusion")
def chatDiffusion():
    return render_template("diffusion.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    # MODEL_TYPE = "PALM2"
    # MODEL_TYPE = "GEMINI"
    MODEL_TYPE = request.form["model_type"]

    query = request.form["msg"]

    result = ""

    if MODEL_TYPE == "PALM2":
        PROJECT_ID = os.environ.get("PROJECT_ID")
        LOCATION = os.environ.get("LOCATION")

        p_llm = Palm2(
            project_id=PROJECT_ID,
            location=LOCATION,
        )

        result = p_llm.chat(msg=query)

    elif MODEL_TYPE == "GEMINI":
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

        g_llm = Gemini(google_api_key=GOOGLE_API_KEY)

        result = g_llm.chat(msg=query)

    return result


@app.route("/savePdf", methods=["POST"])
def savePdf():
    result = {
        "RETURN_FLAG": "LOADED",
        "RETURN_INFO": "",
    }
    os.makedirs(PDF_DN_FOLDER, exist_ok=True)

    file = request.files["file"]
    fullFilename = str(file.filename)
    fname, fextension = os.path.splitext(fullFilename)
    if fextension != ".pdf":
        result = {
            "RETURN_FLAG": "FAIL",
            "RETURN_INFO": "IT IS NOT A PDF FILE.",
        }
    else:
        fileFullPath = os.path.join(PDF_DN_FOLDER, fullFilename)
        file.save(fileFullPath)
        result = {
            "RETURN_FLAG": "SUCCESS",
            "RETURN_INFO": "THE FILE WAS SAVED SUCCESSFULLY.",
        }

    return result


@app.route("/chatWithPdf", methods=["GET", "POST"])
def chatWithPdf():
    MODEL_TYPE = request.form["model_type"]

    MSG = request.form["msg"]
    FULLFILENAME = request.form["filename"]

    result = ""

    if MODEL_TYPE == "PALM2":
        PROJECT_ID = os.environ.get("PROJECT_ID")
        LOCATION = os.environ.get("LOCATION")

        p_llm = Palm2(
            project_id=PROJECT_ID,
            location=LOCATION,
        )

        result = p_llm.chatWithPdf(
            msg=MSG,
            fullFilename=FULLFILENAME,
            pdf_dn_folder=PDF_DN_FOLDER,
        )

    elif MODEL_TYPE == "GEMINI":
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

        g_llm = Gemini(google_api_key=GOOGLE_API_KEY)

        result = g_llm.chatWithPdf(
            msg=MSG,
            fullFilename=FULLFILENAME,
            pdf_dn_folder=PDF_DN_FOLDER,
        )

    return result


if __name__ == "__main__":
    app.run()
