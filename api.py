from flask import Flask, request, Response
from flask_cors import CORS
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI

from src.AgentChatGPT import AgentChatGPT
from src.data.DatabaseManager import DatabaseManager
from src.data.VectorStoreManager import VectorStoreManager
from src.tools.DocumentSearch import DocumentSearch
from src.tools.ChatBot import ChatBot


load_dotenv()

app = Flask(__name__)
CORS(app)

db = DatabaseManager()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
singleDocAgent = AgentChatGPT(llm)


@app.route("/api/beta/init", methods=["GET"])
def init_single_doc():
    doc_id = request.args.get("id")
    doc = db.get(doc_id)
    docPath = db.saveToTemp([doc])
    
    docVecStore = VectorStoreManager("Chroma")
    docVecStore.load(docPath)
    
    docRetrieverTool = DocumentSearch(docVecStore, llm).asTool(topic="the company")
    chatbot = ChatBot(llm).asTool()
    singleDocAgent.init([docRetrieverTool, chatbot])
    
    return Response(status=204)


@app.route("/api/beta/prompt", methods=["POST"])
def beta_submit_prompt():
    prompt = request.data.decode('UTF-8')
    answer = singleDocAgent.run(prompt=prompt)
    return answer