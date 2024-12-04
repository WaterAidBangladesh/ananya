import os
import chromadb
import dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from save import Save

dotenv.load_dotenv()

user_history = {}


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model="llama-3.1-8b-instant",
        )
        self.save = Save()
        self.chroma_client = chromadb.PersistentClient('vector_database')
        self.collection = self.chroma_client.get_or_create_collection(name="probahini")

    def get_response(self, message, chat_id):
        retriever = self.collection.query(
            query_texts=message,
            n_results=3
        ).get('documents')

        template = """    
        ### QUESTION FROM USER
        {user_question}
        ### ANSWER FROM VECTOR DATABASE CREATED BY CHROMADB
        {answer}
        ### INSTRUCTIONS
        You are an expert for answering menstrual period related question. Your job is to connect user question and answer and no need to repeat your question. For large response, you can use points rather than heavy answer. IF there is any follow-up question required at the bottom of the response please include it(NO PREAMBLE)
        ### DON'T ADD any QUESTION TO THE RESPONSE
        """

        prompt_template = PromptTemplate.from_template(template)
        history = user_history.get(chat_id, "")
        chain = prompt_template | self.llm
        res = chain.invoke(input={'user_question': message, 'answer': retriever, 'previous_responses': history})
        new_entry = f"user: {message}\nchatbot: {res.content}\n"
        user_history[chat_id] = history + new_entry
        self.save.save_to_csv(message, res.content)
        return res.content
