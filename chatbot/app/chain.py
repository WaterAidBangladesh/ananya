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
        self.chroma_client = chromadb.PersistentClient('vectordb')
        self.collection = self.chroma_client.get_or_create_collection(name="probahini")

    def get_response(self, message, chat_id):
        retriever = self.collection.query(
            query_texts=message,
            n_results=3
        ).get('documents')
        template = """ Relevant information: {answer}

        Background: You are an expert in menstrual health topics, structured to provide information based on both 
        high-level (prime) and specific (follow-up) questions. If the user message aligns with a general or 
        overarching question, respond with the prime answer and, in rare occasions, suggest a couple of follow-up 
        questions below it. If the question seeks specific details, provide the relevant follow-up answer. In cases 
        where multiple relevant details exist, respond concisely with the most applicable information. You are 
        empathetic and considerate, communicating in English or Bangla based on the user's language preference. 
        If you detect a language preference from the user's message, respond accordingly. Engage in conversational 
        interactions, and for questions, provide specific, accurate answers based on the relevant information below. 
        Please don't share any of the question labels; only deliver the content of the answer.

        Note: You must *only* provide answers from the exact information provided in the "Relevant information" 
        above. If no relevant information exists, refer to the "Flow of Chat" for context to create an informed and 
        relevant response.
        ### IF USER QUERIES IN BANGLA RESPONSE GIVE IN BANGLA ELSE ENGLISH ###

        Flow of Chat: {previous_responses}

        User message: {user_question}

        (NO PREAMBLE)
        """

        prompt_template = PromptTemplate.from_template(template)
        history = user_history.get(chat_id, "")
        chain = prompt_template | self.llm
        res = chain.invoke(input={'user_question': message, 'answer': retriever, 'previous_responses': history})
        new_entry = f"user: {message}\nchatbot: {res.content}\n"
        user_history[chat_id] = history + new_entry
        self.save.save_to_csv(message, res.content)
        return res.content
