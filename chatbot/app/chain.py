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
        # gpt-oss reasons privately before it answers, and both come out of one
        # output allowance. Two measurements on the same Bangla question, which
        # carries ~18,000 tokens of retrieved text:
        #
        #   max_tokens 2048 -> 2046 reasoning tokens, finish_reason "length"
        #   max_tokens 8192 -> 8190 reasoning tokens, finish_reason "length"
        #
        # Both returned an empty answer. Raising the ceiling only bought more
        # reasoning: faced with three whole PDF pages of largely unrelated
        # text, the model will think for as long as it is allowed and never
        # reach the answer. English questions retrieve less and fit, which is
        # why only Bangla looked broken.
        #
        # So the limit is not the lever — the amount of thinking is.
        # reasoning_effort caps that directly.
        llm_kwargs = dict(
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model="openai/gpt-oss-20b",
            max_tokens=8192,
        )
        try:
            # Named argument rather than model_kwargs: if this version of
            # langchain-groq knows the field it binds directly, and if it does
            # not, ChatGroq folds unknown kwargs into model_kwargs and Groq
            # receives it anyway.
            self.llm = ChatGroq(**llm_kwargs, reasoning_effort="low")
        except Exception:
            # Never let an unsupported parameter stop the service from
            # starting. Without this the container would crash on import and
            # take the whole chatbot down, which is a far worse failure than
            # the one being fixed.
            self.llm = ChatGroq(**llm_kwargs)
        self.save = Save()
        self.chroma_client = chromadb.PersistentClient('vectordb')
        self.collection = self.chroma_client.get_or_create_collection(name="probahini")

    def get_response(self, message, chat_id):
        # Six, not three. Each entry used to be an entire PDF page, so three
        # of them was already far more text than any answer needed. The store
        # is now built by build_vectordb.py in chunks of about a row or two,
        # so six of those carry more genuinely relevant material than three
        # pages did, at a fraction of the size — and small chunks make
        # retrieval less forgiving, which a wider net offsets.
        retriever = self.collection.query(
            query_texts=message,
            n_results=6
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
