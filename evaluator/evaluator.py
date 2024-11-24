import os
from typing import List, Dict
from trulens.apps.custom import TruCustomApp
from trulens.core import TruSession
from trulens.core import Feedback
from trulens.providers.openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from trulens.apps.custom import instrument
from pinecone.grpc import PineconeGRPC as Pinecone
import numpy as np
from trulens.core import Select
from langchain_google_genai import ChatGoogleGenerativeAI
from trulens.providers.langchain import Langchain
from langchain_community.llms import OpenAI
from trulens.dashboard import run_dashboard
load_dotenv()





class RAG_eval:
    """A RAG Evaluation class utilizing trulens

    Args:
        retriever (Retriever): Retriever object. It returns a list of similarity match chunks through a get_Chunks method
        generator (Generator): Generator object. It returns a string of answer generated with an llm through a generate method
        reset_database (Boolean): specifies wether a user wants to restet trulens session database 
        cos (Boolean): specifies wether a user wants to use Chain of Thought Reasons generated with LLM 
        provider_name (Enum(Gemini, OpenAI)): Specifies which provider a user would like to use Gemini is the default 
        version (string)" enables users to work on multiple versions of the same app and document the differences between them.

    Attributes:
        Private
    Methods:
        run(questions: [string]): runs the evaluation on a list of questions .
        dashboard(): Spens up a dhashboard based on trulens database 
    """
    
    def __init__(self,retriever,generator , reset_database = False, cos = True, provider_name="Gemini", version = "v1.0"):
        # self.db_name = db_name
        self.reset_database = reset_database
        self.version = version
        self.retriever = retriever # Retriever object
        self.generator = generator  # Generator object
        self.session = self._initiate_session()
        # self.index = self._initiate_db()
        self.cos = cos   # For chain of thought evaluation. More costly 
        self.provider_name = provider_name
        self.provider = self._initiate_provider()
        self.feedback = self._initiate_feedback()
        self.app = self._initiate_app()
        
        
    def _initiate_session( self):
        session = TruSession()
        if(self.reset_database):
            session.reset_database()
        
        return session
    
    # def _initiate_db(self):
    #     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    #     index = pc.Index(self.db_name)
    #     return index

    def _initiate_app(self):
        return TruCustomApp(self,app_name=f"RAG-{self.provider_name}-{self.db_name}-{"Cos" if self.cos else""}",
            app_version=self.version,
            feedbacks=self.feedback,
            )

    # Instrumented Retrieval methods
    @instrument
    def retrieve(self, query: str) -> List[str]:
        """
        Method to handle document retrieval.
        IMPORTANT: The method name 'retrieve' will be used in selectors
        """
        chunks = self.retriever.get_Chunks(query)
        return chunks
    
    @instrument
    def generate(self, query: str, context: List[str]) -> str:
        """
        Method to handle response generation.
        IMPORTANT: The method name 'generate' will be used in selectors
        """
        formatted_context = "\n".join([str(doc) for doc in context])
        response = self.generator.generate(query, formatted_context)
        return response
    
    @instrument
    def query(self, question: str) -> Dict:
        """
        Main method that orchestrates the RAG pipeline.
        IMPORTANT: Return keys must match selector paths
        """
        context = self.retrieve(question)
        response = self.generate(question, context)
        
        return response

    def _initiate_feedback(self):
        #Adhere to trulens version 1.2.6 https://www.trulens.org/reference/trulens/providers/openai/#trulens.providers.openai.OpenAI
        if self.cos :
            f_groundedness = (
                Feedback(
                    self.provider.groundedness_measure_with_cot_reasons, name="Groundedness-COS"
                )
                .on(Select.RecordCalls.retrieve.rets.collect())
                .on_output()
            )

            f_answer_relevance = (
                Feedback(self.provider.relevance_with_cot_reasons, name="Answer Relevance-COS")
                .on_input()
                .on_output()
            )

            f_context_relevance = (
                Feedback( self.provider.context_relevance_with_cot_reasons, name="Context Relevance-COS").on_input()
                .on(Select.RecordCalls.retrieve.rets[:])
                .aggregate(np.mean)  # choose a different aggregation method if you wish
            )

        else:
            #In case Chain of thought reasons were not asked to be provided  (cos= false)
            f_groundedness = (
                Feedback(
                    self.provider.groundedness_measure_with_cot_reasons, name="Groundedness - COS"
                )
                .on(Select.RecordCalls.retrieve.rets.collect())
                .on_output()
            )

            f_answer_relevance = (
                Feedback(self.provider.relevance, name="Answer Relevance")
                .on_input()
                .on_output()
            )

            f_context_relevance = (
                Feedback( self.provider.context_relevance, name="Context Relevance").on_input()
                .on(Select.RecordCalls.retrieve.rets[:])
                .aggregate(np.mean)  # choose a different aggregation method if you wish
            )

        return [f_groundedness, f_answer_relevance, f_context_relevance]

    def _initiate_provider(self):
        if self.provider == "Gemini":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002",api_key=os.getenv("GEMINI_API_KEY"))
            provider = Langchain(chain = llm)
            return provider
        elif self.provider == "OpenAI":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        

    def run(self, questions):
        with self.app as recording:
            for eval in questions:
                self.query(eval)
        
    def dashboard(self):
        run_dashboard(self.session)







    





        