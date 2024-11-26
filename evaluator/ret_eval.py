import pandas as pd
from trulens.core import TruSession
from trulens.core import Feedback
from trulens.core.schema.select import Select
from trulens.feedback import GroundTruthAgreement
from trulens.providers.openai import OpenAI as fOpenAI
from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument
from trulens.dashboard import run_dashboard
from utils.chunk_scorer import score_chunk



class retriever_evaluator:
    """
    
    """

    def __init__(self,name, ground_truth, rag_app , reset_db = False):
        self.name = name
        self.rag_app = rag_app
        self.session = self._init_db(reset_db)
        self.ground_truth = self._init_ground_truth(ground_truth) 
        self.feedback = self._feedback_init()
        self.tru_app = self._init_app()

### Move the addition of the scores  to prepare ground truth 
    def _init_ground_truth(self,ground_truth):
        for i in range(len(ground_truth["query"])):
            queries =  ground_truth["query"]
            expected_responses =  ground_truth["expected_response"]
            expected_chunks = ground_truth["expected_chunks"]
            expected_chunks[i] = [{"text":expected_chunk, "title":expected_chunk, "expected_score":score_chunk(expected_chunk,expected_responses[i])} for expected_chunk in expected_chunks[i] ]
            df={"query":[queries[i]],"expected_response":[expected_responses[i]],"expected_chunks":[expected_chunks[i]],"query_id":[str(i+1)]}
            self.session.add_ground_truth_to_dataset(
                dataset_name="groundtruth",
                ground_truth_df=pd.DataFrame(df),
                dataset_metadata={"domain": "Data from Ministry of Health UAE"},)

        return self.session.get_ground_truth("groundtruth")

    def _init_db(self, reset_db):
        session = TruSession()
        session.reset_database() if reset_db else None

        return session
    
    def _feedback_init(self):
        arg_query_selector = (
            Select.RecordCalls.retrieve_and_generate.args.query
        )  # 1st argument of retrieve_and_generate function
        arg_retrieval_k_selector = (
            Select.RecordCalls.retrieve_and_generate.args.k
        )  # 2nd argument of retrieve_and_generate function

        arg_completion_str_selector = Select.RecordCalls.retrieve_and_generate.rets[
            0
        ]  # 1st returned value from retrieve_and_generate function
        arg_retrieved_context_selector = Select.RecordCalls.retrieve_and_generate.rets[
            1
        ]  # 2nd returned value from retrieve_and_generate function
        arg_relevance_scores_selector = Select.RecordCalls.retrieve_and_generate.rets[
            2
        ]  # last returned value from retrieve_and_generate function

        f_ir_hit_rate = (
            Feedback(
                GroundTruthAgreement(self.ground_truth, provider=fOpenAI()).ir_hit_rate,
                name="IR hit rate",
            )
            .on(arg_query_selector)
            .on(arg_retrieved_context_selector)
            .on(arg_retrieval_k_selector)
        )

        f_ndcg_at_k = (
            Feedback(
                GroundTruthAgreement(self.ground_truth, provider=fOpenAI()).ndcg_at_k,
                name="NDCG@k",
            )
            .on(arg_query_selector)
            .on(arg_retrieved_context_selector)
            .on(arg_relevance_scores_selector)
            .on(arg_retrieval_k_selector)
        )


        f_recall_at_k = (
                Feedback(
                GroundTruthAgreement(self.ground_truth, provider=fOpenAI()).recall_at_k,
                name="Recall@k",
            )
            .on(arg_query_selector)
            .on(arg_retrieved_context_selector)
            .on(arg_relevance_scores_selector)
            .on(arg_retrieval_k_selector)
        )
        f_groundtruth_answer = (
            Feedback(
            GroundTruthAgreement(self.ground_truth).agreement_measure,
            name="Ground Truth answer (semantic similarity)",
            )
            .on(arg_query_selector)
            .on(arg_completion_str_selector))
        return [f_ir_hit_rate, f_ndcg_at_k, f_recall_at_k, f_groundtruth_answer]

    def _init_app(self):

        tru_app = TruCustomApp(
            self.rag_app,
            app_name=self.name,
            feedbacks=self.feedback,
            )
        return tru_app
    def run(self ):
        queries = self.ground_truth["query"]
        for i,query in enumerate(queries):
            with self.tru_app as recording:
                self.rag_app.retrieve_and_generate(query,10)
    def leaderboard(self):
        self.session.get_leaderboard(app_ids=[self.tru_app.app_id])



class rag_app:
    def __init__(self, retriever, generator, expected_responses,queries):
        self.retriever = retriever
        self.generator = generator
        self.expected_responses = expected_responses
        self.queries = queries
    
    def _get_scores(self,chunks,expected_response):
        chunks = [chunk["metadata"]["text"] for chunk in chunks]
        return [ score_chunk( chunk , expected_response)  for chunk in chunks]







    @instrument
    def retrieve_and_generate(self, query, k,):
        chunks = self.retriever.get_Chunks(query)
        chunks_dict = [chunk["metadata"]["text"]  for chunk in chunks]
        response = self.generator.generate(query, chunks_dict)
        i = self.queries.index(query)
        expected_response = self.expected_responses[i]
        scores = self._get_scores(chunks,expected_response)
        print(f"retrieved and evaluated \"{query}\"")
        return response, chunks_dict, scores


    

