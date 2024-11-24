import google.generativeai as genai


class rag_gemini:
    def __ini__(self,api_keys):
        self.llm_key=api_keys.LLM_KEY
        self.vectordb_key=api_keys.VECTORDB_KEY
        self.embedding_key=api_keys.EMBEDDING_KEY
        self.__llm = self.__llm_init()
        self.__embedding = self.__embedding_init()
    
    def __llm_init(self, modelName="gemini-1.5-flash"):
        genai.configure(api_key=self.llm_key)
        genModel = genai.GenerativeModel(
            model_name=modelName,
             system_instruction=prompt
             )

        
    def embed(self,query,model="text-embedding-004"):
        query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
        return query_embedding["embedding"]
    
    def retrieve(self, query, embed):
        
    
    def createPrompt(self,query, context):
        return f"You are a helpful assistant. Your task is to respond to this question: {query}. You must adhere to the provided context to generate answers. Here is the context: {context}"
    
    def generate(self,query, context):
        context = self.embed(query)
        prompt = self.createPrompt(query, context)
        response = genModel.generate_content
        
        
    
    # def embedding_init(self, model ):

    # def __vector_db(self,index_name. embedding ):