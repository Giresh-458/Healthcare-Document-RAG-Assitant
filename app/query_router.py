from langchain_core.messages import HumanMessage, SystemMessage

class QueryRouter:
    """
    Acts as a 'Triage Nurse' for incoming questions. 
    It classifies the query intent to dynamically adjust the RAG retrieval strategy
    and block completely irrelevant queries.
    """
    
    @staticmethod
    def classify(question: str, chat_model) -> str:
        """
        Classifies the user query into one of three categories using a fast LLM call:
        FACTOID, AGGREGATION, or IRRELEVANT.
        """
        system_prompt = (
            "You are an intelligent query router for a healthcare document RAG system. "
            "Analyze the user's question and classify it strictly into ONE of these three categories:\n\n"
            "1. FACTOID: The user is asking for a specific fact, value, or single piece of information (e.g., 'What is the hemoglobin level?', 'Who is the patient?').\n"
            "2. AGGREGATION: The user is asking for a summary, a list of items, or asking to analyze multiple values (e.g., 'List all abnormal results', 'Summarize the report', 'What are all the medications?').\n"
            "3. IRRELEVANT: The question is entirely unrelated to healthcare, medical reports, or documents (e.g., 'Write a python script', 'What is the capital of France?').\n\n"
            "Respond with ONLY the category word: FACTOID, AGGREGATION, or IRRELEVANT. Do not include any other text or punctuation."
        )
        
        try:
            response = chat_model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ])
            classification = response.content.strip().upper()
            
            if "AGGREGATION" in classification:
                return "AGGREGATION"
            elif "IRRELEVANT" in classification:
                return "IRRELEVANT"
            else:
                return "FACTOID"
        except Exception as e:
            # Safe fallback if the LLM classification fails for any reason
            return "FACTOID"
