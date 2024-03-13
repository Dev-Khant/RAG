from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import format_document
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, Qdrant, Redis
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader

from langchain_core.messages import get_buffer_string
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

import logging
logger = logging.getLogger("Chains")


class RAG:
    """Class for RAG chains."""

    def __init__(
        self, llm_name, temperature, vector_db, results_to_retrieve, max_tokens
    ):
        self.llm = ChatGoogleGenerativeAI(model=llm_name, google_api_key="AIzaSyBdkNFydYo6kl0809OqjHZVD7zHuCw6LFA", temperature=temperature, max_tokens=max_tokens)
        self.vector_db = vector_db
        self.results_to_retrieve = results_to_retrieve

        self.contextual_question_chain = None
        self.retrieval_chain = None
        self.final_inputs = None
        self.answer_chain = None
        self.qa_chain = None

    def _combine_documents(self, docs, document_separator="\n\n"):
        """Combine documents into a single string."""

        document_prompt = PromptTemplate.from_template(template="{page_content}")
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
    
    def _unique_documents(self, similar_docs):
        """Get unique documents from similar documents."""
        unqiue_docs = []
        unqiue_page_content = set()
        for doc in similar_docs:
            if doc.page_content not in unqiue_page_content:
                unqiue_docs.append(doc)
                unqiue_page_content.add(doc.page_content)
        return unqiue_docs
    
    def _prepare_non_context_answer(self, chain_answer):
        print(chain_answer)
        json_parser = JsonOutputParser()
        try:
            chain_output = json_parser.parse(chain_answer["chain_output"])
        except Exception as e:
            logger.info(e, exc_info=True)
            chain_answer["chain_output"] += "```"
        chain_output = json_parser.parse(chain_answer["chain_output"])

        if chain_output.get("relevant", "YES") == "NO":
            logger.info("Generating non context answer")

            query = chain_answer["question"]
            temp_prompt = f"""Answer the given question based on your knowledge.

            Question: {query}
            Answer:
            """
            answer = self.llm.predict(temp_prompt)
            return {"chain_output": answer, "is_relevant": "NO"}
        else:
            return {"chain_output": chain_output["answer"], "is_relevant": "YES"}

    def prepare_contextualized_question_chain(self):
        """Chain for contextualized question generation from Chat History."""

        contextualized_question_template = """
        Given the following conversation and a follow up question, rephrase the follow up question containing context from Chat History. If there is no relevant context in the Chat History, please return Follow Up Input as it is.
        Chat History: {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
            template=contextualized_question_template
        )

        self.contextual_question_chain = (
            RunnableParallel(
                {
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
            )
            | CONDENSE_QUESTION_PROMPT
            | self.llm
            | StrOutputParser()
        )
        logger.info("Contextualized Question Chain created.")

    def prepare_retrieval_chain(self):
        """Returns retriever for VectorDB."""

        retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": self.results_to_retrieve})
        # retriever = self.vector_db.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={
        #         "fetch_k": 5,
        #         "k": self.results_to_retrieve,
        #         "lambda_mult": 0.3,
        #     },
        # )
        self.retrieval_chain = RunnableParallel(
            {
                "docs": itemgetter("standalone_question")
                | retriever
                | self._unique_documents,
                "contextualized_question": lambda x: x["standalone_question"],
                "query_language": itemgetter("query_language"),
            }
        )
        logger.info("VectorDB retriever created.")

    def prepare_inputs_chain(self):
        """Prepare inputs for the final chain."""

        self.final_inputs = RunnableParallel(
            {
                "context": lambda x: self._combine_documents(x["docs"]),
                "docs": itemgetter("docs"),
                "query_language": itemgetter("query_language"),
                "contextualized_question": itemgetter("contextualized_question"),
            }
        )

    def prepare_qa_chain(self):
        """Chain for QA retrieval."""

        response_schemas = [
            ResponseSchema(name="answer", description="ANSWER in translated language"),
            ResponseSchema(
                name="relevant",
                description="If the ANSWER is given from the CONTEXT then YES Else NO",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        qa_template = """
        You are an expert in question answering. Use your knowledge to answer the question below using given context.
        Do not add any additional information. Use only the given context to answer the question.
        Answer in {query_language} language and strictly follow instructions as in {format_instructions}.\n

        CONTEXT:
        {context}
        \n\n
        QUESTION:
        {contextualized_question}
        \n\n
        ANSWER in {query_language}:

        """
        qa_prompt = PromptTemplate(
            input_variables=["contextualized_question", "context", "query_language"],
            template=qa_template,
            partial_variables={"format_instructions": format_instructions},
        )
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()
        self.answer_chain = RunnableParallel(
            {
                "question": itemgetter("contextualized_question"),
                "query_language": itemgetter("query_language"),
                "chain_output": self.qa_chain,
            }
        )
        logger.info("QA chain created.")

    def run(self):
        """Run the chain."""

        # prepare chains
        self.prepare_contextualized_question_chain()
        self.prepare_retrieval_chain()
        self.prepare_inputs_chain()
        self.prepare_qa_chain()

        final_chain = (
            RunnableParallel(
                {
                    "standalone_question": self.contextual_question_chain,
                    "query_language": itemgetter("query_language"),
                }
            )
            | self.retrieval_chain
            | self.final_inputs
            | self.answer_chain
            | self._prepare_non_context_answer
        )

        logger.info("RAG chain created.")
        return final_chain


def load_and_create_vectordb(file_path):
    loader = TextLoader(file_path)
    docs = loader.load_and_split()

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=100,
    #     length_function = len
    # )
    # chunked_documents = text_splitter.split_documents(docs)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo-1106",
        chunk_size=800,
        chunk_overlap=80,
    )
    chunked_documents = text_splitter.split_documents(docs)

    # embeddings = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyBdkNFydYo6kl0809OqjHZVD7zHuCw6LFA")
    rds = Redis.from_documents(
        chunked_documents,
        embeddings,
        redis_url="redis://:iNESPERehibl@54.169.182.244:6379",
        index_name="testing_chense_123_final_1234",
        
    )
    print("VectorDB Ready!")
    return rds
