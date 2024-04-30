import os
import csv
from typing import List
import string

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import __getattr__ as get_embedding_class
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import asyncio
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select

from database.database import MaliciousPrompt, Model
from env_vars_helpers import DATABASE_URL

async_engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

class MaliciousEmbeddingManager:
    
    def __init__(self):
        # Dynamically get 'model_name' or 'model' from embeddings, with a fallback default value
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
        if not embedding_model_name:
            raise ValueError("EMBEDDING_MODEL environment variable is not set")
        self.embedding_model = get_embedding_class(embedding_model_name)()
        self.model_name = getattr(self.embedding_model, 'model_name', 
                                       getattr(self.embedding_model, 'model', 'default_collection_name'))
        
        self.collection_name = "malicious_prompt_embeddings"
        self.vectorstore = PGVector(
            connection_string=DATABASE_URL,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            )
        self.retriever = self.pg_vector.as_retriever()
        self.tokenizer = MaliciousEmbeddingTokenizer()
        self.malicious_prompt_similarity_threshold = float(os.getenv("MALICIOUS_PROMPT_SIMILARITY_THRESHOLD"),0.95)
        print("finished creating embedding manager")

    def get_embedding_type(self):
        return self.embedding_model.__class__.__name__
    
    async def load_initial_prompts(self, csv_file_path: str = 'malicious_prompts.csv'):
        """Load initial prompts from a CSV file and embed them."""
        model_id = await self.get_model_id()  # Ensures you have the model ID

        # Read prompts from CSV
        prompts = []
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header if there is one
            for row in reader:
                if row:  # Ensure the row is not empty
                    prompts.append(row[0])  # Assumes each row has a prompt in the first column

        # Embed documents
        if prompts:
            metadatas = [{'model': model_id} for _ in prompts]
            await self.embed_documents(prompts, metadatas)

    async def get_model_id(self):
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Model).filter_by(model_name=self.model_name))
            model = result.scalars().first()
            if model:
                return model.id
            model = Model(model_name=self.model_name)
            session.add(model)
            await session.commit()
            return model.id

    async def embed_malicious_prompts(self, malicious_prompts: List[str]):
        new_prompt_ids = []
        new_prompts = []
        model_id = await self.get_model_id()
        for prompt in malicious_prompts:
            prompt_id = self.upsert_malicious_prompt(prompt)
            
            # check to see if this combo of prompt_id & model_id exists for the `malicious_prompt_embeddings` table
            async with AsyncSessionLocal() as session:
                collection = self.vectorstore.get_collection()
                result = await session.execute(
                    select(collection).where(
                    collection.metadata['model_id'].astext == str(model_id),
                    collection.metadata['prompt_id'].astext == str(prompt_id)
                )
                )
                embedding_result = result.scalars().first()

                # add it if it's not there
                if not embedding_result:
                    new_prompt_ids.append(prompt_id)
                    new_prompts.append(prompt)

        # Call the method to do this efficiently
        self.embed_documents(malicious_prompts=new_prompts, malicious_prompt_ids=new_prompt_ids)
            
    async def upsert_malicious_prompt(self, prompt: str) -> int:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(MaliciousPrompt).filter_by(prompt=prompt))
            prompt_entry = result.scalars().first()
            if not prompt_entry:
                prompt_entry = MaliciousPrompt(prompt=prompt)
                session.add(prompt_entry)
                await session.commit()
                await session.refresh(prompt_entry)
            return prompt_entry.id

    async def embed_documents(self, malicious_prompts: List[str], malicious_prompt_ids: List[int]):
        metadatas=[{'model_id': self.model_id, 'malicious_prompt_id': id} for id in malicious_prompt_ids]
        embeddings = []
        if hasattr(self.embedding_model, 'aembed_documents'):
            embeddings = await self.embedding_model.aembed_documents(texts=malicious_prompts)
        else:
            embeddings = self.embedding_model.embed_documents(texts=malicious_prompts)

        # upload the embeddings here more efficiently
        self.vectorstore.add_embeddings(texts=malicious_prompts, embeddings=embeddings, metadatas=metadatas)

    def similarity_search_with_filters(self, query: str, filter: dict, top_k: int = 1):
        return self.retriever.similarity_search(query=query, filter=filter, k=top_k)

    #TODO: Add active_learning_mode, which will embed and store the prompt if it meets the similarity threshold
    def check_for_malicious_content(self, message: str) -> bool:
        """
        Checks if the given message is potentially malicious by tokenizing,
        embedding, and querying a pgvector database.
        """
        tokens = self.tokenizer.tokenize_user_input_to_set(message)
        for token in tokens:
            results = self.similarity_search_with_filters(query=token, filter={'model': self.model_name}, knns=1)
            if results:
                if any(result['score'] >= self.malicious_prompt_similarity_threshold for result in results):
                    return True
        return False


class MaliciousEmbeddingTokenizer:

    def __init__(self):
        self.min_token_length = os.getenv("MIN_TOKEN_LENGTH")
        self.max_token_length = os.getenv("MAX_TOKEN_LENGTH")
        self.token_shift = os.getenv("TOKEN_SHIFT")

    def tokenize_user_input_to_set(self, input: str) -> List[str]:
        # Remove punctuation from the text
        translator = str.maketrans('', '', string.punctuation)
        cleaned_text = input.translate(translator).lower()
        
        # Split the text into words
        words = cleaned_text.split()
        
        # Generate the token groups
        tokenized_output = []
        for size in range(self.min_token_length, self.max_token_length + 1):
            for i in range(len(words) - size + 1):
                tokenized_output.append(" ".join(words[i:i + size]))
                i += self.token_shift - 1 
        
        # As the user could have repeated phrases, return a set to prevent excessive use
        return set(tokenized_output)