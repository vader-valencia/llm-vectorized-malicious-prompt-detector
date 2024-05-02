import os
import csv
from typing import List
import string

from langchain_postgres.vectorstores import PGVector, DistanceStrategy
from advertools import word_tokenize

from langchain_community.embeddings import __getattr__ as get_embedding_class
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import asyncio
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select

from database.database import AsyncSessionLocal, SessionLocal, MaliciousPrompt, Model
from env_vars_helpers import SYNC_DATABASE_URL

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
            connection=SYNC_DATABASE_URL,
            embeddings=self.embedding_model,
            collection_name=self.collection_name,
            distance_strategy=DistanceStrategy.COSINE,
            create_extension=False
            )
        #self.retriever = self.vectorstore.as_retriever()
        self.tokenizer = MaliciousEmbeddingTokenizer()
        self.malicious_prompt_similarity_threshold = float(os.getenv("MALICIOUS_PROMPT_SIMILARITY_THRESHOLD"))
        
    async def initialize(self):
        self.model_id = await self.get_model_id()
        print("finished creating embedding manager")

    def get_embedding_type(self):
        return self.embedding_model.__class__.__name__
    
    async def load_initial_prompts(self, csv_file_path: str = 'malicious_prompts.csv'):
        """Load initial prompts from a CSV file and embed them."""

        # Read prompts from CSV
        prompts = []
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header if there is one
            for row in reader:
                if row:  # Ensure the row is not empty
                    cleaned_prompt = self.tokenizer.clean_prompt(row[0])
                    prompts.append(cleaned_prompt)  # Assumes each row has a prompt in the first column
        
        # Save prompts in the db, then
        # Embed documents
        if prompts:
            await self.embed_malicious_prompts(malicious_prompts=prompts)

    async def get_model_id(self) -> int:
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
        
        for prompt in malicious_prompts:
            prompt_id = await self.upsert_malicious_prompt(prompt)
            
            # check to see if this combo of prompt_id & model_id exists for the `malicious_prompt_embeddings` table
            with SessionLocal() as session:
                result = session.execute(
                    select(self.vectorstore.EmbeddingStore).where(
                    self.vectorstore.EmbeddingStore.cmetadata['model_id'].astext == str(self.model_id),
                    self.vectorstore.EmbeddingStore.cmetadata['malicious_prompt_id'].astext == str(prompt_id)
                )
                )
                embedding_result = result.scalars().first()

                # add it if it's not there
                if not embedding_result:
                    new_prompt_ids.append(prompt_id)
                    new_prompts.append(prompt)

        # Call the method to do this efficiently
        if new_prompts and new_prompt_ids:
            await self.embed_documents(malicious_prompts=new_prompts, malicious_prompt_ids=new_prompt_ids)
            
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

    def similarity_search_with_score_and_filters(self, query: str, filter: dict, top_k: int = 1):
        return self.vectorstore.similarity_search_with_score(query=query, filter=filter, k=top_k)

    #TODO: Add active_learning_mode, which will embed and store the prompt if it meets the similarity threshold
    def check_for_malicious_content(self, message: str) -> bool:
        """
        Checks if the given message is potentially malicious by tokenizing,
        embedding, and querying a pgvector database.
        """
        tokens = self.tokenizer.tokenize_user_input_to_set(message)
        for token in tokens:
            results = self.similarity_search_with_score_and_filters(query=token, filter={'model_id': self.model_id}, top_k=1)
            if results:
                if any(result[1] <= self.malicious_prompt_similarity_threshold for result in results):
                    return True
        return False


# This class was generated by ChatGPT
class MaliciousEmbeddingTokenizer:

    def __init__(self):
        token_lengths = os.getenv("TOKEN_LENGTHS", "5,7,10,12,15")
        self.token_lengths = [int(length) for length in token_lengths.split(',')]

    def clean_prompt(self, input: str) -> str:
        # Remove punctuation from the text
        # We assume a nefarious actor could try to separate out 
        # the malicious instructions by various punctuations
        # make it all lower case as well
        translator = str.maketrans('', '', string.punctuation)
        return input.translate(translator).lower()

    def tokenize_user_input_to_set(self, input: str) -> List[str]:
        cleaned_text = self.clean_prompt(input)
        
        # Split the text into words
        words = cleaned_text.split()
        
        # Generate the token groups for specified n-grams
        # Make this a set so that we don't 
        # embed the same prompt twice
        tokenized_output = set()
        for size in self.token_lengths:
            groups = word_tokenize([cleaned_text], size)
            for group in groups:
                for flat_group in group:  
                    tokenized_output.add(flat_group)

        return tokenized_output
