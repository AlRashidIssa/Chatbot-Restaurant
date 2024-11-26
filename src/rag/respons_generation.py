"""
Generate Responses using a Hugging Face Seq2Seq Model.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log


class IGenerateResponse(ABC):
    """
    Abstract interface for generating a response using a Hugging Face Seq2Seq model.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        retriever: Dict[str, Any],
        generate_model,
        max_length: int = 250,
        do_sample: bool = True,
        temperature: float = 0.5,
        top_p: float = 0.6,
        top_k: int = 50,
    ) -> str:
        """
        Abstract method to generate a response.

        Args:
            query (str): User query.
            retriever (Dict[str, Any]): Retrieved context from FAISS or other retrieval mechanism.
            generate_model (): Pretrained Hugging Face model for text generation.
            max_length (int): Maximum length of the generated response. Default is 250.
            do_sample (bool): Whether to sample tokens during generation. Default is True.
            temperature (float): Sampling temperature. Default is 0.5.
            top_p (float): Nucleus sampling probability. Default is 0.6.
            top_k (int): Top-K sampling value. Default is 50.

        Returns:
            str: Generated response.

        Raises:
            ValueError: If inputs are invalid.
        """
        pass


class GenerateResponse(IGenerateResponse):
    """
    Implementation for generating a response using a Hugging Face Seq2Seq model.
    """

    def generate(
        self,
        query: str,
        retriever: Dict[str, Any],
        generate_model,
        max_length: int = 250,
        do_sample: bool = True,
        temperature: float = 0.5,
        top_p: float = 0.6,
        top_k: int = 50,
    ) -> str:
        """
        Generate a response based on the user query and retrieved context.

        Args:
            query (str): User query.
            retriever (Dict[str, Any]): Retrieved context from FAISS or other retrieval mechanism.
            generate_model (): Pretrained Hugging Face model for text generation.
            max_length (int): Maximum length of the generated response. Default is 250.
            do_sample (bool): Whether to sample tokens during generation. Default is True.
            temperature (float): Sampling temperature. Default is 0.5.
            top_p (float): Nucleus sampling probability. Default is 0.6.
            top_k (int): Top-K sampling value. Default is 50.

        Returns:
            str: Generated response.
        """
        # Validate inputs
        if not isinstance(query, str):
            error_msg = "The 'query' must be a string."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(retriever, dict):
            error_msg = "The 'retriever' must be a dictionary."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        # Start generating context
        pipeline_log.info("Generating context for the model.")
        faq_results = retriever['_1_result']
        menu_results = retriever['_2_result']
        # Step 3: Prepare Context for Generation
        context = "You are a helpful assistant muslim for a restaurant in Saudi Arabia and your name AlRashid. Answer the question based on the provided context.:\n\n"

        # Add FAQ context if the query is FAQ-related
        context += "FAQs:\n"
        for faq in faq_results:
            context += f"- Q: {faq['question']} A: {faq['answer']}\n"

        context += "\nMenu Items:\n"
        for item in menu_results:
            context += f"- {item['name']}: {item['description']} (Ingredients: {item['ingredients']}, Allergens: {item['allergens']})\n"

        # Ensure the user query is included in the prompt
        context += f"\nUser Query: {query}\n\n"
        pipeline_log.info("Context generated successfully.")
        pipeline_log.debug(f"Generated context: {context}")

        try:
            # Generate response
            response = generate_model(
                context,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
            )

            pipeline_log.info("Response generated successfully.")
            return response[0]['generated_text']
        except Exception as e:
            error_msg = f"An error occurred during response generation: {e}"
            error_log.error(error_msg)
            raise RuntimeError(error_msg)