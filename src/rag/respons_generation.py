"""
Generate Responses using a Hugging Face Seq2Seq Model.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        generate_model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
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
            generate_model (AutoModelForSeq2SeqLM): Pretrained Hugging Face model for text generation.
            tokenizer (AutoTokenizer): Tokenizer for the model.
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
        generate_model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
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
            generate_model (AutoModelForSeq2SeqLM): Pretrained Hugging Face model for text generation.
            tokenizer (AutoTokenizer): Tokenizer for the model.
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

        if not isinstance(generate_model, AutoModelForSeq2SeqLM):
            error_msg = "The 'generate_model' must be an instance of AutoModelForSeq2SeqLM."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(tokenizer, AutoTokenizer):
            error_msg = "The 'tokenizer' must be an instance of AutoTokenizer."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        # Start generating context
        pipeline_log.info("Generating context for the model.")
        context = (
            "You are a helpful assistant for a restaurant in Saudi Arabia. "
            "Answer the question based on the provided context:\n\n"
        )

        # Add FAQ context if available
        faq_results = retriever.get("_1_result", [])
        if faq_results:
            context += "FAQs:\n"
            for faq in faq_results:
                context += f"- Q: {faq['question']} A: {faq['answer']}\n"

        # Add menu context if available
        menu_results = retriever.get("_2_result", [])
        if menu_results:
            context += "\nMenu Items:\n"
            for item in menu_results:
                context += (
                    f"- {item['name']}: {item['description']} "
                    f"(Ingredients: {item['ingredients']}, Allergens: {item['allergens']})\n"
                )

        # Ensure the user query is included in the prompt
        context += f"\nUser Query: {query}\n\n"

        pipeline_log.info("Context generated successfully.")
        pipeline_log.debug(f"Generated context: {context}")

        try:
            # Generate response
            input_ids = tokenizer.encode(context, return_tensors="pt", truncation=True)
            outputs = generate_model.generate(
                input_ids,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            pipeline_log.info("Response generated successfully.")
            return response

        except Exception as e:
            error_msg = f"An error occurred during response generation: {e}"
            error_log.error(error_msg)
            raise RuntimeError(error_msg)