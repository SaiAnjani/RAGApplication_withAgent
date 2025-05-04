from typing import List, Dict, Any
import re
from langchain.schema import Document

class MCPFramework:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.thinking_steps = [
            "Understanding the question",
            "Analyzing relevant information",
            "Identifying key insights",
            "Formulating precise answer"
        ]
    
    def process_response(self, question: str, rag_response: str, relevant_chunks: List[Document]) -> Dict[str, Any]:
        """
        Process RAG response through iterative thinking to generate CRISP answer.
        
        Args:
            question: User's original question
            rag_response: Response from RAG system
            relevant_chunks: List of relevant document chunks
            
        Returns:
            Dictionary containing:
            - crisp_answer: One-line CRISP answer
            - supporting_chunks: Relevant supporting information
            - thinking_process: Iterative thinking steps
        """
        # Initialize thinking process
        thinking_process = []
        current_answer = rag_response
        
        # Iterative thinking process
        for iteration in range(self.max_iterations):
            for step in self.thinking_steps:
                # Apply thinking step
                processed_answer = self._apply_thinking_step(
                    step, question, current_answer, relevant_chunks
                )
                
                # Store thinking step
                thinking_process.append({
                    "iteration": iteration + 1,
                    "step": step,
                    "result": processed_answer
                })
                
                current_answer = processed_answer
        
        # Extract CRISP answer
        crisp_answer = self._extract_crisp_answer(current_answer)
        
        # Get supporting chunks
        supporting_chunks = self._get_supporting_chunks(crisp_answer, relevant_chunks)
        
        return {
            "crisp_answer": crisp_answer,
            "supporting_chunks": supporting_chunks,
            "thinking_process": thinking_process
        }
    
    def _apply_thinking_step(self, step: str, question: str, current_answer: str, 
                           relevant_chunks: List[Document]) -> str:
        """Apply a specific thinking step to refine the answer."""
        if step == "Understanding the question":
            return self._understand_question(question, current_answer)
        elif step == "Analyzing relevant information":
            return self._analyze_information(current_answer, relevant_chunks)
        elif step == "Identifying key insights":
            return self._identify_insights(current_answer)
        else:  # Formulating precise answer
            return self._formulate_precise_answer(current_answer)
    
    def _understand_question(self, question: str, current_answer: str) -> str:
        """Analyze the question to ensure answer alignment."""
        # Extract key question components
        question_keywords = set(re.findall(r'\w+', question.lower()))
        answer_keywords = set(re.findall(r'\w+', current_answer.lower()))
        
        # Check alignment
        missing_keywords = question_keywords - answer_keywords
        if missing_keywords:
            return f"{current_answer} [Note: Addressing missing aspects: {', '.join(missing_keywords)}]"
        return current_answer
    
    def _analyze_information(self, current_answer: str, relevant_chunks: List[Document]) -> str:
        """Analyze relevant information for completeness."""
        # Check if answer covers all relevant chunks
        covered_chunks = sum(1 for chunk in relevant_chunks 
                           if any(word in current_answer.lower() 
                                for word in chunk.page_content.lower().split()))
        
        if covered_chunks < len(relevant_chunks):
            return f"{current_answer} [Note: Additional relevant information available]"
        return current_answer
    
    def _identify_insights(self, current_answer: str) -> str:
        """Identify key insights from the answer."""
        # Extract key points
        sentences = current_answer.split('.')
        key_points = [s.strip() for s in sentences if len(s.split()) > 5]
        
        if len(key_points) > 1:
            return f"{current_answer} [Key Points: {', '.join(key_points)}]"
        return current_answer
    
    def _formulate_precise_answer(self, current_answer: str) -> str:
        """Formulate a precise, one-line answer."""
        # Remove thinking process notes
        answer = re.sub(r'\[Note:.*?\]', '', current_answer)
        answer = re.sub(r'\[Key Points:.*?\]', '', answer)
        
        # Extract the most relevant sentence
        sentences = answer.split('.')
        if len(sentences) > 1:
            # Find the sentence with the most information
            return max(sentences, key=lambda x: len(x.split()))
        return answer.strip()
    
    def _extract_crisp_answer(self, processed_answer: str) -> str:
        """Extract the final CRISP answer."""
        # Remove all annotations and notes
        answer = re.sub(r'\[.*?\]', '', processed_answer)
        
        # Get the first complete sentence
        sentences = answer.split('.')
        crisp_answer = sentences[0].strip()
        
        # Ensure it's a complete sentence
        if not crisp_answer.endswith('.'):
            crisp_answer += '.'
        
        return crisp_answer
    
    def _get_supporting_chunks(self, crisp_answer: str, relevant_chunks: List[Document]) -> List[Dict[str, Any]]:
        """Get relevant supporting chunks for the CRISP answer."""
        supporting_chunks = []
        answer_keywords = set(re.findall(r'\w+', crisp_answer.lower()))
        
        for chunk in relevant_chunks:
            chunk_keywords = set(re.findall(r'\w+', chunk.page_content.lower()))
            overlap = answer_keywords & chunk_keywords
            
            if len(overlap) > 0:
                supporting_chunks.append({
                    "content": chunk.page_content,
                    "source": chunk.metadata.get("source", "Unknown"),
                    "relevance_score": len(overlap) / len(answer_keywords)
                })
        
        # Sort by relevance
        supporting_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return supporting_chunks 