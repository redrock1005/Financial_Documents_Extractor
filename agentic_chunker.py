from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_openai import ChatOpenAI
import os
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from dotenv import load_dotenv
from rich import print

load_dotenv()

class AgenticChunker:
    def __init__(self, model_path="/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/KB-Albert"):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True

        # KB-Albert 모델과 토크나이저 로드
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"KB-Albert 모델 로드 실패: {str(e)}")

    def generate_embedding(self, text):
        """텍스트에 대한 임베딩 생성"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 마지막 은닉층의 출력을 평균하여 임베딩 생성
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.cpu().numpy()
        
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {str(e)}")
            return None

    def chunk_and_embed(self, text):
        """텍스트를 청킹하고 각 청크에 대해 임베딩 생성"""
        # 텍스트를 의미 단위로 분할 (예: 문장 단위)
        chunks = self._split_text_into_chunks(text)
        
        # 각 청크에 대해 임베딩 생성
        chunk_embeddings = {chunk: self.generate_embedding(chunk) for chunk in chunks}
        
        return chunk_embeddings

    def _split_text_into_chunks(self, text):
        """텍스트를 의미 단위로 분할하는 메서드"""
        # 간단한 문장 분할 예시 (더 복잡한 로직으로 대체 가능)
        return text.split('. ')

    def generate_response(self, prompt):
        """KB-Albert를 사용하여 응답 생성"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits
            
            predicted_token_ids = torch.argmax(predictions[0], dim=-1)
            response = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
            
            return response.strip()
        
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return None

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)
    
    def add_proposition(self, proposition):
        if self.print_logging:
            print (f"\\nAdding: '{proposition}'")

        # 첫 번째 청크인 경우 새 청크를 만들고 다른 청크는 확인하지 마세요.
        if len(self.chunks) == 0:
            if self.print_logging:
                print ("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # 청크가 발견되면 여기에 명제를 추가합니다.
        if chunk_id:
            if self.print_logging:
                print (f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print ("No chunks found")
            # 청크를 찾을 수 없는 경우 새 청크를 만듭니다.
            self._create_new_chunk(proposition)
        

    def add_proposition_to_chunk(self, chunk_id, proposition):
        # Add then
        self.chunks[chunk_id]['propositions'].append(proposition)

        # 그런 다음 새로운 요약을 확인하세요.
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\\n{proposition}\\n\\nCurrent chunk summary:\\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": "\\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        }).content

        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\\n{proposition}\\n\\nChunk summary:\\n{current_summary}\\n\\nCurrent chunk title:\\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = runnable.invoke({
            "proposition": "\\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        }).content

        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        """새로운 청크 요약 생성"""
        prompt = f"다음 문장을 요약하세요: {proposition}"
        summary = self.generate_response(prompt)
        return summary
    
    def _get_new_chunk_title(self, new_chunk_summary):
        """새로운 청크 제목 생성"""
        prompt = f"다음 요약을 기반으로 제목을 생성하세요: {new_chunk_summary}"
        title = self.generate_response(prompt)
        return title

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit] # I don't want long ids
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }
        if self.print_logging:
            print (f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    
    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\\nSummary: {chunk['summary']}\\n\\n"""
        
            chunk_outline += single_chunk_string
        
        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        """청크 관련성 판단"""
        current_chunk_outline = self.get_chunk_outline()
        
        prompt = f"""
        다음 문장이 어느 청크에 속하는지 판단하세요:
        현재 청크들: {current_chunk_outline}
        
        판단할 문장: {proposition}
        
        [MASK]
        """
        
        response = self.generate_response(prompt)
        if not response:
            return None
            
        # 응답에서 청크 ID 추출
        chunk_id = None
        if len(response) >= self.id_truncate_limit:
            potential_id = response[:self.id_truncate_limit]
            if potential_id.isalnum():
                chunk_id = potential_id
        
        return chunk_id
    
    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks
    
    def pretty_print_chunks(self):
        print (f"\\nYou have {len(self.chunks)} chunks\\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\\n\\n")

    def pretty_print_chunk_outline(self):
        print ("Chunk Outline\\n")
        print(self.get_chunk_outline())

if __name__ == "__main__":
    ac = AgenticChunker()

    ## Comment and uncomment 마음껏 명제를 제안하세요
    propositions = [
        "The month is October.",
        "The year is 2023.",
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        "Teachers and coaches implicitly told us that the returns were linear.",
        "I heard a thousand times that 'You get out what you put in.' ",
        "Teachers and coaches meant well.",
        "In fame, power, military victories, knowledge, and benefit to humanity, the rich get richer."
    ]
    
    ac.add_propositions(propositions)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print (ac.get_chunks(get_type='list_of_strings'))