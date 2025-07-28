import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List, Dict, Tuple
import re

class TextCompressor:
    def __init__(self, model_name: str = "digit82/kobart-summarization"):
        """
        BART 모델을 사용한 텍스트 압축기 초기화
        """
        print(f"Loading {model_name} model...")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """
        JSONL 파일을 로드하여 리스트로 반환
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            print(f"Loaded {len(data)} records from {file_path}")
        except FileNotFoundError:
            print(f"File {file_path} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return []
        
        return data
    
    def extract_text_and_speaker(self, data: List[Dict]) -> str:
        """
        데이터에서 text와 speaker 정보를 추출하여 하나의 텍스트로 결합
        """
        combined_text = ""
        
        for item in data:
            speaker = item.get('speaker', 'UNKNOWN')
            text = item.get('text', '')
            
            # [TGT]와 [/TGT] 태그 제거
            text = re.sub(r'\[TGT\]|\[/TGT\]', '', text).strip()
            
            if text:
                combined_text += f"{speaker}: {text}\n"
        
        print(f"Combined text length: {len(combined_text)} characters")
        return combined_text.strip()
    
    def chunk_text(self, text: str, max_tokens: int = 1024, stride: int = 800) -> List[str]:
        """
        텍스트를 지정된 토큰 수로 청킹 (stride 사용)
        """
        # 텍스트를 토큰으로 변환
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        start = 0
        while start < len(tokens):
            # 청크 끝 위치 계산
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # 토큰을 다시 텍스트로 변환
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # 마지막 청크인 경우 반복 종료
            if end == len(tokens):
                break
            
            # 다음 시작 위치를 stride만큼 이동
            start += stride
        
        print(f"Text split into {len(chunks)} chunks")
        return chunks
    
    def create_compression_prompt(self, chunk: str, prompt_type: str = "default") -> str:
        """
        압축을 위한 프롬프트 생성
        """
        prompts = {
            "default": f"""다음 대화 내용의 핵심 정보를 보존하면서 간결하게 요약하세요. 화자별 주요 발언과 중요한 정보를 놓치지 마세요:

{chunk}

요약:""",
            
            "detailed": f"""다음은 대화 내용입니다. 각 화자의 주요 발언, 결정사항, 중요한 정보를 포함하여 상세하게 요약하세요:

{chunk}

상세 요약:""",
            
            "key_points": f"""다음 대화에서 핵심 포인트만 추출하여 요약하세요:

{chunk}

핵심 포인트:""",
            
            "structured": f"""다음 대화를 구조화된 형태로 요약하세요 (화자별 주요 발언, 결정사항, 액션 아이템 등):

{chunk}

구조화된 요약:"""
        }
        
        return prompts.get(prompt_type, prompts["default"])
    
    def compress_chunk(self, chunk: str, max_length: int = 300, min_length: int = 50, prompt_type: str = "default") -> str:
        """
        BART 모델을 사용하여 개별 청크를 압축
        """
        # 구체적인 프롬프트 구성
        prompt = self.create_compression_prompt(chunk, prompt_type)
        
        # 입력 텍스트 준비
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        ).to(self.device)
        
        # 요약 생성
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                do_sample=False
            )
        
        # 요약 디코딩
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    
    def merge_compressed_chunks(self, compressed_chunks: List[str]) -> str:
        """
        압축된 청크들을 자연스럽게 병합
        """
        if not compressed_chunks:
            return ""
        
        if len(compressed_chunks) == 1:
            return compressed_chunks[0]
        
        # 청크들을 연결하여 최종 병합
        merged_text = "\n".join(compressed_chunks)
        
        # 최종 병합된 텍스트를 한번 더 압축하여 일관성 확보
        if len(self.tokenizer.encode(merged_text)) > 1024:
            final_summary = self.compress_chunk(merged_text, max_length=400, min_length=100)
            return final_summary
        else:
            return merged_text
    
    def process_jsonl_file(self, file_path: str, max_tokens: int = 1024, stride: int = 700, prompt_type: str = "default") -> str:
        """
        전체 파이프라인 실행
        """
        print("="*50)
        print("JSONL 텍스트 압축 파이프라인 시작")
        print("="*50)
        
        # 1. JSONL 파일 로드
        print("\n1. JSONL 파일 로딩...")
        data = self.load_jsonl(file_path)
        if not data:
            return "파일을 로드할 수 없습니다."
        
        # 2. 텍스트와 화자 정보 추출
        print("\n2. 텍스트와 화자 정보 추출...")
        combined_text = self.extract_text_and_speaker(data)
        if not combined_text:
            return "추출할 텍스트가 없습니다."
        
        # 3. 텍스트 청킹
        print(f"\n3. 텍스트 청킹 (max_tokens={max_tokens}, stride={stride})...")
        chunks = self.chunk_text(combined_text, max_tokens, stride)
        
        # 4. 각 청크별 압축
        print("\n4. 청크별 압축 수행...")
        compressed_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"   청크 {i+1}/{len(chunks)} 압축 중...")
            compressed = self.compress_chunk(chunk, prompt_type=prompt_type)
            compressed_chunks.append(compressed)
            print(f"   원본 길이: {len(chunk)} → 압축 길이: {len(compressed)}")
        
        # 5. 압축된 청크들 병합
        print("\n5. 압축된 청크들 병합...")
        final_result = self.merge_compressed_chunks(compressed_chunks)
        
        print(f"\n최종 압축 완료!")
        print(f"원본 텍스트 길이: {len(combined_text)} 문자")
        print(f"최종 압축 길이: {len(final_result)} 문자")
        print(f"압축률: {(1 - len(final_result) / len(combined_text)) * 100:.1f}%")
        
        return final_result

# 사용 예제
def main():
    # 압축기 초기화
    compressor = TextCompressor()
    
    # JSONL 파일 경로 (실제 파일 경로로 변경)
    jsonl_file_path = "250724_data1_intput.jsonl"
    
    # 압축 실행
    result = compressor.process_jsonl_file(
        file_path=jsonl_file_path,
        max_tokens=1024,
        stride=700
    )
    
    print("\n" + "="*50)
    print("최종 압축 결과:")
    print("="*50)
    print(result)
    
    # 결과를 파일로 저장 (선택사항)
    with open("0725_data1_output_v4.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("\n결과가 'compressed_result.txt'에 저장되었습니다.")

if __name__ == "__main__":
    main()