import json
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import threading
from queue import Queue
import time
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CompressionFocusedBart:
    def __init__(self, model_name="digit82/kobart-summarization", max_workers=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_workers = max_workers or min(4, cpu_count())
        logger.info(f"디바이스: {self.device}, 워커 수: {self.max_workers}")
        
        self.model_name = model_name
        self._local = threading.local()
        
        # 메인 스레드용 모델 로드
        self._load_model()
        logger.info("메인 모델 로드 완료!")

    def _load_model(self):
        """스레드별로 모델을 로드"""
        if not hasattr(self._local, 'model'):
            logger.info(f"스레드 {threading.current_thread().name}에서 모델 로딩...")
            self._local.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_name)
            self._local.model = BartForConditionalGeneration.from_pretrained(
                self.model_name,
                use_safetensors=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self._local.model = self._local.model.to(self.device)
            self._local.model.eval()

    @property
    def tokenizer(self):
        self._load_model()
        return self._local.tokenizer
    
    @property 
    def model(self):
        self._load_model()
        return self._local.model

    def load_data(self, file_path):
        """JSONL 파일 로드"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('text') and item.get('speaker'):
                    data.append({
                        'speaker': item['speaker'],
                        'text': item['text']
                    })
        logger.info(f"{len(data)}개 발언 로드")
        return data

    def create_chunks(self, data, max_tokens=1024, stride=600):
        """정보 보존을 위한 청킹 - 더 많은 중복"""
        full_text = ""
        for item in data:
            full_text += f"{item['speaker']}: {item['text']}\n"
        
        all_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(all_tokens), stride):
            chunk_tokens = all_tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            speakers = []
            for line in chunk_text.split('\n'):
                if ':' in line:
                    speaker = line.split(':')[0].strip()
                    if speaker and speaker not in speakers:
                        speakers.append(speaker)
            
            chunks.append({
                'text': chunk_text,
                'speakers': speakers,
                'index': len(chunks)
            })
            
            if i + max_tokens >= len(all_tokens):
                break
        
        logger.info(f"{len(chunks)}개 청크 생성 (중복 증가로 정보 보존)")
        return chunks

    def create_compression_prompt(self, chunk):
        """압축 전용 프롬프트 생성"""
        speakers = ", ".join(chunk['speakers']) if chunk['speakers'] else "알 수 없음"
        
        prompt = f"""다음 대화 내용을 압축해주세요. 모든 핵심 정보와 화자별 발언을 보존하되, 다음만 제거하세요:

제거할 것:
- 반복되는 표현이나 단어
- 불필요한 감탄사 (아, 어, 음, 그냥 등)
- 중복되는 설명
- 장황한 표현을 간결하게

보존할 것:
- 모든 화자의 발언 내용
- 구체적인 숫자, 날짜, 이름
- 중요한 결정이나 계획
- 문제점이나 우려사항
- 대화의 맥락과 흐름

화자: {speakers}

압축할 내용:
{chunk['text']}"""
        
        return prompt

    def compress_chunk(self, chunk):
        """청크 압축 - 정보 보존 중심"""
        text = chunk['text']
        chunk_idx = chunk['index']
        
        try:
            prompt = self.create_compression_prompt(chunk)
            
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=300,  # 압축률 50-60% 수준으로 조정
                    min_length=100,  # 최소 길이 확보로 정보 보존
                    num_beams=3,
                    repetition_penalty=1.1,  # 반복 방지 약하게
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    length_penalty=0.8,  # 길이를 약간 선호
                    do_sample=False
                )
            
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            if result.strip() and len(result.strip()) > 50:
                return chunk_idx, result.strip()
            else:
                return chunk_idx, self.manual_compress(text)
                
        except Exception as e:
            logger.error(f"청크 {chunk_idx} 압축 실패: {e}")
            return chunk_idx, self.manual_compress(text)

    def manual_compress(self, text):
        """수동 압축 - 정보 보존 우선"""
        lines = []
        for line in text.split('\n'):
            if ':' in line and line.strip():
                # 기본 정리만 수행
                clean = line.replace('[TGT]', '').replace('[/TGT]', '').strip()
                
                # 불필요한 표현만 제거
                clean = re.sub(r'\b(아|어|음|그냥|뭐|좀)\b', '', clean)
                clean = re.sub(r'\s+', ' ', clean)  # 중복 공백 제거
                clean = clean.strip()
                
                if len(clean) > 10:
                    lines.append(clean)
        
        # 모든 라인 보존 (길이만 적당히 조정)
        result_lines = []
        for line in lines:
            if len(line) > 120:
                # 너무 긴 경우만 적당히 자르기
                sentences = line.split('.')
                if len(sentences) > 1:
                    result_lines.append(sentences[0] + '.')
                else:
                    result_lines.append(line[:120] + '...')
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines) if result_lines else "내용 없음"

    def process_with_better_compression(self, input_file, output_file="compressed_output.txt"):
        """향상된 압축 처리"""
        data = self.load_data(input_file)
        chunks = self.create_chunks(data, stride=600)  # 더 많은 중복으로 정보 보존
        
        start_time = time.time()
        results = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(self.compress_chunk, chunk): chunk for chunk in chunks}
            
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk_idx, compressed_text = future.result()
                results[chunk_idx] = compressed_text
                completed += 1
                
                # 압축률 계산 및 로깅
                original_len = len(chunks[chunk_idx]['text'])
                compressed_len = len(compressed_text)
                compression_ratio = (1 - compressed_len/original_len) * 100
                
                logger.info(f"완료: {completed}/{len(chunks)} ({completed/len(chunks)*100:.1f}%) - 압축률: {compression_ratio:.1f}%")
        
        # 결과 조합
        final = "\n\n=== 청크 구분 ===\n\n".join(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final)
        
        # 전체 압축률 계산
        original_total = sum(len(chunk['text']) for chunk in chunks)
        compressed_total = len(final)
        total_compression_ratio = (1 - compressed_total/original_total) * 100
        
        elapsed = time.time() - start_time
        logger.info(f"압축 완료: {output_file}")
        logger.info(f"전체 압축률: {total_compression_ratio:.1f}% (소요시간: {elapsed:.2f}초)")
        return final

    def process_parallel_threads(self, input_file, output_file="bart_output_parallel.txt"):
        """기존 방식 유지 (호환성)"""
        return self.process_with_better_compression(input_file, output_file)

    def process_batch(self, input_file, output_file="bart_output_batch.txt", batch_size=4):
        """배치 처리"""
        data = self.load_data(input_file)
        chunks = self.create_chunks(data, stride=600)
        
        start_time = time.time()
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"배치 {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} 압축중...")
            
            with ThreadPoolExecutor(max_workers=min(len(batch), self.max_workers)) as executor:
                futures = [executor.submit(self.compress_chunk, chunk) for chunk in batch]
                batch_results = [None] * len(batch)
                
                for future in as_completed(futures):
                    chunk_idx, compressed_text = future.result()
                    relative_idx = chunk_idx - i
                    if 0 <= relative_idx < len(batch):
                        batch_results[relative_idx] = compressed_text
            
            results.extend(batch_results)
        
        final = "\n\n=== 청크 구분 ===\n\n".join(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final)
        
        elapsed = time.time() - start_time
        logger.info(f"배치 압축 완료: {output_file} (소요시간: {elapsed:.2f}초)")
        return final


if __name__ == "__main__":
    # 압축 전용 처리기 사용
    compressor = CompressionFocusedBart(max_workers=4)
    
    # 향상된 압축 처리 (추천)
    compressor.process_with_better_compression("250724_data1_intput.jsonl", "0725_data1_output_v3.txt")
    
    # 기존 방식도 지원 (호환성)
    # compressor.process_parallel_threads("250724_data1_intput.jsonl", "0725_data1_output_v2.txt")
    # compressor.process_batch("250724_data1_intput.jsonl", "0725_data1_output_batch.txt", batch_size=2)