import json
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import threading
from queue import Queue
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelBartCompressor:
    def __init__(self, model_name="digit82/kobart-summarization", max_workers=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_workers = max_workers or min(4, cpu_count())  # GPU 메모리 고려해서 적당히 제한
        logger.info(f"디바이스: {self.device}, 워커 수: {self.max_workers}")
        
        # 모델과 토크나이저를 스레드 로컬 저장소에 저장
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

    def create_chunks(self, data, max_tokens=1024, stride=800):
        """토큰 기반 청킹"""
        full_text = ""
        for item in data:
            full_text += f"{item['speaker']}: {item['text']}\n"
        
        # 메인 스레드의 토크나이저 사용
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
                'index': len(chunks)  # 순서 보장용
            })
            
            if i + max_tokens >= len(all_tokens):
                break
        
        logger.info(f"{len(chunks)}개 청크 생성")
        return chunks

    def compress_chunk(self, chunk):
        """청크 압축 (스레드별 모델 사용)"""
        text = chunk['text']
        chunk_idx = chunk['index']
        
        try:
            inputs = self.tokenizer(
                f"압축: {text}", 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=150,
                    min_length=20,
                    num_beams=2,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # 입력 제거하고 생성 부분만
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            if result.strip():
                return chunk_idx, result.strip()
            else:
                return chunk_idx, self.manual_compress(text)
                
        except Exception as e:
            logger.error(f"청크 {chunk_idx} 압축 실패: {e}")
            return chunk_idx, self.manual_compress(text)

    def manual_compress(self, text):
        """수동 압축"""
        lines = []
        for line in text.split('\n'):
            if ':' in line and line.strip():
                clean = line.replace('[TGT]', '').replace('[/TGT]', '').strip()
                if len(clean) > 10:
                    if len(clean) > 80:
                        clean = clean[:80] + "..."
                    lines.append(clean)
        return '\n'.join(lines[:3]) if lines else "내용 없음"

    def process_parallel_threads(self, input_file, output_file="bart_output_parallel.txt"):
        """스레드 기반 병렬 처리"""
        data = self.load_data(input_file)
        chunks = self.create_chunks(data)
        
        start_time = time.time()
        results = [None] * len(chunks)  # 순서 보장용
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 청크를 병렬로 제출
            future_to_chunk = {executor.submit(self.compress_chunk, chunk): chunk for chunk in chunks}
            
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk_idx, compressed_text = future.result()
                results[chunk_idx] = compressed_text
                completed += 1
                logger.info(f"완료: {completed}/{len(chunks)} ({completed/len(chunks)*100:.1f}%)")
        
        # 순서대로 결과 조합
        final = "\n\n=== 청크 구분 ===\n\n".join(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final)
        
        elapsed = time.time() - start_time
        logger.info(f"병렬 처리 완료: {output_file} (소요시간: {elapsed:.2f}초)")
        return final

    def process_batch(self, input_file, output_file="bart_output_batch.txt", batch_size=4):
        """배치 처리 최적화"""
        data = self.load_data(input_file)
        chunks = self.create_chunks(data)
        
        start_time = time.time()
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"배치 {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} 처리중...")
            
            with ThreadPoolExecutor(max_workers=min(len(batch), self.max_workers)) as executor:
                futures = [executor.submit(self.compress_chunk, chunk) for chunk in batch]
                batch_results = [None] * len(batch)
                
                for future in as_completed(futures):
                    chunk_idx, compressed_text = future.result()
                    # 배치 내 상대 인덱스로 변환
                    relative_idx = chunk_idx - i
                    if 0 <= relative_idx < len(batch):
                        batch_results[relative_idx] = compressed_text
            
            results.extend(batch_results)
        
        final = "\n\n=== 청크 구분 ===\n\n".join(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final)
        
        elapsed = time.time() - start_time
        logger.info(f"배치 처리 완료: {output_file} (소요시간: {elapsed:.2f}초)")
        return final

    def process_sequential(self, input_file, output_file="bart_output_sequential.txt"):
        """순차 처리 (비교용)"""
        data = self.load_data(input_file)
        chunks = self.create_chunks(data)
        
        start_time = time.time()
        results = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"청크 {i+1}/{len(chunks)} 처리중...")
            _, compressed = self.compress_chunk(chunk)
            results.append(compressed)
        
        final = "\n\n=== 청크 구분 ===\n\n".join(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final)
        
        elapsed = time.time() - start_time
        logger.info(f"순차 처리 완료: {output_file} (소요시간: {elapsed:.2f}초)")
        return final


if __name__ == "__main__":
    # 사용 방법 예시
    compressor = ParallelBartCompressor(max_workers=4)
    
    # 1. 병렬 처리 (추천)
    compressor.process_parallel_threads("250724_data1_intput.jsonl", "0725_data1_output_v2.txt")
    
    # 2. 배치 처리 (GPU 메모리가 부족할 때)
    # compressor.process_batch("250724_data1_intput.jsonl", "0725_data1_output_batch.txt", batch_size=2)
    
    # 3. 순차 처리 (속도 비교용)
    # compressor.process_sequential("250724_data1_intput.jsonl", "0725_data1_output_sequential.txt")