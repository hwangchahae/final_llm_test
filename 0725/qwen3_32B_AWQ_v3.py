"""
고성능 병렬 회의 요약 시스템
- 하이브리드 파이프라인: 파일읽기(스레드) + GPU처리(프로세스)
- 메모리 최적화: 스트리밍 + 제한된 큐 + 모델 공유
- 성능: GPU 배치처리 + 비동기 파이프라인
"""

import os
import json
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import time
from functools import partial
import logging
from typing import List, Dict, Tuple, Optional
import argparse
from glob import glob
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summary_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GPUPerformanceMonitor:
    """GPU 성능 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.processed_chunks = 0
        self.processed_files = 0
        self.total_tokens = 0
        
    def start(self):
        self.start_time = time.time()
        logger.info("GPU 처리 시작")
        
    def update(self, chunks_count: int, tokens_count: int, files_count: int = 0):
        self.processed_chunks += chunks_count
        self.total_tokens += tokens_count
        self.processed_files += files_count
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            chunks_per_sec = self.processed_chunks / elapsed
            tokens_per_sec = self.total_tokens / elapsed
            
            # GPU 메모리 사용량 체크
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                
                logger.info(f"진행률: {self.processed_chunks}청크, "
                           f"{self.total_tokens}토큰 처리 "
                           f"({chunks_per_sec:.1f}청크/초, {tokens_per_sec:.0f}토큰/초) "
                           f"GPU메모리: {gpu_memory:.1f}GB/{gpu_memory_cached:.1f}GB")
            else:
                logger.info(f"진행률: {self.processed_chunks}청크, "
                           f"{self.total_tokens}토큰 처리 "
                           f"({chunks_per_sec:.1f}청크/초)")
    
    def finish(self):
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"완료: {total_time:.2f}초, "
                       f"총 {self.processed_chunks}청크, "
                       f"{self.total_tokens}토큰, "
                       f"{self.processed_files}파일")
            return total_time
        return 0


class ModelManager:
    """모델 관리 클래스 - 메모리 효율적 로딩"""
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str):
        if self._model is None:
            self.model_path = model_path
            self._load_model()
    
    def _load_model(self):
        """모델과 토크나이저 로드 (한 번만)"""
        logger.info(f"모델 로딩 시작: {self.model_path}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self._model = AutoAWQForCausalLM.from_quantized(
                self.model_path,
                fuse_layers=True,
                trust_remote_code=True,
                safetensors=True,
                device_map="auto",
                offload_folder="./offload"
            )
            
            logger.info("모델 로딩 완료")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise
    
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer


class ParallelSummaryProcessor:
    """고성능 병렬 요약 프로세서"""
    
    def __init__(self, model_path: str = "Qwen/Qwen3-32B-AWQ", 
                 max_workers: Optional[int] = None):
        self.model_path = model_path
        self.max_workers = max_workers or min(4, mp.cpu_count())  # GPU 병목 고려
        self.monitor = GPUPerformanceMonitor()
        
        # 모델 매니저 초기화
        self.model_manager = ModelManager(model_path)
        
        logger.info(f"병렬 요약 시스템 초기화 - GPU워커: {self.max_workers}개")
    
    def _load_jsonl_utterances(self, file_path: str) -> List[Dict]:
        """JSONL 파일에서 발언 데이터 로드"""
        utterances = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if "text" in data and data["text"].strip():
                            utterances.append({
                                "speaker": data.get("speaker", f"speaker_{line_num}"),
                                "text": data["text"].strip()
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"Line {line_num}: JSON 파싱 오류")
                        continue
        except Exception as e:
            logger.error(f"파일 로드 오류: {e}")
        
        return utterances
    
    def _chunk_text_with_speakers(self, utterances: List[Dict], 
                                 max_tokens: int = 5000, stride: int = 3500) -> List[Tuple[str, List[str]]]:
        """슬라이딩 윈도우 청킹 with 발언자 추적"""
        tokenizer = self.model_manager.tokenizer
        
        input_ids = []
        metadata = []  # 각 토큰의 발언자 추적
        
        for utt in utterances:
            try:
                tokens = tokenizer.encode(utt["text"], add_special_tokens=False)
                input_ids.extend(tokens)
                metadata.extend([utt["speaker"]] * len(tokens))
            except Exception as e:
                logger.warning(f"토큰화 오류: {e}")
                continue
        
        if not input_ids:
            return []
        
        chunks = []
        i = 0
        while i < len(input_ids):
            chunk_ids = input_ids[i:i + max_tokens]
            chunk_speakers = metadata[i:i + max_tokens] if i < len(metadata) else []
            
            try:
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                unique_speakers = list(set(chunk_speakers)) if chunk_speakers else ["unknown"]
                
                chunks.append((chunk_text, unique_speakers))
                
                # 마지막 청크면 종료
                if i + max_tokens >= len(input_ids):
                    break
                i += max_tokens - stride
                
            except Exception as e:
                logger.warning(f"청크 디코딩 오류: {e}")
                i += max_tokens - stride
                continue
        
        return chunks
    
    def _generate_summary(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """요약 생성 함수"""
        model = self.model_manager.model
        tokenizer = self.model_manager.tokenizer
        
        try:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
                max_length=8192  # 최대 입력 길이 제한
            ).to("cuda")
            
            with torch.no_grad():
                output = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9
                )
            
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 프롬프트 제거
            if prompt.strip() in result:
                result = result.replace(prompt.strip(), "", 1).strip()
            
            # '요약' 이후 내용만 추출
            match = re.search(r"(### 요약[\s\S]*)", result)
            return match.group(1).strip() if match else result
            
        except Exception as e:
            logger.error(f"요약 생성 오류: {e}")
            return f"요약 생성 실패: {str(e)}"
    
    def _build_summary_prompt(self, chunk: str, speakers: List[str], 
                             summary_accum: str = "") -> str:
        """요약 프롬프트 생성"""
        participants_str = ", ".join(speakers)
        
        return f"""다음은 회의의 일부입니다.

[참여자]
{participants_str}

[회의 청크]
{chunk}

[이전까지의 요약]
{summary_accum}

1. 이 회의 내용을 요약해줘.
2. 안건이 있다면 리스트로 정리해줘.
3. 각 안건에 대해 필요한 작업들을 분해해줘. (누가, 무엇을, 언제까지)

결과는 아래 형식으로 줘:

### 요약
- ...

### 안건
1. 제목: ...
   - 세부 설명
   - 관련 발언자

### 업무 분해
- [업무]: 담당자, 마감일, 관련 안건
""".strip()
    
    def _process_file_batch(self, file_batch_data: List[Tuple[str, List[Dict]]], 
                           max_tokens: int = 5000, stride: int = 3500,
                           max_new_tokens: int = 1024) -> List[Dict]:
        """단일 프로세스에서 파일 배치 처리"""
        results = []
        
        for file_path, utterances in file_batch_data:
            if not utterances:
                continue
            
            try:
                # 청킹
                chunks = self._chunk_text_with_speakers(
                    utterances, max_tokens=max_tokens, stride=stride
                )
                
                if not chunks:
                    continue
                
                # 순차적 요약 (누적)
                summary_accum = ""
                file_results = []
                
                for idx, (chunk, speakers) in enumerate(chunks):
                    prompt = self._build_summary_prompt(chunk, speakers, summary_accum)
                    response = self._generate_summary(prompt, max_new_tokens)
                    
                    result = {
                        "file": os.path.basename(file_path),
                        "chunk_index": idx,
                        "response": response,
                        "speakers": speakers,
                        "tokens": len(chunk.split())  # 대략적 토큰 수
                    }
                    
                    file_results.append(result)
                    summary_accum += response + "\n"
                
                results.extend(file_results)
                
            except Exception as e:
                logger.error(f"파일 {file_path} 처리 실패: {e}")
                continue
        
        return results
    
    def process_hybrid_parallel(self, input_pattern: str, output_file: str,
                               batch_size: int = 2, queue_maxsize: int = 4,
                               max_tokens: int = 5000, stride: int = 3500,
                               max_new_tokens: int = 1024) -> bool:
        """하이브리드 병렬처리: 파일읽기(스레드) + GPU처리(프로세스)"""
        
        file_paths = glob(input_pattern)
        if not file_paths:
            logger.error(f"파일을 찾을 수 없습니다: {input_pattern}")
            return False
        
        logger.info(f"하이브리드 병렬처리 시작 - {len(file_paths)}파일, "
                   f"배치크기: {batch_size}, 큐크기: {queue_maxsize}")
        
        # 메모리 효율적인 제한된 큐
        data_queue = Queue(maxsize=queue_maxsize)
        total_files = len(file_paths)
        
        # 스트리밍 파일 읽기 스레드
        def streaming_file_loader():
            current_batch = []
            processed_files = 0
            
            try:
                for file_path in file_paths:
                    try:
                        utterances = self._load_jsonl_utterances(file_path)
                        if utterances:
                            current_batch.append((file_path, utterances))
                            processed_files += 1
                            
                        # 배치가 찼거나 마지막 파일이면 큐에 추가
                        if len(current_batch) >= batch_size or processed_files == total_files:
                            data_queue.put(current_batch, timeout=60)
                            logger.info(f"배치 생성: {len(current_batch)}파일 "
                                       f"({processed_files}/{total_files})")
                            current_batch = []
                            
                    except Exception as e:
                        logger.error(f"파일 로드 실패 {file_path}: {e}")
                        continue
                
                # 종료 신호
                data_queue.put(None)
                logger.info("파일 로딩 완료")
                
            except Exception as e:
                logger.error(f"파일 로더 오류: {e}")
                data_queue.put(None)
        
        # 성능 모니터링 시작
        self.monitor.start()
        
        # 파일 읽기 스레드 시작
        loader_thread = threading.Thread(target=streaming_file_loader, daemon=True)
        loader_thread.start()
        
        # GPU 처리 프로세스 풀
        all_results = []
        process_func = partial(self._process_file_batch,
                              max_tokens=max_tokens, stride=stride,
                              max_new_tokens=max_new_tokens)
        
        try:
            # 결과 파일 준비
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f_out:
                with ThreadPoolExecutor(max_workers=2) as executor:  # GPU 병목 고려
                    active_futures = {}
                    completed_batches = 0
                    
                    while True:
                        try:
                            batch_data = data_queue.get(timeout=30)
                            
                            if batch_data is None:  # 종료 신호
                                logger.info("모든 배치 수신 완료")
                                break
                            
                            # GPU 처리 시작
                            future = executor.submit(process_func, batch_data)
                            active_futures[future] = len(batch_data)
                            
                            # 완료된 작업들 수집 및 즉시 저장
                            completed_futures = [f for f in active_futures.keys() if f.done()]
                            
                            for future in completed_futures:
                                try:
                                    batch_results = future.result()
                                    
                                    # 즉시 파일에 저장 (메모리 절약)
                                    for result in batch_results:
                                        json.dump(result, f_out, ensure_ascii=False)
                                        f_out.write("\n")
                                    f_out.flush()
                                    
                                    completed_batches += 1
                                    file_count = active_futures.pop(future)
                                    
                                    # 통계 업데이트
                                    total_tokens = sum(r.get("tokens", 0) for r in batch_results)
                                    self.monitor.update(len(batch_results), total_tokens, file_count)
                                    
                                except Exception as e:
                                    logger.error(f"배치 처리 실패: {e}")
                                    active_futures.pop(future, None)
                        
                        except Empty:
                            if not loader_thread.is_alive():
                                break
                        except Exception as e:
                            logger.error(f"큐 처리 오류: {e}")
                            break
                    
                    # 남은 작업들 완료 대기
                    if active_futures:
                        logger.info(f"남은 {len(active_futures)}개 작업 완료 대기...")
                        for future in as_completed(active_futures.keys()):
                            try:
                                batch_results = future.result()
                                
                                for result in batch_results:
                                    json.dump(result, f_out, ensure_ascii=False)
                                    f_out.write("\n")
                                
                                completed_batches += 1
                                file_count = active_futures[future]
                                total_tokens = sum(r.get("tokens", 0) for r in batch_results)
                                self.monitor.update(len(batch_results), total_tokens, file_count)
                                
                            except Exception as e:
                                logger.error(f"최종 배치 처리 실패: {e}")
        
        except KeyboardInterrupt:
            logger.warning("사용자 중단 요청")
            return False
        except Exception as e:
            logger.error(f"처리 오류: {e}")
            return False
        finally:
            if loader_thread.is_alive():
                loader_thread.join(timeout=10)
        
        # 성능 모니터링 완료
        total_time = self.monitor.finish()
        
        logger.info(f"처리 완료: {output_file}")
        return True
    
    def process_simple_parallel(self, input_pattern: str, output_file: str,
                               max_tokens: int = 5000, stride: int = 3500,
                               max_new_tokens: int = 1024) -> bool:
        """간단한 병렬처리 모드 (작은 데이터셋용)"""
        
        file_paths = glob(input_pattern)
        if not file_paths:
            logger.error(f"파일을 찾을 수 없습니다: {input_pattern}")
            return False
        
        logger.info(f"간단한 병렬처리 시작 - {len(file_paths)}파일")
        
        # 모든 파일 데이터 로드
        all_file_data = []
        for file_path in file_paths:
            utterances = self._load_jsonl_utterances(file_path)
            if utterances:
                all_file_data.append((file_path, utterances))
        
        if not all_file_data:
            logger.error("처리할 파일이 없습니다")
            return False
        
        self.monitor.start()
        
        # 순차 처리 (GPU는 하나뿐이므로)
        all_results = []
        for file_path, utterances in tqdm(all_file_data, desc="파일 처리"):
            try:
                batch_results = self._process_file_batch(
                    [(file_path, utterances)],
                    max_tokens=max_tokens,
                    stride=stride,
                    max_new_tokens=max_new_tokens
                )
                all_results.extend(batch_results)
                
                # 통계 업데이트
                total_tokens = sum(r.get("tokens", 0) for r in batch_results)
                self.monitor.update(len(batch_results), total_tokens, 1)
                
            except Exception as e:
                logger.error(f"파일 처리 실패 {file_path}: {e}")
                continue
        
        # 결과 저장
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f_out:
                for result in all_results:
                    json.dump(result, f_out, ensure_ascii=False)
                    f_out.write("\n")
            
            self.monitor.finish()
            logger.info(f"처리 완료: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            return False


def get_optimal_settings(input_pattern: str) -> Dict:
    """파일 개수와 크기에 따른 최적 설정"""
    try:
        file_paths = glob(input_pattern)
        total_size_mb = sum(os.path.getsize(f) for f in file_paths) / 1024 / 1024
        file_count = len(file_paths)
        
        if file_count <= 5 or total_size_mb < 10:  # 작은 데이터셋
            return {
                "mode": "simple",
                "batch_size": 1,
                "queue_maxsize": 2
            }
        elif file_count <= 20 or total_size_mb < 100:  # 중간 데이터셋
            return {
                "mode": "hybrid", 
                "batch_size": 2,
                "queue_maxsize": 4
            }
        else:  # 큰 데이터셋
            return {
                "mode": "hybrid",
                "batch_size": 3,
                "queue_maxsize": 6
            }
    except:
        return {
            "mode": "hybrid",
            "batch_size": 2,
            "queue_maxsize": 4
        }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="고성능 병렬 회의 요약 시스템")
    parser.add_argument("input_pattern", help="입력 파일 패턴 (예: 'data/*.jsonl')")
    parser.add_argument("-o", "--output", default="summary_result.jsonl",
                       help="출력 파일 경로")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-32B-AWQ",
                       help="사용할 AWQ 모델")
    parser.add_argument("-b", "--batch-size", type=int,
                       help="파일 배치 크기 (자동 설정)")
    parser.add_argument("-t", "--max-tokens", type=int, default=5000,
                       help="청크 최대 토큰 수")
    parser.add_argument("-s", "--stride", type=int, default=3500,
                       help="슬라이딩 윈도우 스트라이드")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="생성 최대 토큰 수")
    parser.add_argument("--mode", choices=["simple", "hybrid"],
                       help="처리 모드 (자동 선택)")
    
    args = parser.parse_args()
    
    # 최적 설정 결정
    optimal = get_optimal_settings(args.input_pattern)
    
    batch_size = args.batch_size or optimal["batch_size"]
    mode = args.mode or optimal["mode"]
    
    logger.info(f"최적화된 설정: 입력={args.input_pattern}, 출력={args.output}, "
               f"모드={mode}, 배치={batch_size}, 모델={args.model}")
    
    # 프로세서 초기화 및 실행
    try:
        processor = ParallelSummaryProcessor(model_path=args.model)
        
        if mode == "simple":
            success = processor.process_simple_parallel(
                input_pattern=args.input_pattern,
                output_file=args.output,
                max_tokens=args.max_tokens,
                stride=args.stride,
                max_new_tokens=args.max_new_tokens
            )
        else:  # hybrid
            success = processor.process_hybrid_parallel(
                input_pattern=args.input_pattern,
                output_file=args.output,
                batch_size=batch_size,
                queue_maxsize=optimal["queue_maxsize"],
                max_tokens=args.max_tokens,
                stride=args.stride,
                max_new_tokens=args.max_new_tokens
            )
        
        if success:
            logger.info(f"🎉 모든 처리 완료! 결과: {args.output}")
        else:
            logger.error("❌ 처리 실패")
            
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # 기본 설정으로 실행
        input_pattern = "label_0_output.jsonl"
        output_file = "summary_result.jsonl"
        
        if glob(input_pattern):
            logger.info("기본 설정으로 실행")
            
            processor = ParallelSummaryProcessor()
            optimal = get_optimal_settings(input_pattern)
            
            if optimal["mode"] == "simple":
                success = processor.process_simple_parallel(input_pattern, output_file)
            else:
                success = processor.process_hybrid_parallel(
                    input_pattern=input_pattern,
                    output_file=output_file,
                    batch_size=optimal["batch_size"],
                    queue_maxsize=optimal["queue_maxsize"]
                )
            
            if success:
                logger.info(f"🎉 완료! 결과 확인: {output_file}")
        else:
            logger.error(f"기본 파일({input_pattern})이 없습니다. 명령행 인자를 사용하세요.")
            logger.info("사용법: python script.py 'data/*.jsonl'")
    else:
        main()