# !pip install vllm

import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

# ✅ 1. VLLM 모델 로드 (변경됨)
model_path = "Qwen/Qwen3-32B-AWQ"

# VLLM 엔진 초기화
llm = LLM(
    model=model_path,
    quantization="awq_marlin",  # 더 빠른 awq_marlin 사용
    tensor_parallel_size=1,
    max_model_len=16384,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    enforce_eager=False,
)

# 토크나이저는 별도로 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 샘플링 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=2048,
    stop=None
)

def clean_text(text):
    if not text:
        return ""
    
    # 특정 태그들만 제거
    text = re.sub(r'\[TGT\]', '', text)
    text = re.sub(r'\[/TGT\]', '', text)

    # 공백 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ✅ 2. JSONL 로드 
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [
            {
                "speaker": json.loads(line).get("speaker"),
                "text": clean_text(json.loads(line).get("text"))  # 태그 제거
            }
            for line in f if "text" in json.loads(line)
        ]


# ✅ 3. 청크 분할 
def chunk_text(utterances, max_tokens=5000, stride=512):
    input_ids = []
    metadata = []

    for utt in utterances:
        tokens = tokenizer.encode(utt["text"], add_special_tokens=False)
        input_ids.extend(tokens)
        metadata.extend([utt["speaker"]] * len(tokens))

    chunks = []
    speakers_per_chunk = []

    i = 0
    while i < len(input_ids):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_speakers = metadata[i:i + max_tokens]

        chunk_text = tokenizer.decode(chunk_ids)
        unique_speakers = list(set(chunk_speakers))

        chunks.append(chunk_text)
        speakers_per_chunk.append(unique_speakers)

        i += max_tokens - stride

    return list(zip(chunks, speakers_per_chunk))


# ✅ 4. 요약 생성 함수 (VLLM 방식으로 변경)
def generate(prompt):
    # VLLM 추론
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text.strip()
    
    # '요약' 이후 내용만 추출
    match = re.search(r"(### 요약[\s\S]*)", result)
    return match.group(1).strip() if match else result


# ✅ 5. 전체 처리 (동일)
def create_training_dataset(input_dir_pattern, output_jsonl):
    file_paths = glob(input_dir_pattern)
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for file_path in tqdm(file_paths, desc="📂 전체 파일 처리 진행"):
            print(f"\n📁 처리 중: {file_path}")
            utterances = load_jsonl(file_path)
            chunks = chunk_text(utterances)

            summary_accum = ""
            for idx, (chunk, speakers) in enumerate(tqdm(chunks, desc=f"🧩 청크 처리 ({os.path.basename(file_path)})", leave=False)):
                participants_str = ", ".join(speakers)

                prompt = f"""
다음은 회의의 일부입니다.

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
"""
                response = generate(prompt)
                json.dump({
                    "file": os.path.basename(file_path),
                    "chunk_index": idx,
                    "response": response
                }, f_out, ensure_ascii=False)
                f_out.write("\n")
                summary_accum = response + "\n"


# ✅ 6. 실행 진입점 (동일)
if __name__ == "__main__":
    create_training_dataset(
        "/workspace/250724_data1_input.jsonl",   
        "/workspace/250727_data1_output.jsonl"
    )