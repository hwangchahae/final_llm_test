# !pip install git+https://github.com/casper-hansen/AutoAWQ.git

import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
import torch

# ✅ 1. 모델 로드
model_path = "Qwen/Qwen3-32B-AWQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True,
    trust_remote_code=True,
    safetensors=True,
    device_map="auto",
    offload_folder="./offload"
)


# ✅ 2. JSONL 로드 (speaker 포함)
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [
            {
                "speaker": json.loads(line).get("speaker", "unknown"),
                "text": json.loads(line).get("text", "")
            }
            for line in f if "text" in json.loads(line)
        ]


# ✅ 3. 청크 분할 (speaker 추적)
def chunk_text(utterances, max_tokens=5000, stride=512):
    input_ids = []
    metadata = []  # 각 token이 누구 발언인지 추적

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


# ✅ 4. 요약 생성 함수
def generate(prompt, max_new_tokens=1024):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to("cuda")

    output = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # 프롬프트 제거
    if prompt.strip() in result:
        result = result.replace(prompt.strip(), "", 1).strip()

    # '요약' 이후 내용만 추출
    match = re.search(r"(### 요약[\s\S]*)", result)
    return match.group(1).strip() if match else result


# ✅ 5. 전체 처리
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
                summary_accum += response + "\n"


# ✅ 6. 실행 진입점
if __name__ == "__main__":
    create_training_dataset(
        "label_0_output.jsonl",   # 또는 "data/*.jsonl"
        "250724_v2.jsonl"
    )
