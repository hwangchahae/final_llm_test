import json
from konlpy.tag import Okt
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
import torch
import re

# ✅ 모델 로딩 (Qwen3)
model_path = "Qwen/Qwen3-32B-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True,
    trust_remote_code=True,
    safetensors=True,
    device_map="auto",
    offload_folder="./offload"
)

# ✅ 형태소 분석기
okt = Okt()

def extract_keywords(text):
    """형태소 분석: 명사, 동사, 형용사 추출 (한 글자도 포함)"""
    tokens = okt.pos(text, stem=True, norm=True)
    return ' '.join([
        word for word, tag in tokens if tag in ['Noun', 'Verb', 'Adjective']
    ])

def load_jsonl(file_path):
    """발언자, 텍스트 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [
            {
                "speaker": json.loads(line).get("speaker"),
                "text": json.loads(line).get("text")
            }
            for line in f if "text" in json.loads(line)
        ]

def build_compressed_lines(utterances):
    """발언자: 형태소 압축 형식으로 변환"""
    lines = []
    for utt in utterances:
        speaker = utt["speaker"]
        compressed = extract_keywords(utt["text"])
        lines.append(f"{speaker}: {compressed}")
    return "\n".join(lines)

def summarize_with_qwen(prompt, max_new_tokens=2048):
    """Qwen3로 요약 수행"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 제거
    if prompt.strip() in result:
        result = result.replace(prompt.strip(), "", 1).strip()
    match = re.search(r"(### 요약[\s\S]*)", result)
    return match.group(1).strip() if match else result

def main(input_jsonl_path, output_txt_path):
    # 1. 회의 데이터 로드
    utterances = load_jsonl(input_jsonl_path)
    
    # 2~4. 형태소 기반 압축
    compressed_text = build_compressed_lines(utterances)

    # 5. Qwen 프롬프트
    prompt = f"""다음은 회의 내용에서 핵심 단어들만 추출한 기록입니다.
이를 바탕으로 전체 회의를 요약해 주세요.

1. 안건이 있다면 리스트로 정리해줘.
2. 각 안건에 대해 필요한 작업들을 분해해줘. (누가, 무엇을, 언제까지)

결과는 아래 형식으로 줘:

### 요약
- ...

### 안건
1. 제목: ...
   - 세부 설명
   - 관련 발언자

### 업무 분해
- [업무]: 담당자, 마감일, 관련 안건


### 회의 키워드
{compressed_text}

### 요약
"""

    # 6. Qwen으로 요약
    result = summarize_with_qwen(prompt)

    # 7. 저장
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"[입력 파일]: {input_jsonl_path}\n\n")
        f.write(result)
    
    print(f"✅ 회의 요약 저장 완료: {output_txt_path}")

