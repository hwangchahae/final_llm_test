# !pip install vllm

import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from datetime import datetime

# ✅ 1. 하위 모델 옵션들 - 원하는 것으로 선택하세요
# 옵션 1: Qwen 7B (빠르고 가벼움) - 추천!
model_path = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# 옵션 2: Qwen 14B (중간 성능)
# model_path = "Qwen/Qwen2.5-14B-Instruct-AWQ" 

# 옵션 3: 더 작은 모델 (매우 빠름)
# model_path = "Qwen/Qwen2.5-3B-Instruct"


print(f"🚀 선택된 모델: {model_path}")

# VLLM 엔진 초기화 (하위 모델용 설정 최적화)
llm = LLM(
    model=model_path,
    quantization="awq_marlin" if "AWQ" in model_path else None,  # AWQ 모델만 quantization 적용
    tensor_parallel_size=1,
    max_model_len=8192,  # 하위 모델은 컨텍스트 길이 줄임
    gpu_memory_utilization=0.8,  # 메모리 사용량 조금 줄임
    trust_remote_code=True,
    enforce_eager=False,
)

# 토크나이저는 별도로 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 하위 모델용 샘플링 파라미터 (더 보수적으로 설정)
sampling_params = SamplingParams(
    temperature=0.2,  # 약간 높여서 창의성 증가
    max_tokens=1536,  # 토큰 수 줄임
    stop=None,
    skip_special_tokens=True,
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
                "text": clean_text(json.loads(line).get("text"))
            }
            for line in f if "text" in json.loads(line)
        ]

# ✅ 3. 청크 분할 (하위 모델용 - 청크 크기 줄임)
def chunk_text(utterances, max_tokens=3000, stride=256):  # 청크 크기 줄임
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

# ✅ 4. 파일 날짜 추출 함수
def get_file_date(file_path):
    """파일명 우선 → 메타데이터 차선"""
    filename = os.path.basename(file_path)
    date_match = re.search(r'(\d{6})', filename)
    if date_match:
        date_str = date_match.group(1)
        try:
            year = "20" + date_str[:2]
            month = date_str[2:4] 
            day = date_str[4:6]
            return f"{year}-{month}-{day}"
        except:
            pass
    
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        return mtime.strftime("%Y-%m-%d")
    except:
        pass
    
    return datetime.now().strftime("%Y-%m-%d")

# ✅ 5. 요약 생성 함수 (하위 모델용 - 프롬프트 간소화)
def generate(prompt, chunk_index):
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text.strip()
    
    if chunk_index == 0:
        match = re.search(r"(### 요약[\s\S]*)", result)
        return match.group(1).strip() if match else result
    else:
        summary_matches = list(re.finditer(r"### 요약", result))
        if len(summary_matches) >= 2:
            return result[summary_matches[1].start():].strip()
        else:
            return result

# ✅ 6. 전체 처리 (하위 모델용 - 프롬프트 최적화)
def create_training_dataset(input_dir_pattern, output_jsonl, model_used):
    file_paths = glob(input_dir_pattern)
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for file_path in tqdm(file_paths, desc="📂 전체 파일 처리 진행"):
            print(f"\n📁 처리 중: {file_path}")
            
            file_date = get_file_date(file_path)
            print(f"📅 기준 날짜: {file_date}")
            
            utterances = load_jsonl(file_path)
            chunks = chunk_text(utterances)

            summary_accum = ""
            for idx, (chunk, speakers) in enumerate(tqdm(chunks, desc=f"🧩 청크 처리 ({os.path.basename(file_path)})", leave=False)):
                participants_str = ", ".join(speakers)

                # 하위 모델용 - 더 간단하고 명확한 프롬프트
                if idx == 0:
                    prompt = f"""회의 내용을 분석해주세요.

참여자: {participants_str}
회의 날짜: {file_date}

회의 내용:
{chunk}

다음 형식으로 정리해주세요:

### 요약
- 주요 내용을 3-5개 문장으로 요약

### 안건
1. 안건명: 설명
2. 안건명: 설명

### 업무 분해
- 업무내용: 담당자, 마감일(1-2주 후), 관련안건

**중요**: 마감일은 {file_date} 기준 1-2주 후로 다양하게 설정하세요."""

                else:
                    prompt = f"""이전 요약을 참고하여 추가 회의 내용을 분석해주세요.

참여자: {participants_str}
회의 날짜: {file_date}

이전 요약:
{summary_accum}

추가 회의 내용:
{chunk}

다음 형식으로 정리해주세요:

### 요약
- 전체 내용 요약 (이전 + 현재)

### 안건
1. 안건명: 설명

### 업무 분해
- 업무내용: 담당자, 마감일(1-2주 후), 관련안건

**중요**: 마감일은 {file_date} 기준 1-2주 후로 다양하게 설정하세요."""
                
                response = generate(prompt, idx)
                json.dump({
                    "file": os.path.basename(file_path),
                    "chunk_index": idx,
                    "file_date": file_date,
                    "model": model_used,  # 파라미터로 받은 model_used 사용
                    "response": response
                }, f_out, ensure_ascii=False)
                f_out.write("\n")
                summary_accum = response + "\n"

def save_final_result_as_txt(output_file, txt_file):
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            print("❌ 파일이 비어있습니다.")
            return
        
        data = json.loads(lines[-1].strip())
        response = data['response']
        file_date = data.get('file_date', 'N/A')
        model_used = data.get('model', 'Unknown')
        
        content = []
        content.append("=" * 80)
        content.append(f"📋 최종 회의 요약 결과 (청크 {data['chunk_index']})")
        content.append(f"📅 회의 날짜: {file_date}")
        content.append(f"🤖 사용 모델: {model_used}")
        content.append("=" * 80)
        content.append("")
        
        sections = response.split('### ')
        for section in sections:
            if not section.strip():
                continue
                
            section = section.strip()
            
            if section.startswith('요약'):
                content.append(f"🎯 요약")
                content.append("-" * 60)
                text_content = section.split('\n', 1)[1] if '\n' in section else ""
                for line in text_content.split('\n'):
                    if line.strip():
                        content.append(f"  {line.strip()}")
                content.append("")
                        
            elif section.startswith('안건'):
                content.append(f"📌 안건")
                content.append("-" * 60)
                text_content = section.split('\n', 1)[1] if '\n' in section else ""
                for line in text_content.split('\n'):
                    if line.strip():
                        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                            content.append(f"\n  📍 {line.strip()}")
                        else:
                            content.append(f"     {line.strip()}")
                content.append("")
                            
            elif section.startswith('업무 분해'):
                content.append(f"⚡ 업무 분해")
                content.append("-" * 60)
                text_content = section.split('\n', 1)[1] if '\n' in section else ""
                for line in text_content.split('\n'):
                    if line.strip() and line.strip().startswith('-'):
                        content.append(f"  ✅ {line.strip()[1:].strip()}")
                content.append("")
        
        content.append("=" * 80)
        
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"✅ 결과가 {txt_file}에 저장되었습니다!")
        print(f"🤖 사용된 모델: {model_used}")
        
    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")

# ✅ 8. 실행
if __name__ == "__main__":
    # 모델명에서 파일명용 문자열 추출
    model_used = model_path.split('/')[-1].replace('-', '_').replace('.', '_')
    print(f"📁 파일명용 모델명: {model_used}")
    
    input_file = "/workspace/250724_data2_intput_sk.jsonl"
    output_file = f"/workspace/250728_{model_used}_data2_output_sk.jsonl"
    txt_file = f"/workspace/250728_{model_used}_data2_output_sk_final1.txt"
    
    print(f"🚀 시작: {model_path} 모델 사용")
    print(f"📝 출력 파일: {output_file}")
    print(f"📄 최종 파일: {txt_file}")
    
    create_training_dataset(input_file, output_file, model_used)
    save_final_result_as_txt(output_file, txt_file)