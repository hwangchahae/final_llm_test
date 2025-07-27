# !pip install vllm

import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from datetime import datetime

# ✅ 1. VLLM 모델 로드
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
    stop=None,  # 모델이 자연스럽게 끝내도록 함
    skip_special_tokens=True,  # EOS 토큰에서 자동 종료
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

# ✅ 4. 파일 날짜 추출 함수 (간단화)
def get_file_date(file_path):
    """파일명 우선 → 메타데이터 차선"""
    # 1. 파일명에서 날짜 추출
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
    
    # 2. 메타데이터에서 날짜 추출
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        return mtime.strftime("%Y-%m-%d")
    except:
        pass
    
    # 3. 현재 날짜
    return datetime.now().strftime("%Y-%m-%d")

# ✅ 5. 요약 생성 함수 (청크별 처리)
def generate(prompt, chunk_index):
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text.strip()
    
    if chunk_index == 0:
        # 첫 번째 청크: 전체 구조 반환
        match = re.search(r"(### 요약[\s\S]*)", result)
        return match.group(1).strip() if match else result
    else:
        # 두 번째 청크부터: 두 번째 ### 요약부터 끝까지
        summary_matches = list(re.finditer(r"### 요약", result))
        if len(summary_matches) >= 2:
            return result[summary_matches[1].start():].strip()
        else:
            return result

# ✅ 6. 전체 처리 (중복 제거 및 날짜 개선)
def create_training_dataset(input_dir_pattern, output_jsonl):
    file_paths = glob(input_dir_pattern)
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for file_path in tqdm(file_paths, desc="📂 전체 파일 처리 진행"):
            print(f"\n📁 처리 중: {file_path}")
            
            # 파일 날짜 추출
            file_date = get_file_date(file_path)
            print(f"📅 기준 날짜: {file_date}")
            
            utterances = load_jsonl(file_path)
            chunks = chunk_text(utterances)

            summary_accum = ""
            for idx, (chunk, speakers) in enumerate(tqdm(chunks, desc=f"🧩 청크 처리 ({os.path.basename(file_path)})", leave=False)):
                participants_str = ", ".join(speakers)

                # 첫 번째 청크는 이전 요약 제외
                if idx == 0:
                    prompt = f"""
다음은 회의의 일부입니다.

[참여자]
{participants_str}

[회의 청크]
{chunk}

1. 이 회의 내용을 요약해줘.
2. 안건이 있다면 리스트로 정리해줘.
3. 각 안건에 대해 필요한 작업들을 분해해줘. (누가, 무엇을, 언제까지)

**중요**: 업무 분해에서 마감일은 회의 날짜({file_date}) 기준으로 1주일~2주일 후의 현실적인 날짜를 다양하게 사용해줘. 모든 업무가 같은 날짜면 안 됨.

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
                else:
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

**중요**: 업무 분해에서 마감일은 회의 날짜({file_date}) 기준으로 1주일~2주일 후의 현실적인 날짜를 다양하게 사용해줘. 모든 업무가 같은 날짜면 안 됨.

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
                
                response = generate(prompt, idx)
                json.dump({
                    "file": os.path.basename(file_path),
                    "chunk_index": idx,
                    "file_date": file_date,
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
        
        # 마지막 결과 파싱
        data = json.loads(lines[-1].strip())
        response = data['response']
        file_date = data.get('file_date', 'N/A')
        
        # 내용 구성
        content = []
        content.append("=" * 80)
        content.append(f"📋 최종 회의 요약 결과 (청크 {data['chunk_index']})")
        content.append(f"📅 회의 날짜: {file_date}")
        content.append("=" * 80)
        content.append("")
        
        # 섹션별 포맷팅
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
        
        # 파일 저장
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"✅ 결과가 {txt_file}에 저장되었습니다!")
        
    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")

# ✅ 8. 실행
if __name__ == "__main__":
    input_file = "/workspace/250724_data2_input_lee.jsonl"
    output_file = "/workspace/250724_data2_output_lee.jsonl"
    txt_file = "/workspace/250727_data2_final_result_lee.txt"
    
    create_training_dataset(input_file, output_file)
    save_final_result_as_txt(output_file, txt_file)