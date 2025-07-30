# !pip install vllm

import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from datetime import datetime

#  1. 모델 선택
model_path = "Qwen/Qwen3-14B-AWQ"
print(f"🚀 선택된 모델: {model_path}")

# VLLM 엔진 초기화 (하위 모델용 설정 최적화)
llm = LLM(
    model=model_path,
    quantization="awq_marlin" if "AWQ" in model_path else None,  # AWQ 모델만 quantization 적용
    tensor_parallel_size=1,
    max_model_len=16384,  # 하위 모델은 컨텍스트 길이 줄임
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

# ✅ 2. JSON/JSONL 로드 함수 (확장자 자동 감지)
def load_json_file(file_path):
    """JSON 또는 JSONL 파일을 자동으로 감지하여 로드"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_ext == '.jsonl':
                # JSONL 파일 처리
                print(f"📄 JSONL 파일로 인식: {os.path.basename(file_path)}")
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        if "text" in json_obj:
                            data.append({
                                "speaker": json_obj.get("speaker"),
                                "text": clean_text(json_obj.get("text"))
                            })
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSONL 라인 {line_num} 파싱 오류: {e}")
                        continue
                return data
                
            elif file_ext == '.json':
                # JSON 파일 처리
                print(f"📄 JSON 파일로 인식: {os.path.basename(file_path)}")
                json_data = json.load(f)
                
                # JSON 구조 자동 감지
                if isinstance(json_data, list):
                    # 리스트 형태의 JSON
                    data = []
                    for item in json_data:
                        if isinstance(item, dict) and "text" in item:
                            data.append({
                                "speaker": item.get("speaker"),
                                "text": clean_text(item.get("text"))
                            })
                    return data
                    
                elif isinstance(json_data, dict):
                    # 딕셔너리 형태의 JSON
                    if "text" in json_data:
                        # 단일 객체
                        return [{
                            "speaker": json_data.get("speaker"),
                            "text": clean_text(json_data.get("text"))
                        }]
                    else:
                        # 키-값 쌍에서 텍스트 데이터 찾기
                        data = []
                        for key, value in json_data.items():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, dict) and "text" in item:
                                        data.append({
                                            "speaker": item.get("speaker"),
                                            "text": clean_text(item.get("text"))
                                        })
                            elif isinstance(value, dict) and "text" in value:
                                data.append({
                                    "speaker": value.get("speaker"),
                                    "text": clean_text(value.get("text"))
                                })
                        return data
                else:
                    print(f"⚠️  지원하지 않는 JSON 구조: {type(json_data)}")
                    return []
            else:
                print(f"⚠️  지원하지 않는 파일 확장자: {file_ext}")
                return []
                
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return []
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return []

# ✅ 3. 청크 분할 (하위 모델용 - 청크 크기 줄임)
def chunk_text(utterances, max_tokens=3000, stride=256):  # 청크 크기 줄임
    if not utterances:
        print("⚠️  빈 데이터입니다.")
        return []
        
    input_ids = []
    metadata = []

    for utt in utterances:
        if not utt["text"]:
            continue
        tokens = tokenizer.encode(utt["text"], add_special_tokens=False)
        input_ids.extend(tokens)
        metadata.extend([utt["speaker"]] * len(tokens))

    if not input_ids:
        print("⚠️  토큰화된 데이터가 없습니다.")
        return []

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

# ✅ 6. 단일 파일 처리 함수
def process_single_file(input_file_path, output_dir, model_used, folder_name):
    """단일 05_final_result.json 파일을 처리하여 해당 폴더에 결과 저장"""
    
    print(f"\n📁 처리 중: {input_file_path}")
    
    # 출력 파일명 생성
    output_jsonl = os.path.join(output_dir, f"250730_{model_used}_{folder_name}_summary.jsonl")
    output_txt = os.path.join(output_dir, f"250730_{model_used}_{folder_name}_summary.txt")
    
    file_date = get_file_date(input_file_path)
    print(f"📅 기준 날짜: {file_date}")
    
    utterances = load_json_file(input_file_path)
    if not utterances:
        print(f"⚠️  {input_file_path}에서 유효한 데이터를 찾을 수 없습니다.")
        return False
        
    print(f"📊 로드된 발화 수: {len(utterances)}")
    
    chunks = chunk_text(utterances)
    if not chunks:
        print(f"⚠️  {input_file_path}에서 청크를 생성할 수 없습니다.")
        return False

    # JSONL 파일에 결과 저장
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        summary_accum = ""
        for idx, (chunk, speakers) in enumerate(tqdm(chunks, desc=f"🧩 청크 처리 ({folder_name})", leave=False)):
            participants_str = ", ".join(speakers) if speakers else "알 수 없음"

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
                "file": os.path.basename(input_file_path),
                "folder": folder_name,
                "chunk_index": idx,
                "file_date": file_date,
                "model": model_used,
                "response": response
            }, f_out, ensure_ascii=False)
            f_out.write("\n")
            summary_accum = response + "\n"
    
    # TXT 파일로 최종 결과 저장
    save_final_result_as_txt(output_jsonl, output_txt, folder_name)
    return True

def save_final_result_as_txt(jsonl_file, txt_file, folder_name):
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
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
        content.append(f"📋 최종 회의 요약 결과 - {folder_name}")
        content.append(f"📅 회의 날짜: {file_date}")
        content.append(f"🤖 사용 모델: {model_used}")
        content.append(f"📂 처리 폴더: {folder_name}")
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
        
    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")

# ✅ 7. 배치 처리 메인 함수
def batch_process_folders(base_dir, model_used):
    """batch_triplet_results1 폴더 내의 모든 하위 폴더 처리"""
    
    if not os.path.exists(base_dir):
        print(f"❌ 기본 디렉토리가 존재하지 않습니다: {base_dir}")
        return
    
    # 하위 폴더들 찾기
    subfolders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            json_file = os.path.join(item_path, "05_final_result.json")
            if os.path.exists(json_file):
                subfolders.append((item, item_path, json_file))
            else:
                print(f"⚠️  {item} 폴더에 05_final_result.json 파일이 없습니다.")
    
    if not subfolders:
        print(f"❌ {base_dir}에서 처리할 수 있는 폴더를 찾을 수 없습니다.")
        return
    
    print(f"📂 총 {len(subfolders)}개 폴더를 처리합니다:")
    for folder_name, _, _ in subfolders:
        print(f"  - {folder_name}")
    
    # 각 폴더 처리
    success_count = 0
    for folder_name, folder_path, json_file in tqdm(subfolders, desc="📁 전체 폴더 처리"):
        print(f"\n🔄 처리 중: {folder_name}")
        
        if process_single_file(json_file, folder_path, model_used, folder_name):
            success_count += 1
            print(f"✅ {folder_name} 처리 완료")
        else:
            print(f"❌ {folder_name} 처리 실패")
    
    print(f"\n🎉 배치 처리 완료!")
    print(f"✅ 성공: {success_count}/{len(subfolders)} 폴더")

# ✅ 8. 실행
if __name__ == "__main__":
    # 모델명에서 파일명용 문자열 추출
    model_used = model_path.split('/')[-1].replace('-', '_').replace('.', '_')
    print(f"📁 파일명용 모델명: {model_used}")
    
    # 배치 처리할 기본 디렉토리
    base_directory = "/workspace/batch_triplet_results1"
    
    print(f"🚀 배치 처리 시작: {model_path} 모델 사용")
    print(f"📂 기본 디렉토리: {base_directory}")
    
    batch_process_folders(base_directory, model_used)