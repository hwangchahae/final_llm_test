# !pip install vllm

import os, json, re
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import threading

#  1. 모델 선택
model_path = "Qwen/Qwen3-8B-AWQ"
print(f"🚀 선택된 모델: {model_path}")

# 전역 모델 및 토크나이저 (프로세스별로 초기화)
llm = None
tokenizer = None
sampling_params = None

def initialize_model():
    """각 프로세스에서 모델 초기화"""
    global llm, tokenizer, sampling_params
    
    if llm is None:
        print(f"🔧 프로세스 {os.getpid()}에서 모델 초기화 중...")
        
        # VLLM 엔진 초기화
        llm = LLM(
            model=model_path,
            quantization="awq_marlin" if "AWQ" in model_path else None,
            tensor_parallel_size=1,
            max_model_len=16384,
            gpu_memory_utilization=0.7,  # 병렬 처리 시 메모리 사용량 조정
            trust_remote_code=True,
            enforce_eager=False,
        )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 샘플링 파라미터
        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=2048,
            stop=None,
            skip_special_tokens=True,
        )
        print(f"✅ 프로세스 {os.getpid()} 모델 초기화 완료")

def generate_notion_project_prompt(meeting_transcript: str) -> str:
    """노션 기획안 생성 프롬프트"""
    return f"""다음 회의 전사본을 바탕으로 노션에 업로드할 프로젝트 기획안을 작성하세요.

**회의 전사본:**
{meeting_transcript}

**작성 지침:**
1. 회의에서 논의된 내용을 바탕으로 체계적인 기획안을 작성
2. 프로젝트명은 회의 내용을 바탕으로 적절히 명명
3. 목적과 목표는 명확하고 구체적으로 작성
4. 실행 계획은 실현 가능한 단계별로 구성
5. 기대 효과는 정량적/정성적 결과를 포함
6. 모든 내용은 한국어로 작성

**응답 형식:**
다음 JSON 형식으로 응답하세요:
{{
    "project_name": "프로젝트명",
    "project_purpose": "프로젝트의 주요 목적",
    "project_period": "예상 수행 기간 (예: 2025.01.01 ~ 2025.03.31)",
    "project_manager": "담당자명 (회의에서 언급된 경우)",
    "core_objectives": [
        "목표 1: 구체적인 목표",
        "목표 2: 구체적인 목표",
        "목표 3: 구체적인 목표"
    ],
    "core_idea": "핵심 아이디어 설명",
    "idea_description": "아이디어의 기술적/비즈니스적 설명",
    "execution_plan": "단계별 실행 계획과 일정",
    "expected_effects": [
        "기대효과 1: 자세한 설명",
        "기대효과 2: 자세한 설명",
        "기대효과 3: 자세한 설명"
    ]
}}"""

def clean_text(text):
    if not text:
        return ""
    
    # 특정 태그들만 제거
    text = re.sub(r'\[TGT\]', '', text)
    text = re.sub(r'\[/TGT\]', '', text)

    # 공백 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_json_file(file_path):
    """JSON 또는 JSONL 파일을 자동으로 감지하여 로드"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_ext == '.jsonl':
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
                json_data = json.load(f)
                
                if isinstance(json_data, list):
                    data = []
                    for item in json_data:
                        if isinstance(item, dict) and "text" in item:
                            data.append({
                                "speaker": item.get("speaker"),
                                "text": clean_text(item.get("text"))
                            })
                    return data
                    
                elif isinstance(json_data, dict):
                    if "text" in json_data:
                        return [{
                            "speaker": json_data.get("speaker"),
                            "text": clean_text(json_data.get("text"))
                        }]
                    else:
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
                return []
                
    except Exception as e:
        print(f"❌ 파일 로드 오류 ({file_path}): {e}")
        return []

def chunk_text(utterances, max_tokens=5000, stride=512):
    """텍스트를 청크로 분할"""
    if not utterances:
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

def generate_notion_project_plan(chunk_data):
    """개별 청크에서 노션 프로젝트 기획안 생성 (병렬 처리용)"""
    chunk, speakers, chunk_index, file_date, project_accum = chunk_data
    
    # 모델이 초기화되지 않았다면 초기화
    if llm is None:
        initialize_model()
    
    participants_str = ", ".join(speakers) if speakers else "알 수 없음"
    
    # 전체 회의 내용을 하나의 프로젝트 기획안으로 생성
    if chunk_index == 0:
        # 첫 번째 청크: 초기 프로젝트 기획안 생성
        meeting_transcript = f"참여자: {participants_str}\n회의 날짜: {file_date}\n\n회의 내용:\n{chunk}"
        prompt = generate_notion_project_prompt(meeting_transcript)
    else:
        # 추가 청크: 기존 기획안을 업데이트
        meeting_transcript = f"참여자: {participants_str}\n회의 날짜: {file_date}\n\n추가 회의 내용:\n{chunk}"
        prompt = f"""이전에 생성된 프로젝트 기획안을 다음 추가 회의 내용을 반영하여 업데이트하세요.

**기존 프로젝트 기획안:**
{project_accum}

**추가 회의 내용:**
{meeting_transcript}

**업데이트 지침:**
1. 기존 기획안의 내용을 유지하면서 새로운 정보를 통합
2. 중복되는 내용은 병합하고 상충하는 내용은 최신 정보를 우선
3. 프로젝트의 전체적인 일관성을 유지

**응답 형식:**
다음 JSON 형식으로 업데이트된 전체 기획안을 응답하세요:
{{
    "project_name": "프로젝트명",
    "project_purpose": "프로젝트의 주요 목적",
    "project_period": "예상 수행 기간 (예: 2025.01.01 ~ 2025.03.31)",
    "project_manager": "담당자명 (회의에서 언급된 경우)",
    "core_objectives": [
        "목표 1: 구체적인 목표",
        "목표 2: 구체적인 목표",
        "목표 3: 구체적인 목표"
    ],
    "core_idea": "핵심 아이디어 설명",
    "idea_description": "아이디어의 기술적/비즈니스적 설명",
    "execution_plan": "단계별 실행 계획과 일정",
    "expected_effects": [
        "기대효과 1: 자세한 설명",
        "기대효과 2: 자세한 설명",
        "기대효과 3: 자세한 설명"
    ]
}}"""
    
    # 생성
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text.strip()
    
    return result

def process_single_file_parallel(input_file_path, output_dir, model_used, folder_name):
    """단일 파일을 병렬 처리로 노션 프로젝트 기획안 생성"""
    
    print(f"\n📁 병렬 처리 중: {input_file_path}")
    
    # 모델 초기화 (메인 프로세스에서)
    if llm is None:
        initialize_model()
    
    # 출력 파일명 생성
    output_jsonl = os.path.join(output_dir, f"250730_{model_used}_{folder_name}_notion_project.jsonl")
    output_txt = os.path.join(output_dir, f"250730_{model_used}_{folder_name}_notion_project.txt")
    
    file_date = get_file_date(input_file_path)
    
    utterances = load_json_file(input_file_path)
    if not utterances:
        print(f"⚠️  {input_file_path}에서 유효한 데이터를 찾을 수 없습니다.")
        return False
        
    chunks = chunk_text(utterances)
    if not chunks:
        print(f"⚠️  {input_file_path}에서 청크를 생성할 수 없습니다.")
        return False

    # 청크별 병렬 처리 (순차적으로 처리해야 함 - 이전 결과 필요)
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        project_accum = ""
        
        # 청크들을 순차적으로 처리 (각 청크는 이전 결과에 의존)
        for idx, (chunk, speakers) in enumerate(tqdm(chunks, desc=f"🧩 청크 처리 ({folder_name})", leave=False)):
            chunk_data = (chunk, speakers, idx, file_date, project_accum)
            response = generate_notion_project_plan(chunk_data)
            
            json.dump({
                "file": os.path.basename(input_file_path),
                "folder": folder_name,
                "chunk_index": idx,
                "file_date": file_date,
                "model": model_used,
                "response": response
            }, f_out, ensure_ascii=False)
            f_out.write("\n")
            project_accum = response + "\n"
    
    # TXT 파일로 최종 결과 저장
    save_final_result_as_txt(output_jsonl, output_txt, folder_name)
    return True

def save_final_result_as_txt(jsonl_file, txt_file, folder_name):
    """최종 노션 프로젝트 기획안을 TXT 파일로 저장"""
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
        content.append(f"📋 노션 프로젝트 기획안 - {folder_name}")
        content.append(f"📅 회의 날짜: {file_date}")
        content.append(f"🤖 사용 모델: {model_used}")
        content.append(f"📂 처리 폴더: {folder_name}")
        content.append("=" * 80)
        content.append("")
        
        # JSON 파싱 시도
        try:
            # JSON 형태로 파싱
            project_data = json.loads(response)
            
            content.append(f"🎯 프로젝트명: {project_data.get('project_name', 'N/A')}")
            content.append("-" * 60)
            content.append("")
            
            content.append(f"📌 프로젝트 목적")
            content.append(f"  {project_data.get('project_purpose', 'N/A')}")
            content.append("")
            
            content.append(f"📅 프로젝트 기간")
            content.append(f"  {project_data.get('project_period', 'N/A')}")
            content.append("")
            
            content.append(f"👤 프로젝트 담당자")
            content.append(f"  {project_data.get('project_manager', 'N/A')}")
            content.append("")
            
            content.append(f"🎯 핵심 목표")
            objectives = project_data.get('core_objectives', [])
            for i, obj in enumerate(objectives, 1):
                content.append(f"  {i}. {obj}")
            content.append("")
            
            content.append(f"💡 핵심 아이디어")
            content.append(f"  {project_data.get('core_idea', 'N/A')}")
            content.append("")
            
            content.append(f"📖 아이디어 상세 설명")
            content.append(f"  {project_data.get('idea_description', 'N/A')}")
            content.append("")
            
            content.append(f"⚡ 실행 계획")
            content.append(f"  {project_data.get('execution_plan', 'N/A')}")
            content.append("")
            
            content.append(f"🎊 기대 효과")
            effects = project_data.get('expected_effects', [])
            for i, effect in enumerate(effects, 1):
                content.append(f"  {i}. {effect}")
            content.append("")
            
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 원문 그대로 출력
            content.append("📝 프로젝트 기획안 내용:")
            content.append("-" * 60)
            content.append(response)
            content.append("")
        
        content.append("=" * 80)
        
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"✅ 노션 프로젝트 기획안이 {txt_file}에 저장되었습니다!")
        
    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")

def process_file_wrapper(args):
    """ThreadPoolExecutor를 위한 래퍼 함수"""
    folder_name, folder_path, json_file, model_used = args
    
    try:
        print(f"🔄 처리 시작: {folder_name} (프로세스 {os.getpid()})")
        result = process_single_file_parallel(json_file, folder_path, model_used, folder_name)
        if result:
            print(f"✅ {folder_name} 처리 완료")
            return (folder_name, True, None)
        else:
            print(f"❌ {folder_name} 처리 실패")
            return (folder_name, False, "처리 실패")
    except Exception as e:
        print(f"❌ {folder_name} 처리 중 오류: {e}")
        return (folder_name, False, str(e))

def batch_process_folders_parallel(base_dir, model_used, max_workers=None):
    """병렬로 여러 폴더 처리"""
    
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
                subfolders.append((item, item_path, json_file, model_used))
            else:
                print(f"⚠️  {item} 폴더에 05_final_result.json 파일이 없습니다.")
    
    if not subfolders:
        print(f"❌ {base_dir}에서 처리할 수 있는 폴더를 찾을 수 없습니다.")
        return
    
    print(f"📂 총 {len(subfolders)}개 폴더를 병렬 처리합니다:")
    for folder_name, _, _, _ in subfolders:
        print(f"  - {folder_name}")
    
    # 최대 워커 수 결정
    if max_workers is None:
        # GPU 메모리를 고려하여 적절한 수로 제한
        max_workers = min(4, len(subfolders))  # 최대 4개 프로세스
    
    print(f"🚀 {max_workers}개의 워커로 병렬 처리 시작...")
    
    # 메인 프로세스에서 모델 초기화
    initialize_model()
    
    success_count = 0
    failed_folders = []
    
    # ThreadPoolExecutor를 사용한 병렬 처리
    # GPU 메모리 제약으로 인해 ProcessPoolExecutor 대신 ThreadPoolExecutor 사용
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_folder = {
            executor.submit(process_file_wrapper, args): args[0] 
            for args in subfolders
        }
        
        # 진행 상황 모니터링
        with tqdm(total=len(subfolders), desc="📁 전체 폴더 처리", unit="folder") as pbar:
            for future in as_completed(future_to_folder):
                folder_name = future_to_folder[future]
                try:
                    folder_name, success, error = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_folders.append((folder_name, error))
                except Exception as e:
                    failed_folders.append((folder_name, str(e)))
                
                pbar.update(1)
    
    # 결과 출력
    print(f"\n🎉 병렬 배치 처리 완료!")
    print(f"✅ 성공: {success_count}/{len(subfolders)} 폴더")
    
    if failed_folders:
        print(f"❌ 실패한 폴더들:")
        for folder, error in failed_folders:
            print(f"  - {folder}: {error}")

# 실행 부분
if __name__ == "__main__":
    # 모델명에서 파일명용 문자열 추출
    model_used = model_path.split('/')[-1].replace('-', '_').replace('.', '_')
    print(f"📁 파일명용 모델명: {model_used}")
    
    # 배치 처리할 기본 디렉토리
    base_directory = "/workspace/a_results/a"
    
    print(f"🚀 병렬 배치 처리 시작: {model_path} 모델 사용")
    print(f"📂 기본 디렉토리: {base_directory}")
    
    # 병렬 처리 실행 (최대 4개 워커)
    batch_process_folders_parallel(base_directory, model_used, max_workers=4)