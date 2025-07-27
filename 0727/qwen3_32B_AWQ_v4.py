# !pip install git+https://github.com/casper-hansen/AutoAWQ.git
# !pip install pyyaml

# 필요한 라이브러리 가져오기
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, AutoTokenizer
import os, json, re, yaml
from glob import glob
from tqdm import tqdm
from awq import AutoAWQForCausalLM
import torch

# 설정 파일 로드
with open('/workspace/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 1. 모델 로드
model_path = config['models']['qwen_path']

# Qwen용 tokenizer 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Qwen 양자화 모델 로드 (AWQ 압축된 모델)
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True,          # 레이어 융합으로 속도 향상
    trust_remote_code=True,    # 커스텀 코드 실행 허용
    safetensors=True,          # 안전한 텐서 형식 사용
    device_map="auto",         # GPU 메모리에 자동 배치
    offload_folder=config['models']['offload_folder']  # GPU 메모리 부족시 저장할 폴더
)

# 압축 모델 KoBART 로드 (한국어 텍스트 압축용)
compress_model_name = config['models']['kobart_path']
compress_tokenizer = PreTrainedTokenizerFast.from_pretrained(compress_model_name)
compress_model = BartForConditionalGeneration.from_pretrained(compress_model_name).to("cuda")


def compress_text_kobart(text, max_input_tokens=compress_tokenizer.model_max_length):
    """
    KoBART를 사용하여 긴 텍스트를 압축하는 함수
    
    Args:
        text: 압축할 텍스트
        max_input_tokens: 입력 토큰 최대 길이 (기본값: 1024)
    
    Returns:
        압축된 텍스트 또는 에러 메시지
    """
    # 설정 파일에서 압축 프롬프트 가져오기
    prompt = config['prompts']['kobart_compression']
    
    # 프롬프트를 토큰으로 변환하여 길이 계산
    prompt_ids = compress_tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # 입력 텍스트가 들어갈 수 있는 최대 토큰 수 계산
    allowed_text_len = max_input_tokens - prompt_len
    text_ids = compress_tokenizer.encode(text, add_special_tokens=False)[:allowed_text_len]

    # 프롬프트 + 본문 결합하여 입력 준비
    full_ids = prompt_ids + text_ids
    input_ids = torch.tensor([full_ids]).to("cuda")

    # KoBART로 압축 텍스트 생성
    try:
        kobart_config = config['generation']['kobart']
        summary_ids = compress_model.generate(
            input_ids, 
            max_length=kobart_config['max_length'],    # 출력 최대 길이 (512)
            num_beams=kobart_config['num_beams'],      # 빔 서치 개수 (4)
            early_stopping=kobart_config['early_stopping']  # 조기 종료 여부 (True)
        )
        output_text = compress_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output_text.strip()
    except Exception as e:
        return "[압축 실패: " + str(e) + "]"
    

def sliding_window_compress(text, max_input_tokens=None, stride=None, return_list=False):
    """
    슬라이딩 윈도우 방식으로 긴 텍스트를 청크 단위로 나누어 압축
    
    Args:
        text: 압축할 전체 텍스트
        max_input_tokens: 각 청크의 최대 토큰 수
        stride: 윈도우 이동 간격 (기본값: 700)
        return_list: True면 리스트로, False면 합친 문자열로 반환
    
    Returns:
        압축된 청크들의 리스트 또는 합쳐진 문자열
    """
    # 기본값 설정
    if max_input_tokens is None:
        max_input_tokens = config['processing']['max_input_tokens'] or compress_tokenizer.model_max_length
    if stride is None:
        stride = config['processing']['stride']
        
    # 전체 텍스트를 토큰으로 변환
    input_ids = compress_tokenizer.encode(text, add_special_tokens=False)
    compressed_chunks = []
    
    # 슬라이딩 윈도우 방식으로 텍스트 분할 및 압축
    i = 0
    while i < len(input_ids):
        # 현재 위치에서 max_input_tokens만큼 추출
        chunk_ids = input_ids[i:i + max_input_tokens]
        chunk_text = compress_tokenizer.decode(chunk_ids, skip_special_tokens=True)
        
        # 각 청크를 KoBART로 압축
        compressed = compress_text_kobart(chunk_text, max_input_tokens)
        compressed_chunks.append(compressed)
        
        # stride만큼 이동 (오버랩 방식)
        i += stride
        
    return compressed_chunks if return_list else "\n".join(compressed_chunks)


def load_jsonl(file_path):
    """
    JSONL 파일에서 발언자와 텍스트 정보를 로드
    
    Args:
        file_path: JSONL 파일 경로
    
    Returns:
        발언자와 텍스트가 포함된 딕셔너리 리스트
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [
            {
                "speaker": json.loads(line).get("speaker"),
                "text": json.loads(line).get("text")
            }
            for line in f if "text" in json.loads(line)
        ]


def qwen_generate(prompt, max_new_tokens=None):
    """
    Qwen 모델을 사용하여 프롬프트에 대한 응답 생성
    
    Args:
        prompt: 입력 프롬프트
        max_new_tokens: 생성할 최대 토큰 수 (기본값: 2048)
    
    Returns:
        생성된 요약 텍스트
    """
    if max_new_tokens is None:
        max_new_tokens = config['generation']['qwen']['max_new_tokens']
        
    # 프롬프트를 토큰으로 변환
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to("cuda")

    # Qwen 모델로 텍스트 생성
    output = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,                    # 새로 생성할 토큰 수
        pad_token_id=tokenizer.eos_token_id,              # 패딩 토큰 ID
        temperature=config['generation']['qwen']['temperature']  # 생성 창의성 조절 (0.1)
    )

    # 생성된 토큰을 텍스트로 변환
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # 출력에서 입력 프롬프트 제거
    if prompt.strip() in result:
        result = result.replace(prompt.strip(), "", 1).strip()

    # '### 요약' 패턴 이후 내용만 추출
    match = re.search(r"(### 요약[\s\S]*)", result)
    return match.group(1).strip() if match else result


def create_qwen_summary(
    input_jsonl_path: str = None,
    output_txt_path: str = None,
    max_input_tokens=None,
    stride=None
):
    """
    전체 회의 기록 처리 파이프라인: 압축 → 요약 → 저장
    
    Args:
        input_jsonl_path: 입력 JSONL 파일 경로
        output_txt_path: 출력 TXT 파일 경로
        max_input_tokens: 압축시 최대 입력 토큰 수
        stride: 슬라이딩 윈도우 이동 간격
    """
    
    # 설정 파일에서 기본값 사용
    if input_jsonl_path is None:
        input_jsonl_path = config['files']['input_jsonl']
    if output_txt_path is None:
        output_txt_path = config['files']['output_txt']
    if max_input_tokens is None:
        max_input_tokens = config['processing']['max_input_tokens'] or compress_tokenizer.model_max_length
    if stride is None:
        stride = config['processing']['stride']

    # 1. JSONL 파일에서 회의 내용 로드
    utterances = load_jsonl(input_jsonl_path)

    # 2. [발언자] 발언내용 형식으로 텍스트 구성
    full_text = "\n".join(f"[{utt['speaker']}] {utt['text']}" for utt in utterances)

    # 3. 슬라이딩 윈도우 방식으로 텍스트 압축
    compressed_text = sliding_window_compress(full_text, max_input_tokens, stride)

    # 4. Qwen용 프롬프트 생성 (압축된 텍스트 포함)
    qwen_prompt = config['prompts']['qwen_summary_template'].format(content=compressed_text)
    
    # 5. Qwen으로 구조화된 요약 생성
    response = qwen_generate(qwen_prompt)

    # 6. 결과를 파일로 저장
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"FILE: {os.path.basename(input_jsonl_path)}\n\n")
        f.write(response)

    print(f"저장 완료: {output_txt_path}")


# 실행 진입점
if __name__ == "__main__":
    # 설정 파일의 기본값으로 회의 요약 실행
    create_qwen_summary()