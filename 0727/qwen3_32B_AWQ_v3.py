# !pip install git+https://github.com/casper-hansen/AutoAWQ.git

# 필요한 라이브러리 가져오기
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, AutoTokenizer
import os, json, re, yaml
from glob import glob
from tqdm import tqdm                      # 진행 상황 보기용
from awq import AutoAWQForCausalLM        # Qwen AWQ 모델 로더
import torch

# ✅ 1. 모델 로드
model_path = "Qwen/Qwen3-32B-AWQ"

# Qwen용 tokenizer 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Qwen 양자화 모델 로드
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True,
    trust_remote_code=True,
    safetensors=True,
    device_map="auto",
    offload_folder="./offload"
)

# 압축 모델 KoBART 로드
compress_model_name = "digit82/kobart-summarization"
compress_tokenizer = PreTrainedTokenizerFast.from_pretrained(compress_model_name)
compress_model = BartForConditionalGeneration.from_pretrained(compress_model_name).to("cuda")


def compress_text_kobart(text, max_input_tokens = compress_tokenizer.model_max_length):

    prompt = """
    다음은 회의 기록의 일부분이야. 회의정보 압축 전문가로서
1. 회의 텍스트의 발언자와 발언내용을 보고 핵심 의미를 보존을 최우선하면서, 정보 소실이 없게 문장과 단어의 수를 가능한 한 줄여 압축해주세요. 
2. 불필요한 수식어, 반복, 예시는 생략하고, 문장 간 중복도 최대한 제거해주세요.
3. 압축된 내용을 정리하고 '발언자': '발언내용' 형식으로 작성하세요. 
  """
    prompt_ids = compress_tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    allowed_text_len = max_input_tokens - prompt_len
    text_ids = compress_tokenizer.encode(text, add_special_tokens=False)[:allowed_text_len]


    # 프롬프트 + 본문 결합
    full_ids = prompt_ids + text_ids
    input_ids = torch.tensor([full_ids]).to("cuda")

    # 생성
    try:
        summary_ids = compress_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
        output_text = compress_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output_text.strip()
    except Exception as e:
        return "[압축 실패: " + str(e) + "]"
    

def sliding_window_compress(text, max_input_tokens=compress_tokenizer.model_max_length, stride=700, return_list=False):
    input_ids = compress_tokenizer.encode(text, add_special_tokens=False)
    compressed_chunks = []
    i = 0
    while i < len(input_ids):
        chunk_ids = input_ids[i:i + max_input_tokens]
        chunk_text = compress_tokenizer.decode(chunk_ids, skip_special_tokens=True)
        compressed = compress_text_kobart(chunk_text, max_input_tokens)
        compressed_chunks.append(compressed)
        i += stride
    return compressed_chunks if return_list else "\n".join(compressed_chunks)


def create_training_dataset(input_dir_pattern, output_jsonl):

    file_paths = glob(input_dir_pattern)
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for file_path in tqdm(file_paths, desc="📂 전체 파일 처리 진행"):
            print(f"\n📁 처리 중: {file_path}")
            utterances = load_jsonl(file_path)

            #  1. [speaker] 발언으로 텍스트 구성
            full_text = "\n".join(f"[{utt['speaker']}] {utt['text']}" for utt in utterances)

            #  2. 슬라이딩 압축
            compressed_chunks = sliding_window_compress(full_text)

            summary_accum = ""
            for idx, chunk in enumerate(tqdm(compressed_chunks, desc=f"🧩 압축 청크 처리 ({os.path.basename(file_path)})", leave=False)):
                prompt = f"""
다음은 회의의 일부입니다.

[회의 내용]
{chunk}

[이전까지의 압축내용]
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
                response = qwen_generate(prompt)
                json.dump({
                    "file": os.path.basename(file_path),
                    "chunk_index": idx,
                    "response": response
                }, f_out, ensure_ascii=False)
                f_out.write("\n")
                summary_accum += response + "\n"


# ✅ 2. JSONL 로드 (speaker 포함)
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [
            {
                "speaker": json.loads(line).get("speaker"),
                "text": json.loads(line).get("text")
            }
            for line in f if "text" in json.loads(line)
        ]


# ✅ 3. 요약 생성 함수
def qwen_generate(prompt, max_new_tokens=2048):
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


# ✅ 4. 전체 처리
def create_qwen_summary(
    input_jsonl_path: str,
    output_txt_path: str,
    max_input_tokens=compress_tokenizer.model_max_length,
    stride=700
):
    """전체 압축 + Qwen 요약 처리 후 txt 저장"""

    utterances = load_jsonl(input_jsonl_path)

    # ✅ [발언자] 형식으로 이어붙이기
    full_text = "\n".join(f"[{utt['speaker']}] {utt['text']}" for utt in utterances)

    # ✅ 슬라이딩 압축
    compressed_text = sliding_window_compress(full_text, max_input_tokens, stride)


    qwen_prompt = f"""
다음은 회의 내용 일부입니다:

[회의 내용]
{compressed_text}

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
    # ✅ Qwen으로 요약 생성
    response = qwen_generate(qwen_prompt)

    # ✅ 결과 저장
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"📄 FILE: {os.path.basename(input_jsonl_path)}\n\n")
        f.write(response)

    print(f"✅ 저장 완료: {output_txt_path}")


# ✅ 6. 실행 진입점
if __name__ == "__main__":
    create_qwen_summary(
        input_jsonl_path="250724_data2_input_sk.jsonl",
        output_txt_path="250724_data2_output_sk.txt"
    )