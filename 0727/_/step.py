# 필요한 라이브러리 가져오기
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, AutoTokenizer
import torch
from awq import AutoAWQForCausalLM

print("라이브러리 로딩 완료")

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 가능한 디바이스: {device}")

# ✅ 1. Qwen 모델 로드
print("Qwen 모델 로딩 시작...")
model_path = "Qwen/Qwen3-32B-AWQ"

try:
    # Qwen용 tokenizer 불러오기
    print("Qwen tokenizer 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✅ Qwen tokenizer 로딩 완료")
    
    # Qwen 양자화 모델 로드
    print("Qwen 모델 로딩 중... (시간이 걸릴 수 있습니다)")
    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=True,
        trust_remote_code=True,
        safetensors=True,
        device_map="auto",
        offload_folder="./offload"
    )
    print("✅ Qwen 모델 로딩 완료")
    
except Exception as e:
    print(f"❌ Qwen 모델 로딩 실패: {e}")
    model = None
    tokenizer = None

# ✅ 2. KoBART 모델 로드 (torch 버전 문제 해결)
print("\nKoBART 모델 로딩 시작...")
compress_model_name = "digit82/kobart-summarization"

try:
    # 방법 1: use_safetensors=True 옵션 사용 (safetensors 파일이 있는 경우)
    print("KoBART tokenizer 로딩 중...")
    compress_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        compress_model_name,
        use_safetensors=True
    )
    print("✅ KoBART tokenizer 로딩 완료")
    
    print("KoBART 모델 로딩 중... (safetensors 사용)")
    compress_model = BartForConditionalGeneration.from_pretrained(
        compress_model_name,
        use_safetensors=True
    )
    
    # GPU로 이동 (GPU 사용 가능한 경우)
    if device == "cuda":
        compress_model = compress_model.to("cuda")
        print("✅ KoBART 모델 GPU로 이동 완료")
    else:
        print("✅ KoBART 모델 CPU에서 실행")
        
    print("✅ KoBART 모델 로딩 완료 (safetensors)")
    
except Exception as e1:
    print(f"safetensors 방법 실패: {e1}")
    print("대체 모델로 시도 중...")
    
    try:
        # 방법 2: 다른 KoBART 모델 사용
        compress_model_name = "gogamza/kobart-base-v2"
        print(f"대체 모델 시도: {compress_model_name}")
        
        compress_tokenizer = PreTrainedTokenizerFast.from_pretrained(compress_model_name)
        compress_model = BartForConditionalGeneration.from_pretrained(compress_model_name)
        
        if device == "cuda":
            compress_model = compress_model.to("cuda")
            
        print("✅ 대체 KoBART 모델 로딩 완료")
        
    except Exception as e2:
        print(f"❌ 대체 모델도 실패: {e2}")
        print("\n해결 방법:")
        print("1. PyTorch 업그레이드: pip install torch>=2.6")
        print("2. 또는 다음 명령어로 강제 로딩:")
        print("   export TORCH_WEIGHTS_ONLY=False")
        
        compress_model = None
        compress_tokenizer = None

# 로딩 결과 확인
print("\n" + "="*50)
print("모델 로딩 결과:")
print(f"Qwen 모델: {'✅ 성공' if model is not None else '❌ 실패'}")
print(f"KoBART 모델: {'✅ 성공' if compress_model is not None else '❌ 실패'}")
print("="*50)