# BART 프롬프트 토큰 수만 계산하는 간단한 코드

from transformers import PreTrainedTokenizerFast

# KoBART 토크나이저 로드
compress_tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")

# 현재 사용 중인 BART 프롬프트
bart_prompt = """
    다음은 회의 기록의 일부분이야. 회의정보 압축 전문가로서
1. 회의 텍스트의 발언자와 발언내용을 보고 핵심 의미를 보존을 최우선하면서, 정보 소실이 없게 문장과 단어의 수를 가능한 한 줄여 압축해주세요. 
2. 불필요한 수식어, 반복, 예시는 생략하고, 문장 간 중복도 최대한 제거해주세요.
3. 압축된 내용을 정리하고 '발언자': '발언내용' 형식으로 작성하세요. 
  """

# 토큰 수 계산
prompt_tokens = compress_tokenizer.encode(bart_prompt, add_special_tokens=False)
token_count = len(prompt_tokens)

print(f"🔢 토큰 수: {token_count} 토큰")