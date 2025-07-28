import json
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BartChunker:
    def __init__(self, model_name="digit82/kobart-summarization"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"디바이스: {self.device}")
        
        # BART 모델 로드
        logger.info("BART 모델 로딩...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("모델 로드 완료!")

    def load_jsonl(self, file_path):
        """JSONL 파일 로드"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('text') and item.get('speaker'):
                    data.append({
                        'speaker': item['speaker'],
                        'text': item['text']
                    })
        logger.info(f"{len(data)}개 발언 로드")
        return data

    def create_chunks(self, data, max_tokens=1024, stride=800):
        """1024토큰씩 청킹 (stride=800)"""
        # 전체 텍스트 합치기
        full_text = ""
        speaker_map = {}  # 토큰 위치별 화자 매핑
        
        current_pos = 0
        for item in data:
            text_part = f"{item['speaker']}: {item['text']}\n"
            tokens = self.tokenizer.encode(text_part, add_special_tokens=False)
            
            # 화자 정보 저장
            for i in range(len(tokens)):
                speaker_map[current_pos + i] = item['speaker']
            
            full_text += text_part
            current_pos += len(tokens)
        
        # 전체 토큰화
        all_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        chunks = []
        for i in range(0, len(all_tokens), stride):
            chunk_tokens = all_tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # 이 청크에 포함된 화자들
            chunk_speakers = set()
            for j in range(i, min(i + max_tokens, len(all_tokens))):
                if j in speaker_map:
                    chunk_speakers.add(speaker_map[j])
            
            chunks.append({
                'text': chunk_text,
                'speakers': list(chunk_speakers)
            })
            
            if i + max_tokens >= len(all_tokens):
                break
        
        logger.info(f"{len(chunks)}개 청크 생성")
        return chunks

    def compress_chunk(self, chunk):
        """청크를 압축 (요약 아님)"""
        text = chunk['text']
        speakers = chunk['speakers']
        
        # 압축 프롬프트 - 핵심 정보만 추출
        prompt = f"다음 회의 내용에서 핵심 정보만 압축해주세요:\n{text}"

        try:
            inputs = self.tokenizer(
                prompt, 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    min_length=10,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 입력 길이만큼 제거하고 생성된 부분만 가져오기
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            compressed = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 빈 결과면 수동 압축
            if not compressed.strip():
                return self._manual_compress(text, speakers)
                
            return compressed.strip()
            
        except Exception as e:
            logger.error(f"압축 실패: {e}")
            return self._manual_compress(text, speakers)
    
    def _manual_compress(self, text, speakers):
        """수동 압축 - 발언자별 핵심 정보 추출"""
        lines = text.split('\n')
        compressed_lines = []
        
        for line in lines:
            if ':' in line and line.strip():
                # [TGT] 태그 제거하고 핵심만
                clean_line = line.replace('[TGT]', '').replace('[/TGT]', '').strip()
                if len(clean_line) > 10:  # 너무 짧은 건 제외
                    # 길면 앞부분만
                    if len(clean_line) > 100:
                        clean_line = clean_line[:100] + "..."
                    compressed_lines.append(clean_line)
        
        if not compressed_lines:
            return f"참여자: {', '.join(speakers)}"
            
        return '\n'.join(compressed_lines[:5])  # 최대 5줄

    def process_file(self, input_file, output_file="output.txt"):
        """전체 처리 프로세스"""
        # 1. JSONL 로드
        data = self.load_jsonl(input_file)
        
        # 2. 청킹
        chunks = self.create_chunks(data, max_tokens=1024, stride=800)
        
        # 3. 각 청크 압축
        compressed_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"청크 {i+1}/{len(chunks)} 처리 중...")
            compressed = self.compress_chunk(chunk)
            compressed_results.append(compressed)
        
        # 4. 결과 머지
        final_result = "\n\n=== 청크 구분 ===\n\n".join(compressed_results)
        
        # 5. 출력 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_result)
        
        logger.info(f"완료! 결과: {output_file}")
        return final_result

# 사용법
if __name__ == "__main__":
    # 기본 파일로 실행
    input_file = "250724_data1_intput.jsonl"
    output_file = "0725_data1_output_v1.txt"
    
    chunker = BartChunker()
    result = chunker.process_file(input_file, output_file)
    print(f"처리 완료: {output_file}")