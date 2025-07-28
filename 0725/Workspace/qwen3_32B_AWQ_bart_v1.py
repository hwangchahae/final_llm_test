import json
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BartCompressor:
    def __init__(self, model_name="digit82/kobart-summarization"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"디바이스: {self.device}")
        
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

    def load_data(self, file_path):
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
        """토큰 기반 청킹"""
        full_text = ""
        for item in data:
            full_text += f"{item['speaker']}: {item['text']}\n"
        
        all_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(all_tokens), stride):
            chunk_tokens = all_tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            speakers = []
            for line in chunk_text.split('\n'):
                if ':' in line:
                    speaker = line.split(':')[0].strip()
                    if speaker and speaker not in speakers:
                        speakers.append(speaker)
            
            chunks.append({
                'text': chunk_text,
                'speakers': speakers
            })
            
            if i + max_tokens >= len(all_tokens):
                break
        
        logger.info(f"{len(chunks)}개 청크 생성")
        return chunks

    def compress_chunk(self, chunk):
        """청크 압축"""
        text = chunk['text']
        
        try:
            inputs = self.tokenizer(
                f"압축: {text}", 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=150,
                    min_length=20,
                    num_beams=2,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # 입력 제거하고 생성 부분만
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            if result.strip():
                return result.strip()
            else:
                return self.manual_compress(text)
                
        except Exception as e:
            logger.error(f"압축 실패: {e}")
            return self.manual_compress(text)

    def manual_compress(self, text):
        """수동 압축"""
        lines = []
        for line in text.split('\n'):
            if ':' in line and line.strip():
                clean = line.replace('[TGT]', '').replace('[/TGT]', '').strip()
                if len(clean) > 10:
                    if len(clean) > 80:
                        clean = clean[:80] + "..."
                    lines.append(clean)
        return '\n'.join(lines[:3]) if lines else "내용 없음"

    def process(self, input_file, output_file="bart_output.txt"):
        """전체 처리"""
        data = self.load_data(input_file)
        chunks = self.create_chunks(data)
        
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"청크 {i+1}/{len(chunks)} 처리중...")
            compressed = self.compress_chunk(chunk)
            results.append(compressed)
        
        final = "\n\n=== 청크 구분 ===\n\n".join(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final)
        
        logger.info(f"완료: {output_file}")
        return final

if __name__ == "__main__":
    compressor = BartCompressor()
    compressor.process("250724_data1_intput.jsonl", "0725_data1_output_v1.txt")