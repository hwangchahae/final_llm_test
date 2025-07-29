#!/usr/bin/env python3
"""
TtalKkak Triplet 파이프라인 통합 테스트 코드
팀원들이 독립적으로 테스트할 수 있도록 3개 파일을 하나로 통합

기능:
1. WhisperX JSON 파싱 (whisperX_parser.py)  
2. Triplet 구조 생성 (create_triplets.py)
3. Triplet 전처리 (triplet_preprocessor.py)

사용법:
python triplet_pipeline_test.py [options]
"""

import json
import re
from typing import List, Dict
from datetime import timedelta
import argparse
import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# Part 0: BERT 분류 모델
# ===============================================

class TtalkkakBERTClassifier(nn.Module):
    """TtalKkak 회의 발화 중요도 분류 BERT 모델"""
    
    def __init__(self, model_name="klue/bert-base", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class BERTClassificationService:
    """BERT 분류 서비스"""
    
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(f"BERT 모델 파일을 찾을 수 없습니다: {model_path}")
            logger.info("시뮬레이션 모드로 동작합니다.")
    
    def load_model(self, model_path: str):
        """BERT 모델 로드"""
        try:
            logger.info(f"BERT 모델 로딩 중: {model_path}")
            
            # 모델 폴더 경로 추출
            model_dir = os.path.dirname(model_path)
            
            # 실제 모델 폴더의 토크나이저와 config 사용
            logger.info(f"모델 폴더에서 config와 tokenizer 로딩: {model_dir}")
            
            # 토크나이저 로드 (모델 폴더에서)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # config 로드 (모델 폴더에서)
            from transformers import AutoConfig, AutoModelForSequenceClassification
            config = AutoConfig.from_pretrained(model_dir, num_labels=2)
            
            # 모델 아키텍처 생성
            self.model = AutoModelForSequenceClassification.from_config(config)
            
            # 저장된 가중치 로드
            if model_path.endswith('.pt'):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                logger.error(f"지원하지 않는 모델 형식: {model_path}")
                return False
                
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"BERT 모델 로드 완료 (Device: {self.device})")
            logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
            return True
            
        except Exception as e:
            logger.error(f"❌ BERT 모델 로드 실패: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def classify_text(self, text: str) -> int:
        """텍스트 분류 (0: 중요, 1: 노이즈)"""
        if self.model is None or self.tokenizer is None:
            # 모델이 없으면 시뮬레이션
            return self._simulate_classification(text)
        
        try:
            # 텍스트 전처리
            text = text.replace("[TGT]", "").replace("[/TGT]", "").strip()
            
            # 토큰화
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_label = torch.argmax(probabilities, dim=-1).item()
            
            return predicted_label
            
        except Exception as e:
            logger.error(f"분류 중 오류: {e}")
            return self._simulate_classification(text)
    
    def _simulate_classification(self, text: str) -> int:
        """BERT 분류 시뮬레이션"""
        text = text.lower()
        
        # 중요해 보이는 키워드들
        important_keywords = [
            "프로젝트", "개발", "일정", "완료", "담당", "예산", "계획", "결정", 
            "회의", "보고", "분석", "검토", "승인", "진행", "업무", "과제", "마감"
        ]
        
        # 불필요해 보이는 키워드들  
        noise_keywords = [
            "안녕", "감사", "수고", "그럼", "네", "아", "음", "어", "잠깐", 
            "아무튼", "그냥", "뭐", "좀", "이제", "그거", "이거"
        ]
        
        # 키워드 기반 점수 계산
        important_score = sum(1 for kw in important_keywords if kw in text)
        noise_score = sum(1 for kw in noise_keywords if kw in text)
        
        if important_score > noise_score:
            return 0  # 중요한 발화
        elif noise_score > important_score:
            return 1  # 불필요한 발화
        else:
            # 문장 길이로 판단 (너무 짧으면 노이즈로 간주)
            return 1 if len(text.strip()) < 10 else 0

# ===============================================
# Part 1: WhisperX Parser (whisperX_parser.py)
# ===============================================

def seconds_to_timestamp(seconds: float) -> str:
    """초를 HH:MM:SS 형태로 변환"""
    td = timedelta(seconds=seconds)
    return str(td).split('.')[0].zfill(8)

def split_sentences(text: str) -> List[str]:
    """문장부호 기준으로 텍스트를 문장 단위로 분리"""
    text = re.sub(r'([.?!])(?!\s)', r'\1 ', text)  # 문장부호 뒤에 공백 없으면 추가
    text = re.sub(r'([.?!])\s{2,}', r'\1 ', text)  # 문장부호 뒤 2칸 이상 공백을 1칸으로
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text.strip()) if s.strip()]

def parse_whisperx_json(json_path: str) -> List[Dict]:
    """WhisperX JSON 파일을 파싱하여 구조화된 데이터로 변환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    entries = []
    global_index = 1
    
    for seg_index, seg in enumerate(segments, 1):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start_time = seg.get("start", 0.0)
        timestamp = seconds_to_timestamp(start_time)
        
        sentences = split_sentences(text)
        for i, sent in enumerate(sentences, 1):
            entries.append({
                "timestamp": timestamp,
                "timestamp_order": f"{global_index}-{i}",
                "speaker": speaker,
                "text": sent
            })
        global_index += 1
        
    return entries

def parse_jsonl_file(jsonl_path: str) -> List[Dict]:
    """JSONL 파일을 파싱하여 구조화된 데이터로 변환"""
    entries = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # JSON 라인 파싱
                data = json.loads(line)
                
                # 필수 필드 추출
                timestamp = data.get("timestamp", f"00:00:{line_num:02d}")
                speaker = data.get("speaker", "UNKNOWN")
                text = data.get("text", "").strip()
                
                if text:  # 빈 텍스트가 아닌 경우만 추가
                    # 문장 단위로 분리
                    sentences = split_sentences(text)
                    for i, sent in enumerate(sentences, 1):
                        entries.append({
                            "timestamp": timestamp,
                            "timestamp_order": f"{line_num}-{i}",
                            "speaker": speaker,
                            "text": sent
                        })
                        
            except json.JSONDecodeError as e:
                print(f"Warning: JSON 파싱 오류 (라인 {line_num}): {e}")
                continue
    
    return entries

def parse_text_input(text: str) -> List[Dict]:
    """일반 텍스트를 WhisperX 형태로 변환"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    entries = []
    
    for idx, line in enumerate(lines, 1):
        # 화자: 내용 형태로 파싱 시도
        if ':' in line and not line.startswith('http'):
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            text = parts[1].strip()
        else:
            speaker = "UNKNOWN"
            text = line
            
        # 문장 단위로 분리
        sentences = split_sentences(text)
        for i, sent in enumerate(sentences, 1):
            entries.append({
                "timestamp": f"00:00:{idx:02d}",  # 가상 타임스탬프
                "timestamp_order": f"{idx}-{i}",
                "speaker": speaker,
                "text": sent
            })
    
    return entries

# ===============================================
# Part 2: Triplet Creator (create_triplets.py)
# ===============================================

def create_structured_triplets(data: List[Dict]) -> List[Dict]:
    """WhisperX 데이터를 triplet 구조로 변환"""
    speech_data = [d for d in data if "text" in d]
    
    # 자연스러운 정렬 (natsort 없이 구현)
    def natural_sort_key(item):
        order = item["timestamp_order"]
        parts = order.split('-')
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    
    sorted_data = sorted(speech_data, key=natural_sort_key)
    result = []
    
    for i in range(len(sorted_data)):
        item = sorted_data[i]
        
        # 이전 맥락: 현재 발화 기준 이전 2개 발화 텍스트 결합
        prev_1 = sorted_data[i - 2]["text"] if i - 2 >= 0 else ""
        prev_2 = sorted_data[i - 1]["text"] if i - 1 >= 0 else ""
        prev = f"{prev_1} {prev_2}".strip()
        
        # 다음 맥락: 현재 발화 기준 다음 2개 발화 텍스트 결합
        next_1 = sorted_data[i + 1]["text"] if i + 1 < len(sorted_data) else ""
        next_2 = sorted_data[i + 2]["text"] if i + 2 < len(sorted_data) else ""
        next_ = f"{next_1} {next_2}".strip()
        
        # triplet 구조 데이터 생성
        result.append({
            "timestamp": item.get("timestamp", ""),
            "timestamp_order": item["timestamp_order"],
            "speaker": item["speaker"],
            "prev": prev,
            "target": f"[TGT] {item['text']} [/TGT]",
            "next": next_,
            "label": None  # 임시로 None 설정
        })
    
    return result

# ===============================================
# Part 3: Triplet Preprocessor (triplet_preprocessor.py)  
# ===============================================

def preprocess_triplets(triplets_with_labels: List[Dict], log_file_path: str = None) -> List[Dict]:
    """라벨 기반 triplet 분기 처리"""
    label_0_items = []
    label_1_items = []
    
    for triplet in triplets_with_labels:
        if triplet.get("label") == 0:
            # label 0: 중요한 발화 (유지)
            item = {
                "timestamp": triplet.get("timestamp", ""),
                "timestamp_order": triplet["timestamp_order"],
                "speaker": triplet["speaker"],
                "text": triplet["target"].replace("[TGT] ", "").replace(" [/TGT]", "")
            }
            label_0_items.append(item)
            
        elif triplet.get("label") == 1:
            # label 1: 불필요한 발화 (로그)
            item = {
                "timestamp": triplet.get("timestamp", ""),
                "timestamp_order": triplet["timestamp_order"],
                "speaker": triplet["speaker"],
                "text": triplet["target"].replace("[TGT] ", "").replace(" [/TGT]", ""),
                "label": triplet["label"]
            }
            label_1_items.append(item)
    
    # label 1 항목들을 로그 파일에 저장
    if log_file_path and label_1_items:
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            for item in label_1_items:
                log_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return label_0_items

# ===============================================
# Part 4: 통합 테스트 및 시뮬레이션 함수들
# ===============================================

def bert_classification(triplets: List[Dict], model_path: str = None) -> List[Dict]:
    """실제 BERT 모델을 사용한 분류"""
    
    # 기본 모델 경로 설정
    if model_path is None:
        model_path = r"c:\Users\SH\Desktop\TtalKkak\Bert모델\Ttalkkak_model_v2\Ttalkkak_model_v2.pt"
    
    # BERT 분류 서비스 초기화
    bert_service = BERTClassificationService(model_path)
    
    logger.info(f"🤖 BERT 분류 시작 (총 {len(triplets)}개 triplet)")
    
    for i, triplet in enumerate(triplets):
        text = triplet["target"]
        predicted_label = bert_service.classify_text(text)
        triplet["label"] = predicted_label
        
        if (i + 1) % 50 == 0:  # 50개마다 진행률 출력
            logger.info(f"   진행률: {i+1}/{len(triplets)} ({(i+1)/len(triplets)*100:.1f}%)")
    
    # 분류 결과 통계
    label_0_count = sum(1 for t in triplets if t["label"] == 0)
    label_1_count = sum(1 for t in triplets if t["label"] == 1)
    
    logger.info(f"✅ BERT 분류 완료: 중요 발화 {label_0_count}개, 노이즈 {label_1_count}개")
    
    return triplets

def run_full_pipeline(input_data: str, input_type: str = "auto", output_dir: str = "./output", model_path: str = None):
    """전체 파이프라인 실행"""
    print("TtalKkak Triplet 파이프라인 테스트 시작")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: 입력 데이터 파싱
    print("Step 1: 입력 데이터 파싱")
    if input_type == "json" or (input_type == "auto" and input_data.endswith('.json')):
        # JSON 파일 처리
        parsed_data = parse_whisperx_json(input_data)
        print(f"   WhisperX JSON 파일 파싱 완료: {len(parsed_data)}개 항목")
    elif input_type == "jsonl" or (input_type == "auto" and input_data.endswith('.jsonl')):
        # JSONL 파일 처리
        parsed_data = parse_jsonl_file(input_data)
        print(f"   JSONL 파일 파싱 완료: {len(parsed_data)}개 항목")
    else:
        # 텍스트 처리
        if os.path.isfile(input_data):
            with open(input_data, 'r', encoding='utf-8') as f:
                text_content = f.read()
        else:
            text_content = input_data
        parsed_data = parse_text_input(text_content)
        print(f"   텍스트 입력 파싱 완료: {len(parsed_data)}개 항목")
    
    # 파싱 결과 저장
    with open(f"{output_dir}/01_parsed_data.json", 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
    
    # Step 2: Triplet 구조 생성
    print("\nStep 2: Triplet 구조 생성")
    triplets = create_structured_triplets(parsed_data)
    print(f"   Triplet 구조 생성 완료: {len(triplets)}개 triplet")
    
    # Triplet 결과 저장
    with open(f"{output_dir}/02_triplets.json", 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    
    # Step 3: 실제 BERT 분류
    print("\nStep 3: 실제 BERT 모델을 사용한 분류")
    classified_triplets = bert_classification(triplets, model_path)
    label_0_count = sum(1 for t in classified_triplets if t["label"] == 0)
    label_1_count = sum(1 for t in classified_triplets if t["label"] == 1)
    print(f"   분류 완료: 중요 발화 {label_0_count}개, 노이즈 {label_1_count}개")
    
    # 분류 결과 저장
    with open(f"{output_dir}/03_classified_triplets.json", 'w', encoding='utf-8') as f:
        json.dump(classified_triplets, f, ensure_ascii=False, indent=2)
    
    # Step 4: Triplet 전처리
    print("\nStep 4: Triplet 전처리")
    filtered_data = preprocess_triplets(
        classified_triplets, 
        log_file_path=f"{output_dir}/04_noise_log.jsonl"
    )
    print(f"   전처리 완료: 최종 {len(filtered_data)}개 발화 유지")
    
    # 최종 결과 저장
    with open(f"{output_dir}/05_final_result.json", 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # Step 5: 결과 요약
    print("\nStep 5: 결과 요약")
    print(f"   원본 데이터: {len(parsed_data)}개 항목")
    print(f"   생성된 Triplet: {len(triplets)}개")
    print(f"   중요 발화: {label_0_count}개 ({label_0_count/len(triplets)*100:.1f}%)")
    print(f"   노이즈 발화: {label_1_count}개 ({label_1_count/len(triplets)*100:.1f}%)")
    print(f"   최종 유지: {len(filtered_data)}개")
    print(f"   노이즈 제거율: {(len(triplets)-len(filtered_data))/len(triplets)*100:.1f}%")
    
    print(f"\n파이프라인 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")
    
    return {
        "parsed_data": parsed_data,
        "triplets": triplets,
        "classified_triplets": classified_triplets,
        "filtered_data": filtered_data,
        "stats": {
            "original_count": len(parsed_data),
            "triplet_count": len(triplets),
            "important_count": label_0_count,
            "noise_count": label_1_count,
            "final_count": len(filtered_data),
            "noise_reduction_rate": (len(triplets)-len(filtered_data))/len(triplets)*100
        }
    }

def create_sample_data():
    """샘플 데이터 생성"""
    sample_text = """팀장 김민준: 안녕하세요, 모든 분들. 오늘은 2025년 1월 28일 화요일이고, 새로운 마케팅 캠페인 기획 회의를 시작하겠습니다.
마케팅 매니저 이서연: 마케팅 매니저 이서연 참석했습니다.
디자이너 박지훈: 디자이너 박지훈입니다.
데이터 분석가 최유진: 데이터 분석가 최유진 참석했습니다.
팀장 김민준: 좋습니다. 오늘 회의 목적은 다음 분기 신제품 출시를 위한 마케팅 전략을 수립하는 것입니다.
마케팅 매니저 이서연: 네, 경쟁사 분석 결과 우리 타겟 고객층인 20-30대에서 소셜미디어 마케팅이 가장 효과적인 것으로 나타났습니다.
데이터 분석가 최유진: 맞습니다. 지난 분기 데이터를 보면 인스타그램 광고의 전환율이 12%, 틱톡이 8%였습니다.
팀장 김민준: 그렇다면 소셜미디어 중심으로 전략을 짜는 게 좋겠네요.
디자이너 박지훈: 트렌드를 분석해보니 짧은 영상 콘텐츠와 인터랙티브한 스토리 형태가 인기가 많습니다.
마케팅 매니저 이서연: 좋은 아이디어네요. 인플루언서 협업도 고려해봐야겠습니다.
팀장 김민준: 구체적인 실행 계획을 세워봅시다. 이서연님은 인플루언서 섭외를, 박지훈님은 크리에이티브 제작을, 최유진님은 성과 측정 체계를 담당해주세요.
마케팅 매니저 이서연: 인플루언서 리스트업은 이번 주 금요일까지 완료하겠습니다.
디자이너 박지훈: 시안 작업은 다음 주 수요일까지 준비하겠습니다.
데이터 분석가 최유진: 성과 측정 대시보드는 다음 주 월요일까지 구축하겠습니다.
팀장 김민준: 완벽합니다. 그럼 2주 후인 2월 11일에 중간 점검 회의를 갖겠습니다."""
    
    return sample_text

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='TtalKkak Triplet 파이프라인 테스트')
    parser.add_argument('--input', '-i', type=str, help='입력 파일 경로 또는 텍스트')
    parser.add_argument('--type', '-t', choices=['auto', 'json', 'jsonl', 'text'], default='auto', 
                       help='입력 타입 (auto: 자동감지, json: WhisperX JSON, jsonl: JSON Lines, text: 일반 텍스트)')
    parser.add_argument('--output', '-o', type=str, default='./triplet_test_output', 
                       help='출력 디렉토리')
    parser.add_argument('--sample', '-s', action='store_true', 
                       help='샘플 데이터로 테스트')
    parser.add_argument('--model', '-m', type=str, 
                       default=r"c:\Users\SH\Desktop\TtalKkak\Bert모델\Ttalkkak_model_v2\Ttalkkak_model_v2.pt",
                       help='BERT 모델 파일 경로')
    
    args = parser.parse_args()
    
    try:
        if args.sample:
            # 샘플 데이터 사용
            print("샘플 데이터를 사용한 테스트를 시작합니다.")
            input_data = create_sample_data()
            result = run_full_pipeline(input_data, input_type='text', output_dir=args.output, model_path=args.model)
            
        elif args.input:
            # 지정된 입력 사용
            result = run_full_pipeline(args.input, input_type=args.type, output_dir=args.output, model_path=args.model)
            
        else:
            # 대화형 모드
            print("TtalKkak Triplet 파이프라인 대화형 테스트")
            print("=" * 50)
            print("1. 샘플 데이터로 테스트")
            print("2. 파일 경로 입력")
            print("3. 직접 텍스트 입력")
            
            while True:
                choice = input("\n선택하세요 (1-3): ").strip()
                
                if choice == '1':
                    input_data = create_sample_data()
                    result = run_full_pipeline(input_data, input_type='text', output_dir=args.output, model_path=args.model)
                    break
                elif choice == '2':
                    file_path = input("파일 경로를 입력하세요: ").strip()
                    if os.path.exists(file_path):
                        result = run_full_pipeline(file_path, input_type=args.type, output_dir=args.output, model_path=args.model)
                        break
                    else:
                        print("파일이 존재하지 않습니다.")
                elif choice == '3':
                    print("텍스트를 입력하세요 (빈 줄을 두 번 입력하면 종료):")
                    lines = []
                    empty_count = 0
                    while True:
                        line = input()
                        if line.strip() == "":
                            empty_count += 1
                            if empty_count >= 2:
                                break
                        else:
                            empty_count = 0
                            lines.append(line)
                    
                    if lines:
                        input_data = '\n'.join(lines)
                        result = run_full_pipeline(input_data, input_type='text', output_dir=args.output, model_path=args.model)
                        break
                    else:
                        print("입력된 텍스트가 없습니다.")
                else:
                    print("잘못된 선택입니다. 1-3 중에서 선택하세요.")
        
        # 간단한 통계 출력
        print("\n" + "="*60)
        print("테스트 완료!")
        print(f"결과 파일들이 '{args.output}' 폴더에 저장되었습니다.")
        print("\n생성된 파일들:")
        print("   01_parsed_data.json      - 파싱된 원본 데이터")
        print("   02_triplets.json         - 생성된 Triplet 구조")  
        print("   03_classified_triplets.json - BERT 분류 결과")
        print("   04_noise_log.jsonl       - 제거된 노이즈 발화들")
        print("   05_final_result.json     - 최종 필터링된 결과")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()