import zipfile
import os

zip_path = "/workspace/batch_triplet_results.zip"         # 압축파일 경로
extract_to = "/workspace/batch_triplet_results"    # 풀 폴더

# 폴더 없으면 생성
os.makedirs(extract_to, exist_ok=True)

# 압축 해제
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"✅ 압축이 {extract_to}에 성공적으로 풀렸습니다.")