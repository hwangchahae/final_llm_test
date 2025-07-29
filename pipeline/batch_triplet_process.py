import os
import subprocess
import sys
from pathlib import Path
import time

def batch_process_triplets():
    """배치 처리 실행"""
    
    # 입력 디렉토리 및 출력 디렉토리 설정
    input_dir = Path(r"C:\Users\Playdata\Desktop\ccc")
    output_base_dir = Path("./batch_triplet_results")
    triplet_script = Path("./triplet_pipeline_test.py")
    
    # 출력 디렉토리 생성
    output_base_dir.mkdir(exist_ok=True)
    
    # 지원하는 파일 확장자
    supported_extensions = {'.json', '.jsonl', '.txt'}
    
    # 입력 디렉토리의 모든 파일 찾기
    input_files = []
    for ext in supported_extensions:
        input_files.extend(list(input_dir.glob(f"*{ext}")))
    
    if not input_files:
        print(f"❌ {input_dir}에서 처리할 파일을 찾을 수 없습니다.")
        print(f"지원하는 확장자: {', '.join(supported_extensions)}")
        return
    
    print(f"🔍 총 {len(input_files)}개 파일 발견")
    print("=" * 80)
    
    success_count = 0
    error_count = 0
    
    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] 처리 중: {input_file.name}")
        
        # 출력 디렉토리 설정 (파일명 기반)
        file_stem = input_file.stem
        output_dir = output_base_dir / f"result_{file_stem}"
        
        # 파일 타입 자동 감지
        if input_file.suffix == '.json':
            file_type = 'json'
        elif input_file.suffix == '.jsonl':
            file_type = 'jsonl'
        else:
            file_type = 'text'
        
        try:
            # triplet_pipeline_test.py 실행
            cmd = [
                sys.executable,
                str(triplet_script),
                "--input", str(input_file),
                "--type", file_type,
                "--output", str(output_dir)
            ]
            
            print(f"   실행 명령어: {' '.join(cmd)}")
            
            # 프로세스 실행
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=".",
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   ✅ 성공 ({elapsed_time:.1f}초)")
                success_count += 1
                
                # 간단한 결과 요약 출력
                if output_dir.exists():
                    result_files = list(output_dir.glob("*.json"))
                    print(f"   📁 결과 파일 {len(result_files)}개 생성: {output_dir}")
                
            else:
                print(f"   ❌ 실패 ({elapsed_time:.1f}초)")
                print(f"   오류: {result.stderr}")
                error_count += 1
                
            
        except Exception as e:
            print(f"   💥 예외 발생: {e}")
            error_count += 1
    
    # 최종 결과 요약
    print("\n" + "=" * 80)
    print("🎯 배치 처리 완료!")
    print(f"   전체 파일: {len(input_files)}개")
    print(f"   성공: {success_count}개")
    print(f"   실패: {error_count}개")
    print(f"   성공률: {success_count/len(input_files)*100:.1f}%")
    print(f"   결과 저장 위치: {output_base_dir.absolute()}")

if __name__ == "__main__":
    print("TtalKkac Triplet 파이프라인 배치 처리 시작")
    print("=" * 80)
    batch_process_triplets()