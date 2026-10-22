# 장애인콜택시 전날 탑승시간 예측 안내

교통문제 해결 데이터 분석 공모전의 보고서에 대응하는 분석 코드입니다. 지역·요일·시간·과거 기상예보·최근 운행 패턴으로 접수부터 승차까지의 대기를 예측하고, 전날 계획에 쓸 수 있는 탑승시간 범위의 포함률·폭·적용 비중을 검증합니다.

## 보고서의 핵심 결과

2025년 하반기 연구 대상 57,291건에서 AI의 90% 목표 범위가 45분 이하인 22,495건(39.26%)의 실제 포함률은 90.71%, 평균 폭은 35.89분이었습니다. 상반기 기준으로 고정한 15개 구·시간 조건 전체에 같은 폭 기준을 적용하면 하반기 8,489건(14.82%), 포함률 91.45%였습니다.

비중의 분모는 두 대표 예보 격자에 연결된 47개 동의 완료 기록입니다. 서울 전체 동일 기준 하반기 완료 511,443건 중 확인된 비중은 4.40%이며 나머지 지역은 미평가입니다. 취소·미탑승, 실제 일정 준수, 기존 앱 표시값 대비 우월성은 검증하지 않았습니다. 2025년은 이전에 탐색한 내부 개발 검증 기간입니다.

## 설치

Python 3.12 환경에서 실행합니다. 학습에 사용한 라이브러리 버전은 requirements.txt에 고정했습니다.

```text
python -m venv .venv
python -m pip install -r requirements.txt
```

명령 실행 전 해당 가상환경을 활성화합니다. 전체 학습은 CPU에서 실행하며 시간은 기기에 따라 달라집니다. 원자료와 중간 파일은 저장소에 포함하지 않습니다.

## 입력 자료

data_manifest.json에 원파일명·SHA-256과 출처를 기록했습니다. DATA_ROOT에는 아래 구조로 자료를 준비합니다.

```text
DATA_ROOT/
  data/raw/
    *탑승내역_20231231.xlsx
    *탑승내역_20241231.csv
    *탑승내역_20251231.csv
    openmeteo_previous_runs_2024_2025/
      jma_msm_ihwa_2024_day2.json
      jma_msm_ihwa_2025_day2.json
      jma_msm_ihwa_2026_day2.json
      jma_msm_sinchon_2024_day2.json
      jma_msm_sinchon_2025_day2.json
      jma_msm_sinchon_2026_day2.json
  analysis/calltaxi/aws/district_hour_weather.csv
```

- 탑승내역: 서울시설공단 / 공공데이터포털 https://www.data.go.kr/data/15115859/fileData.do . 연도별 보관본을 사용하므로 현재 목록만으로 과거 원파일을 모두 받을 수 있다는 뜻은 아닙니다.
- 과거 예보: JMA MSM / Open-Meteo Previous Runs https://open-meteo.com/en/docs/previous-runs-api . fetch_alternative_forecasts.py에 조회 조건과 다운로드·변환 코드가 있습니다. 2026년 파일은 1월 1일 기상 경계 시간만 포함하며 2026년 탑승 분석이 아닙니다. 공개 서비스의 향후 가용성은 별개이므로 결선에는 사용한 응답 파일을 함께 준비합니다.
- 실제 우천 평가: 기상청 AWS https://data.kma.go.kr/data/grnd/selectAwsRltmList.do?pgmNo=56 . 사용한 지역·시간 집계 파일에는 district, time, rain_next 열이 필요합니다. time은 한국시각 기준 시간이며 rain_next는 실제 평가에 사용한 강수값입니다. 기존 분석에서 사용한 집계 파일을 제공해야 동일한 우천 부분표본이 재현됩니다. 이 저장소가 AWS 원문부터 해당 집계 파일을 새로 만드는 절차까지 포함한 것은 아닙니다.
- 지역·예보 연결: analysis/calltaxi_operations_scope/improved_day_ahead/stable_seoul_grid_map.csv . 기상청 공개 행정동·좌표 자료를 기반으로 기존 분석에서 확정한 연결표이며 경계 재구축 자료가 아닙니다.

위 입력 자료는 결선 실행 패키지에서 코드와 별도로 제공·정리해야 합니다. 코드만 내려받으면 원자료 없이 학습할 수 있다는 뜻은 아닙니다.

## 실행

DATA_ROOT는 위 구조의 실제 경로로 바꿉니다. 저장소 폴더에서 실행합니다.

```text
python run.py check --data-root DATA_ROOT
python run.py reproduce --data-root DATA_ROOT
```

전체 순서는 원자료 정제 → 과거 예보 연결·최근 실적 특성 구성 → 2024년 설정 선택 → 월별 학습·순차 보정 → 양측 구간 검증 → 지역·시간 적용 범위 → 결과 검증입니다. 기존 결과를 정답으로 넣어 학습하지 않습니다. 중간 결과를 캐시하므로 원자료·설정이 바뀌면 별도 복사본의 출력 폴더에서 다시 실행해야 합니다.

결과는 analysis/calltaxi_operations_scope/multi_region_wait/annual_alternative_weather/ 아래에 생성됩니다. 핵심 출력은 boarding_intervals/short_interval_scope/gate_summary.csv, frozen_candidate_cells.csv, frozen_candidate_summary.csv입니다. results/expected/에는 보고서와 대조할 집계 결과만 포함했습니다.

## 기존 작업 자료를 이용한 실행 확인

이미 계산된 원 프로젝트 자료가 있으면 다음 검증을 실행할 수 있습니다. DATA_ROOT에는 원 프로젝트의 data 및 analysis 폴더가 모두 있어야 합니다.

```text
python run.py verify-reference --data-root DATA_ROOT
python run.py smoke-fit --data-root DATA_ROOT
```

verify-reference는 공개용 소스로 113,776건의 적용 조건·분모·상반기 선택·하반기 확인을 다시 계산하고 보고서 집계와 대조합니다. smoke-fit은 2025년 1월의 5·10·95분위 모형을 새로 학습해 기존 예측과 비교합니다. 두 검증이 전체 기간의 전처리·학습을 처음부터 다시 실행한 것과 같지는 않습니다. 실행 결과는 validation/에 저장합니다.

## 코드 구성과 검증 범위

- prepare_multiyear_trips.py: 연도별 원자료 정제, 접수→승차 목표값.
- fetch_alternative_forecasts.py: 당시 사용한 JMA 예보 요청·변환.
- compare_annual_alternative_weather.py: 월별 예측과 과거 잔차 보정.
- compare_boarding_intervals.py: 80%·90% 목표 양측 범위.
- analyze_short_interval_scope.py: 30·45분 적용 규칙, 상반기 후보 선택·하반기 확인.
- verify_*.py: 시간 마감, 동일 표본, 범위 순서, 지표·선택 조건 재산출.
- *_PROTOCOL.md: 분석 설계와 한계. source_provenance.json: 원 코드와 공개용 코드의 대응. 공개용으로 공통 import를 분리했으며 수치 계산 코드는 유지했습니다.

현재 실행 확인 내역은 VALIDATION.md를 참조하세요. 결선에서는 제출 시점의 소스·입력 자료·라이브러리 환경을 고정해 전체 재실행을 확인해야 합니다. 코드 게시 및 접근 확인 상태도 따로 관리합니다.

## 보고서 결과 이미지 재생성

전체 재실행 이후 집계 결과로 제출 보고서의 적용 비중·포함률 이미지를 생성할 수 있습니다.

```text
python -m pip install -r requirements-figures.txt
python render_results.py
```

Windows 맑은 고딕을 기본으로 사용합니다. 다른 환경에서는 한국어 TrueType 글꼴의 경로를 --font 및 --bold-font로 지정합니다. 원본 글꼴 파일은 저장소에 포함하지 않습니다.
