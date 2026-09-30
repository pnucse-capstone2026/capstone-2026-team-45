# 이종 AI 반도체 기반 거대언어모델 추론 최적화를 위한 하드웨어 특성 고려 실행 구조 설계

> 부산대학교 정보컴퓨터공학부 2026 졸업과제 45팀 State Of The Art  
> **팀원:** 이지원 (202355572)  
> **지도교수:** 권용인

본 프로젝트는 NVIDIA GPU와 Rebellions RBLN NPU를 하나의 LLM 추론 경로로 연결하고, **CUDA GPU에서 Prefill을 수행한 뒤 생성된 KV Cache를 NIXL을 통해 전달하여 RBLN NPU에서 compiled Decode를 이어서 수행하는 이종 가속기 기반 Prefill-Decode Disaggregation(PDD) 구조**를 구현한다.

핵심 개발 내용은 단순한 KV Cache 전송에 그치지 않고, 서로 다른 하드웨어와 런타임 사이에서 발생하는 **KV Cache layout, block/slot mapping, source dtype, runtime-private 수치 표현 차이**를 진단하고 변환하여 실제 Decode가 이어질 수 있도록 하는 것이다.

---

### 1. 프로젝트 배경

#### 1.1. 국내외 시장 현황 및 문제점

거대언어모델(LLM)의 활용이 확대되면서 모델 추론을 빠르고 효율적으로 수행하기 위한 AI 가속기와 LLM serving software의 중요성이 높아지고 있다. 기존 LLM 추론 인프라는 NVIDIA GPU를 중심으로 구축되는 경우가 많지만, 최근에는 NPU와 같은 AI 전용 가속기를 함께 활용하여 하드웨어 선택의 폭을 넓히려는 시도가 이루어지고 있다.

LLM 추론은 크게 입력 프롬프트를 처리하여 KV Cache를 생성하는 **Prefill 단계**와, 생성된 KV Cache를 반복적으로 참조하며 후속 토큰을 생성하는 **Decode 단계**로 나뉜다. 두 단계는 연산량과 메모리 접근 특성이 다르기 때문에 하나의 장치에서 모든 과정을 수행하는 방식 외에도, 하드웨어 특성에 따라 각 단계를 서로 다른 장치에 배치하는 Prefill-Decode Disaggregation 구조를 적용할 수 있다.

하지만 서로 다른 종류의 가속기를 연결하는 경우에는 단순히 KV Cache를 송수신하는 것만으로 동일한 추론을 이어갈 수 없다. 송신 측과 수신 측의 메모리 위치, 텐서 축 순서, block/slot mapping, dtype 및 내부 수치 표현이 서로 다를 수 있기 때문이다. 실제 구현 과정에서도 KV Cache 전송과 응답 반환 자체는 성공했지만, Decode 결과가 기준 실행과 달라지는 문제가 발생하였다.

따라서 본 프로젝트는 **GPU에서 생성한 KV Cache를 RBLN NPU가 실제로 사용할 수 있는 위치와 표현으로 전달·변환하는 문제**에 초점을 맞추었다.

#### 1.2. 필요성과 기대효과

본 프로젝트의 필요성과 기대효과는 다음과 같다.

- 서로 다른 종류의 AI 가속기를 하나의 LLM 추론 파이프라인에서 활용할 수 있는 기반을 마련한다.
- Prefill과 Decode를 서로 다른 장치에서 수행하여 단계별 하드웨어 특성에 맞는 역할 분담 가능성을 검토할 수 있다.
- CUDA GPU와 Rebellions RBLN NPU 사이의 실제 KV Cache 호환 문제를 확인하고 해결함으로써 이종 AI 가속기 연동 기술을 검증할 수 있다.
- 기존 vLLM의 KV connector 구조와 RBLN NPU를 연결하여 향후 다양한 모델과 하드웨어 조합으로 확장할 수 있는 기반을 제공한다.
- 국내 AI 반도체를 실제 LLM serving framework에 연동함으로써 활용 가능성을 넓힐 수 있다.

본 프로젝트에서는 성능 우위를 미리 가정하지 않고, 먼저 **서로 다른 가속기 사이에서 KV Cache가 손상 없이 전달되고 올바른 형태로 반영되어 Decode를 이어갈 수 있는 기능적·정확성 기반을 확보하는 것**을 우선 목표로 한다.

---

### 2. 개발 목표

#### 2.1. 목표 및 세부 내용

본 프로젝트의 최종 목표는 **CUDA GPU를 Prefill 장치로, Rebellions RBLN NPU를 Decode 장치로 사용하는 이종 가속기 기반 LLM Prefill-Decode Disaggregation 실행 구조를 설계·구현하고 검증하는 것**이다.

세부 목표는 다음과 같다.

1. CUDA vLLM 인스턴스에서 LLM Prefill 수행
2. Prefill 단계에서 생성된 KV Cache 확보
3. vLLM KV connector와 NIXL을 이용한 KV Cache 전달 경로 구성
4. 송신 및 수신 측 KV Cache의 layout과 block/slot mapping 정합
5. 실제 송신 dtype을 고려한 KV Cache 수치 해석
6. 전달된 KV Cache를 RBLN runtime-private 표현으로 변환
7. 변환된 KV Cache를 RBLN runtime cache에 반영
8. Rebellions RBLN NPU에서 compiled Decode 수행
9. 전송 전·수신 후·runtime update 전후·정답 KV를 구간별로 비교하여 오류 원인 분리
10. 기준 실행과 token, text, logit을 비교하여 생성 정확성 검증

대표 실험 환경에서는 `Qwen/Qwen3-0.6B` 모델을 사용하며, NVIDIA RTX PRO 6000 Blackwell Workstation Edition에서 Prefill을 수행하고 Rebellions RBLN-CA22 NPU에서 Decode를 수행한다.

#### 2.2. 기존 서비스 대비 차별성

기존 `vllm-rbln`은 Rebellions RBLN NPU에서 vLLM을 실행할 수 있도록 하는 하드웨어 플러그인을 제공한다. 또한 vLLM은 Prefill과 Decode를 분리하고 KV Cache를 전달하기 위한 KV connector 구조를 제공한다.

본 프로젝트는 이를 기반으로 다음과 같은 **CUDA GPU와 RBLN NPU 간 이종 실행 경로**를 구성한다.

- **CUDA GPU:** Prefill 수행 및 KV Cache 생성
- **`NixlConnector`:** CUDA 측 KV producer
- **NIXL:** GPU 측 KV Cache를 RBLN 프로세스의 Host 수신 버퍼로 전달
- **`RblnNixlConnector`:** RBLN 측 KV consumer
- **KV Cache 변환:** Host-visible KV를 RBLN runtime이 사용할 수 있는 내부 표현으로 변환
- **RBLN NPU:** 변환된 KV Cache를 사용하여 compiled Decode 수행

특히 본 프로젝트의 차별점은 **KV Cache가 전달되었다는 사실만 확인하는 것이 아니라, 서로 다른 vendor와 runtime 사이에서 실제 추론이 이어지기 위해 필요한 메모리 위치, layout, block/slot mapping, source dtype, runtime-private 표현의 정합을 구분하여 진단하고 처리한다는 점**이다.

#### 2.3. 사회적 가치 도입 계획

본 프로젝트는 다음과 같은 사회적 가치를 지향한다.

- **국내 AI 반도체 활용 확대:** Rebellions RBLN NPU를 실제 LLM serving framework와 연동하여 국내 AI 반도체의 활용 사례를 확장한다.
- **하드웨어 선택권 확대:** 특정 종류의 가속기에 종속되지 않고 서로 다른 AI 가속기를 함께 활용할 수 있는 기술적 기반을 마련한다.
- **오픈소스 기반 재현성 확보:** vLLM, vllm-rbln, NIXL 등 공개된 소프트웨어 생태계를 기반으로 구현 및 검증 과정을 관리한다.
- **향후 자원 활용 효율화 연구 기반 제공:** 서로 다른 특성을 가진 장치를 단계별로 활용하는 구조를 바탕으로 향후 처리량, 지연시간, 전력 및 비용을 함께 고려하는 최적화 연구로 확장할 수 있다.

단, 본 프로젝트의 현재 결과는 이종 실행 경로와 KV Cache 호환성 구현 및 검증에 초점을 두며, GPU-NPU 배치가 단일 장치 대비 성능상 항상 우수하다고 주장하지 않는다.

---

### 3. 시스템 설계

#### 3.1. 시스템 구성도

```text
                         LLM Inference Request
                                  |
                                  v
                    +---------------------------+
                    |      CUDA vLLM Server     |
                    |          Prefill          |
                    |   NVIDIA RTX PRO 6000     |
                    +-------------+-------------+
                                  |
                                  | KV Cache in CUDA VRAM
                                  v
                    +---------------------------+
                    |       NixlConnector       |
                    |       KV Producer         |
                    +-------------+-------------+
                                  |
                                  | NIXL Transfer
                                  v
                    +---------------------------+
                    |   CPU Host DRAM Buffer    |
                    |   in RBLN Decode Process  |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   KV Cache Compatibility  |
                    |                           |
                    | - Layout / stride         |
                    | - Block / slot mapping    |
                    | - Source dtype            |
                    | - Runtime-private format  |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |     RBLN Runtime KV       |
                    |          Cache            |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |      RBLN NPU Server      |
                    |     Compiled Decode       |
                    |     RBLN-CA22 NPU         |
                    +-------------+-------------+
                                  |
                                  v
                         Generated Tokens
```

본 구현은 **Host 경유 경로**를 대상으로 한다. CUDA vLLM 인스턴스에서 Prefill을 수행하면 KV Cache가 GPU VRAM에 생성되고, NIXL을 통해 RBLN Decode 프로세스의 CPU Host DRAM 수신 버퍼로 전달된다. 수신 측에서는 KV Cache의 위치와 수치 표현을 RBLN runtime에 맞게 변환한 뒤 runtime cache에 반영하고, RBLN NPU가 compiled Decode를 수행한다.

CPU는 KV Cache의 수신과 변환을 담당하며 LLM Decode 연산 자체를 대체하지 않는다.

#### 3.2. 사용 기술

| 구분 | 사용 기술 | 용도 |
| --- | --- | --- |
| Language | Python | 실행 스크립트, 검증 및 PDD orchestration |
| LLM Serving | vLLM | CUDA Prefill 및 KV transfer framework |
| RBLN Integration | vllm-rbln | Rebellions NPU에서 vLLM 실행 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | Prefill 수행 |
| NPU | Rebellions RBLN-CA22 | Compiled Decode 수행 |
| Data Transfer | NIXL | Prefill/Decode 사이 KV Cache 전달 |
| Prefill Connector | `NixlConnector` | CUDA 측 KV producer |
| Decode Connector | `RblnNixlConnector` | RBLN 측 KV consumer |
| KV Layout | HND | Prefill/Decode KV Cache layout 정합 |
| KV Conversion | Source-aware conversion | BF16 source 해석 및 RBLN runtime-private 표현 변환 |
| Model | `Qwen/Qwen3-0.6B` | 대표 검증 모델 |
| Environment | Python virtual environment | CUDA Prefill / RBLN Decode 실행 환경 분리 |

현재 `run_pdd_once.py`는 기본적으로 다음 서버를 실행한다.

| 구성 요소 | 기본 포트 | 역할 |
| --- | ---: | --- |
| CUDA Prefill Server | `8100` | Prompt 처리 및 KV Cache 생성 |
| RBLN Decode Server | `8200` | 전달된 KV Cache 기반 Decode |
| Toy Proxy Server | `8192` | Prefill과 Decode 요청 흐름 연결 |
| Prefill NIXL Side Channel | `5559` | Prefill 측 NIXL 통신 |
| Decode NIXL Side Channel | `5659` | Decode 측 NIXL 통신 |

---

### 4. 개발 결과

#### 4.1. 전체 시스템 흐름도

```text
[1] 사용자 추론 요청
        |
        v
[2] CUDA GPU Prefill
    - 입력 prompt 처리
    - KV Cache 생성
        |
        v
[3] NIXL KV Cache Transfer
    - CUDA VRAM -> RBLN process CPU Host DRAM
        |
        v
[4] RBLN 측 KV Cache 수신
    - request ID 대응
    - block / slot mapping 확인
    - position / context 정보 대응
        |
        v
[5] KV Cache 변환
    - tensor layout / stride 확인
    - 실제 source dtype(BF16) 해석
    - 필요 수치 변환
    - RBLN runtime-private representation 인코딩
        |
        v
[6] RBLN Runtime KV Cache Update
        |
        v
[7] RBLN NPU Compiled Decode
        |
        v
[8] token / text / logit 결과 검증 및 반환
```

KV Cache 전달 및 반영 과정은 다음 지점으로 나누어 검증하였다.

- **A:** CUDA 전송 직전 KV Cache
- **B:** RBLN 수신 직후 KV Cache
- **C:** RBLN runtime update 직전 KV Cache
- **D:** runtime update 후 readback KV Cache
- **T:** RBLN-only Prefill이 생성한 정답 runtime-private KV Cache

이와 같이 전송, runtime 반영, 수치 표현, Decode read path를 구간별로 분리하여 문제가 발생하는 지점을 진단하였다.

#### 4.2. 기능 설명 및 주요 기능 명세서

| 기능 | 입력 | 출력 | 설명 |
| --- | --- | --- | --- |
| CUDA Prefill | Prompt, Model | KV Cache | NVIDIA GPU에서 입력 token을 처리하고 KV Cache 생성 |
| KV Cache Transfer | GPU KV Cache | Host 수신 KV Cache | NIXL을 이용해 CUDA VRAM의 KV block을 RBLN 프로세스의 CPU Host DRAM으로 전달 |
| Layout 정합 | 송수신 KV tensor | 정렬된 KV tensor | token, head, head dimension의 논리 좌표와 stride를 확인 |
| Block/Slot Mapping | 송신 block ID | RBLN 수신 slot | 송신 논리 block을 수신 측 물리 block/slot에 대응 |
| Source dtype 해석 | Raw KV bits | 올바른 수치 | 실제 CUDA KV dump의 BF16 dtype을 기준으로 raw 값을 해석 |
| Runtime-private 변환 | Host-visible KV | RBLN runtime KV | 수신 값을 RBLN runtime-private 표현으로 인코딩 |
| Runtime KV Update | 변환된 KV Cache | RBLN runtime cache | 레이어별 KV Cache를 compiled runtime에 반영 |
| RBLN Decode | Runtime KV Cache | Generated tokens | RBLN NPU에서 compiled Decode 수행 |
| 구간별 검증 | A/B/C/D/T KV | 비교 결과 | 전송 손상, update/readback, 수치 표현 문제를 분리 진단 |
| 출력 정확성 검증 | Generated token/logit | 기준 실행 비교 결과 | text, token, logit 및 최초 분기 지점을 확인 |

주요 검증 결과는 다음과 같다.

| 검증 항목 | 결과 |
| --- | --- |
| 전송 block raw bit 불일치 | `0` |
| Runtime update 후 readback 최대 절대 오차 | `0.0` |
| 대표 모델의 레이어별 Runtime KV update | `28회` 확인 |
| Python syntax / unit tests | `47 passed` |
| Native scalar parity | `19/19` |
| Synthetic pattern 비교 | `13종 x 262,144개 원소`, raw mismatch `0` |
| 대표 8/64/128-token prompt | 각 대표 사례에서 first token 및 full text 일치 |
| 대표 32-token prompt | Step 6에서 분기, 해당 step의 top-5 overlap `5/5` |

초기에는 layout만 조정하는 방식과 경험적 affine 보정을 시도했으나 전체 prompt에 일반화되지 않았다. 이후 정답 KV 주입과 A/B/C/D/T 구간별 진단을 통해 전송 자체와 Decode read path를 분리하여 확인했고, 실제 CUDA KV source가 BF16임을 확인한 뒤 source-aware 변환 경로를 구현하였다.

현재 결과는 KV Cache의 전송·반영·표현 호환성을 검증한 결과이며, 최종 TTFT, ITL, 처리량 측정에 근거한 GPU-NPU 배치의 성능 우위 또는 최적성을 주장하지 않는다.

#### 4.3. 디렉토리 구조

프로젝트의 주요 파일 및 디렉토리는 다음과 같다.

```text
capstone-2026-team-45/
├── run_pdd_once.py
├── run_gpu_pdd_once.py
├── scripts/
│   └── setup_pdd_envs.sh
├── requirements/
│   ├── pdd-prefill.txt
│   └── pdd-decode.txt
├── vllm_rbln/
│   └── ...
├── tests/
│   └── ...
├── examples/
│   └── ...
├── docs/
│   └── ...
├── pyproject.toml
├── LICENSE
└── README.md
```

주요 실행 및 설정 파일은 다음과 같다.

- `run_pdd_once.py`
  - CUDA Prefill server 실행
  - RBLN Decode server 실행
  - Toy proxy server 실행
  - NIXL KV transfer 설정
  - End-to-End 요청 실행 및 로그 저장
- `run_gpu_pdd_once.py`
  - GPU 기반 Prefill/Decode 경로의 비교 및 검증에 사용
- `scripts/setup_pdd_envs.sh`
  - Prefill과 Decode용 Python virtual environment 구성
- `requirements/pdd-prefill.txt`
  - CUDA Prefill 환경 의존성
- `requirements/pdd-decode.txt`
  - RBLN Decode 환경 의존성

#### 4.4. 산업체 멘토링 의견 및 반영 사항

2026년 7월 26일 **Rebellions 안민욱 박사**의 산업체 자문을 통해 구현 및 검증 방향을 점검하였다. 자문에서는 현재의 계층별 원인 분리 접근을 긍정적으로 평가하는 한편, 정확성 검증의 범위, 장문·동시 요청, 성능 비교 방법, 기술적 기여의 명확화, 재현성 측면의 보완이 필요하다는 의견을 받았다.

자문 의견과 반영 내용은 다음과 같다.

| 자문 요지 | 반영 내용 및 대응 |
| --- | --- |
| 정확성 주장 보강 | 대표 사례의 성공과 전체 정확성을 구분하였다. BF16 source-aware 변환을 대상으로 pattern 및 특수값을 포함한 검사를 수행하고, unit test, native scalar parity, synthetic pattern 비교와 실제 prompt 기반 검증 결과를 구분하여 기록하였다. |
| 장문·동시 요청 선행 | 단일 요청 성공을 전체 조건으로 일반화하지 않도록 범위를 명시하였다. 장문 다중 block 및 동시 요청에서 block/slot이 올바르게 분리되는지 검증한 뒤 성능 평가로 확장하는 순서를 후속 계획에 반영하였다. |
| 성능 비교 설계 | CUDA-only, RBLN-only, GPU Prefill-NPU Decode 등의 실행 구성을 비교 대상으로 두고, TTFT·TPOT뿐 아니라 지연 목표를 만족하는 처리율(goodput), KV Cache 전송·변환·runtime update 비용을 함께 측정하는 방향으로 평가 계획을 보완하였다. 현재 최종보고서에서는 성능 우위를 근거 없이 단정하지 않는다. |
| 기여 및 관련 기술 정리 | source dtype 오해석과 RBLN runtime-private encoding 불일치를 서로 다른 문제로 구분하였다. 메모리 영역, layout, block/slot mapping, 원소 표현 및 인덱스 정보의 정합을 명시하고, 구현에 직접 활용한 vLLM과 NIXL을 참고 기술로 정리하였다. |
| 일반화 및 재현성 확보 | 더 큰 모델과 입력 길이로 검증 범위를 확장하고, 양측 framework·SDK·compiler·commit 정보를 기록하도록 정리하였다. 공식 KV import API로의 대체 가능성, 종료 오류, 결과 파일 완결성, warm-up 분리 및 반복 측정을 후속 점검 항목으로 두었다. |

---

### 5. 설치 및 실행 방법

#### 5.1. 설치절차 및 실행 방법

##### 1) Repository Clone

```bash
git clone https://github.com/pnucse-capstone2026/capstone-2026-team-45.git
cd capstone-2026-team-45
```

##### 2) 사전 요구사항

- Linux 환경
- Python `3.10` ~ `3.13`
- CUDA를 사용할 수 있는 NVIDIA GPU 및 정상 설치된 NVIDIA driver
- Rebellions RBLN NPU 및 정상 설치된 RBLN driver/runtime
- 시스템 Python 환경에 설치된 `rebel-compiler`
- 필요한 Python package 및 model을 설치·다운로드할 수 있는 환경

##### 3) Prefill / Decode 실행 환경 구성

프로젝트는 CUDA Prefill과 RBLN Decode를 각각 별도의 virtual environment에서 실행한다.

```bash
bash scripts/setup_pdd_envs.sh
```

기본적으로 다음 환경이 생성된다.

```text
.venvs/
├── pdd-prefill/
└── pdd-decode/
```

`run_pdd_once.py`는 필요한 virtual environment가 존재하지 않을 경우 설정 스크립트를 자동으로 실행할 수 있다.

##### 4) 실행 전 설정 확인

실제 server를 실행하기 전에 경로와 실행 명령을 검증할 수 있다.

```bash
python3 run_pdd_once.py --check-config
```

정상적으로 설정되어 있으면 server를 시작하지 않고 configuration 검사 결과만 출력한다.

##### 5) End-to-End PDD 실행

```bash
python3 run_pdd_once.py
```

기본 실행 구성은 다음과 같다.

```text
Model               : Qwen/Qwen3-0.6B
Prefill Device      : NVIDIA CUDA GPU
Decode Device       : Rebellions RBLN NPU
Prefill Port        : 8100
Decode Port         : 8200
Proxy Port          : 8192
Prefill NIXL Port   : 5559
Decode NIXL Port    : 5659
```

사용 가능한 세부 실행 옵션은 다음 명령으로 확인할 수 있다.

```bash
python3 run_pdd_once.py --help
```

#### 5.2. 오류 발생 시 해결 방법

##### Python 버전 오류

환경 구성 스크립트는 Python `3.10`, `3.11`, `3.12`, `3.13`을 지원한다.

```bash
python3 --version
```

지원되지 않는 버전이 기본 Python으로 설정되어 있다면 `PDD_BOOTSTRAP_PYTHON` 환경변수로 사용할 Python interpreter를 지정할 수 있다.

##### CUDA 장치 인식 오류

다음 명령으로 NVIDIA GPU와 driver 상태를 확인한다.

```bash
nvidia-smi
```

##### RBLN SDK / Runtime 인식 오류

환경 구성 스크립트는 시스템 Python에 설치된 `rebel-compiler`를 확인한 뒤 Decode virtual environment에 필요한 SDK 경로를 연결한다. `rebel-compiler`가 인식되지 않는다면 시스템 Python 및 RBLN SDK 설치 상태를 먼저 확인한다.

##### Port 충돌

기본적으로 다음 포트를 사용한다.

```text
8100
8200
8192
5559
5659
```

`run_pdd_once.py`는 관리 대상 포트를 점유한 이전 vLLM/Proxy process를 정리하는 기능을 포함한다. 다른 서비스가 동일한 포트를 사용 중이라면 충돌 원인을 먼저 확인한다.

##### KV Cache 전달은 되지만 Decode 결과가 다른 경우

전송 성공과 생성 정확성은 별개의 문제이므로 다음 순서로 점검한다.

1. CUDA 전송 직전과 RBLN 수신 직후의 raw KV bit 비교
2. Runtime update 직전과 update 후 readback 값 비교
3. 송신 KV Cache의 실제 source dtype 확인
4. token/head/head-dimension의 layout 및 stride 확인
5. 송신 block과 수신 physical block/slot mapping 확인
6. RBLN runtime-private format 변환 경로 확인
7. RBLN-only Prefill이 생성한 정답 KV를 주입하여 Decode read path 분리 검증
8. 기준 실행과 token, logit 및 최초 분기 step 비교

현재 Decode 측의 핵심 KV transfer 설정은 다음과 같다.

```text
kv_connector                  = RblnNixlConnector
kv_role                       = kv_consumer
kv_buffer_device              = cpu
rbln_external_kv_format       = host_visible_hnd_to_runtime_private
rbln_external_kv_source_dtype = bfloat16
```

---

### 6. 소개 자료 및 시연 영상

#### 6.1. 프로젝트 소개 자료

본 프로젝트의 개발 배경, 시스템 구조, KV Cache 변환 과정, 검증 결과 및 산업체 자문 반영 내용은 다음 최종 기술보고서와 발표 포스터를 통해 정리하였다.

**최종 기술보고서**

- 제목: **이종 AI 반도체 기반 거대언어모델 추론 최적화를 위한 하드웨어 특성 고려 실행 구조 설계**
- 기관: 부산대학교 정보컴퓨터공학부
- 문서: **Technical Report 2026-09**
- 저자: **이지원 (202355572)**
- 지도교수: **권용인**

최종 기술보고서에서는 다음 내용을 다룬다.

- CUDA GPU Prefill과 RBLN NPU compiled Decode의 분리 실행 구조
- NIXL을 이용한 Host 경유 KV Cache 전달 경로
- KV Cache layout 및 block/slot mapping 정합
- BF16 source dtype 분석 및 RBLN runtime-private 표현 변환
- A/B/C/D/T 구간별 KV Cache 검증 방법
- 생성 정확성 검증 및 오류 원인 분석
- 산업체 자문 의견 반영 및 향후 연구 방향

또한 프로젝트 결과를 요약한 **발표 포스터**를 제작하여 GPU-NPU 분리 실행 구조와 KV Cache 변환·검증 과정을 시각적으로 정리하였다.

#### 6.2. 시연 영상

프로젝트 시연 영상은 아래 링크에서 확인할 수 있다.

[![이종 AI 반도체 기반 LLM Prefill-Decode Disaggregation 시연 영상](https://img.youtube.com/vi/zCDZ8504F0E/0.jpg)](https://www.youtube.com/watch?v=zCDZ8504F0E)

- YouTube: https://www.youtube.com/watch?v=zCDZ8504F0E

---

### 7. 팀 구성

#### 7.1. 팀원별 소개 및 역할 분담

본 프로젝트는 **45팀 State Of The Art의 이지원(202355572)이 단독 수행**하였다.

| 이름 | 학번 | 담당 업무 |
| --- | --- | --- |
| 이지원 | 202355572 | 연구 목표 및 일정 수립, 관련 기술 조사, GPU/NPU 하드웨어·소프트웨어 환경 구성, CUDA Prefill-RBLN Decode 분리 실행 구조 설계, NIXL connector 및 KV Cache 수신·변환·반영 구현, layout 및 block/slot mapping 분석, source dtype 및 runtime-private 수치 표현 분석, A/B/C/D/T 검증 실험 설계·수행, token/logit 기반 생성 정확성 검증, 로그 분석 및 오류 수정, 최종보고서·발표 포스터·시연 자료 작성 |

주요 결과물은 GPU-NPU 분리 실행 구조, KV Cache 수신·변환·반영 구현, 구간별 정확성 진단 자료, 보고서, 발표 포스터 및 시연 자료이다. 산업체 자문은 개발 수행과 구별되는 외부 검토 및 피드백으로 반영하였다.

#### 7.2. 팀원 별 참여 후기

**이지원**

본 프로젝트를 수행하면서 가장 크게 배운 점은 서로 다른 하드웨어를 연결하는 작업에서는 데이터가 단순히 "전송되었다"는 사실만으로 기능적 정확성이 보장되지 않는다는 점이었다. 초기에는 CUDA GPU에서 생성한 KV Cache를 RBLN 측으로 전달하면 Decode를 이어갈 수 있을 것으로 예상했지만, 실제로는 전송과 응답 반환이 정상적으로 이루어져도 생성 token이 기준 실행과 달라지는 문제가 발생하였다.

문제를 해결하는 과정에서 layout만 조정하거나 경험적인 값 보정을 적용하는 방식으로는 여러 prompt에 일반화되지 않는다는 것을 확인하였다. 이후 문제를 한 번에 해결하려 하기보다 KV Cache 전달 과정을 A/B/C/D/T 지점으로 나누고, 전송 전후의 raw bit, runtime update 전후의 readback, RBLN-only Prefill이 생성한 정답 KV와의 차이를 각각 비교하였다. 이를 통해 전송 손상 여부, cache update 문제, Decode read path 문제, 수치 표현 문제를 단계적으로 분리할 수 있었다.

특히 실제 CUDA KV dump의 source dtype이 BF16이라는 점과, RBLN runtime이 외부에서 보이는 일반적인 tensor 값과 다른 runtime-private 표현을 사용한다는 점을 확인하면서 **메모리 layout, block mapping, dtype, 내부 수치 표현을 각각 독립적으로 확인해야 한다는 것**을 배웠다. 이 과정을 바탕으로 source-aware KV Cache 변환 경로를 구현하고, unit test와 synthetic pattern, 실제 prompt 기반 검증을 통해 구현 결과를 확인하였다.

프로젝트를 단독으로 진행하면서 환경 구성부터 코드 구현, 실험 설계, 로그 분석, 오류 수정, 문서화까지 전체 과정을 직접 수행해야 했기 때문에 한 단계의 문제를 다른 단계의 문제로 오인하지 않도록 실험 조건과 결과를 체계적으로 기록하는 것이 중요했다. 또한 원하는 성능 결과를 먼저 결론으로 두기보다, 현재 확보한 증거가 무엇을 말할 수 있고 무엇까지는 말할 수 없는지를 구분하는 것이 연구 및 개발 결과를 정리하는 데 중요하다는 점을 경험하였다.

현재 프로젝트를 통해 GPU Prefill과 RBLN compiled Decode를 연결하는 실행 구조와 KV Cache 호환성 검증 기반을 마련하였다. 향후에는 장문 및 동시 요청 조건에서 정확성 검증 범위를 넓히고, KV Cache 전송·변환 비용을 포함하여 단일 장치와 이종 실행 구조의 TTFT, TPOT, 처리량 및 goodput을 비교하는 방향으로 확장할 수 있다.

---

### 8. 참고 문헌 및 출처

1. Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, **"DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving,"** Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI), pp. 193-210, 2024.

2. vLLM Project, **"Disaggregated Prefilling,"** vLLM Documentation.  
   https://docs.vllm.ai/en/latest/features/disagg_prefill/

3. NVIDIA / AI-Dynamo, **"NVIDIA Inference Xfer Library (NIXL),"** Official Repository.  
   https://github.com/ai-dynamo/nixl

4. SOTA-PNU, **vllm-rbln**.  
   https://github.com/SOTA-PNU/vllm-rbln

5. RBLN-SW, **vllm-rbln**.  
   https://github.com/RBLN-SW/vllm-rbln

---

## Upstream 및 라이선스 안내

본 프로젝트는 [`SOTA-PNU/vllm-rbln`](https://github.com/SOTA-PNU/vllm-rbln)을 기반으로 개발되었으며, 해당 저장소는 [`RBLN-SW/vllm-rbln`](https://github.com/RBLN-SW/vllm-rbln)에서 파생된 프로젝트이다.

원본 `vllm-rbln` 프로젝트는 **Apache License 2.0**으로 배포된다. 본 저장소에서도 기존 `LICENSE` 파일과 upstream의 저작권 및 라이선스 고지를 유지한다.

---

---

# Original vLLM-RBLN Documentation

# vLLM RBLN Plugin
<div align="center">
<picture>
  <source srcset="assets/vllm-rbln-white.png" media="(prefers-color-scheme: dark)">
  <source srcset="assets/vllm-rbln-black.png" media="(prefers-color-scheme: light)">
  <img src="assets/vllm-rbln-black.png" alt="main-logo" width=90%>
</picture>
[![PyPI version](https://badge.fury.io/py/vllm-rbln.svg)](https://badge.fury.io/py/vllm-rbln)
[![License](https://img.shields.io/github/license/rbln-sw/vllm-rbln)](https://github.com/rbln-sw/vllm-rbln/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](./CODE_OF_CONDUCT.md)
</div>
This repository provides the hardware plugin that enables vLLM on RBLN NPUs, including [ATOM](https://rebellions.ai/rebellions-product/rbln-ca25/) and [REBEL](https://rebellions.ai/rebellions-product/rebel-quad/).
Built on top of [vLLM’s Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html), it allows seamless integration with the vLLM ecosystem and provides high-throughput, low-latency LLM serving on RBLN hardware. Our plugin supports a wide range of popular LLMs and continues to expand to support all features enabled in vLLM, including advanced attention mechanisms.
## 🚀 Getting Started

### 📋 Prerequisites

- `rebel-compiler`
- `optimum-rbln`

### ⚙️ Installation

You can install this project using `pip` or from source.

#### Install via PyPI

##### Using uv
```bash
uv pip install vllm-rbln --extra-index-url https://wheels.vllm.ai/0.22.0/cpu --torch-backend cpu
```

##### Using pip
```bash
pip install vllm-rbln --extra-index-url https://wheels.vllm.ai/0.22.0/cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Or from source
##### Using uv
```bash
git clone https://github.com/rbln-sw/vllm-rbln.git
cd vllm-rbln
uv pip install -e .
```

##### Using pip
```bash
git clone https://github.com/rbln-sw/vllm-rbln.git
cd vllm-rbln
pip install -e . --extra-index-url https://wheels.vllm.ai/0.22.0/cpu --extra-index-url https://download.pytorch.org/whl/cpu
```
### 📚 Documentation

- [Overview & Supported Models](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html)
- [API Tutorial](https://docs.rbln.ai/software/model_serving/vllm_support/tutorial/vllm_llama3-8b.html)


## 🤝 Contributing

We welcome all contributions! Whether it's reporting issues, proposing enhancements, or improving docs—your input helps make the project better.

See our [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.
## 📄 License

This project is licensed under the Apache License 2.0.

See the [LICENSE](./LICENSE) file for more information.

## 📧 Contact

- Join discussions and get answers in our [Developer Community](https://discuss.rebellions.ai/)
- Contact maintainers at [support@rebellions.ai](mailto:support@rebellions.ai)
