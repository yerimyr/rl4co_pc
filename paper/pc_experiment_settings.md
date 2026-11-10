# PC 실험 설정 및 하이퍼파라미터 정리

이 문서는 2026-10-24 현재 저장소의 실제 코드, Hydra 실행 설정, 데이터 파일 및 최종 비교 스크립트를 기준으로 작성하였다. 논문 본문에는 표 1~4를 사용하고, 구현 세부값과 소프트웨어 버전은 부록 또는 재현성 자료로 제시하는 구성을 권장한다.

## 표 1. PC 문제 환경 및 인스턴스 생성 설정

| 구분 | 설정값 | 설명 |
|---|---:|---|
| 문제 | Part Consolidation (PC) | 부품을 실행 가능한 그룹으로 순차 구성 |
| 행동 공간 | `0=SEP`, `1..N=부품 선택` | SEP는 현재 그룹을 종료하고 새 그룹을 시작 |
| 최소 그룹 크기 | 1 | 빈 그룹 종료는 허용하지 않음 |
| 학습 크기 범위 | `N=10~20`, `21~30`, `31~40` | curriculum stage별 범위 |
| modularity 해상도 계수 | `gamma={0, 0.25, 0.5, 0.75, 1.0}` | 각 gamma마다 별도 정책 학습 |
| material type 상한 | 2 | 인스턴스별 실제 type 수를 `{1,2}`에서 균등 추출한 뒤 각 부품의 type을 균등 추출 |
| maintenance-high 확률 | `p ~ U(0.01, 0.02)` | 인스턴스마다 확률을 먼저 추출한 뒤 Bernoulli 생성 |
| standard-part 확률 | `p ~ U(0.01, 0.02)` | standard 부품은 비단일 그룹에 포함될 수 없음 |
| relative-motion 확률 | `p ~ U(0.01, 0.02)` | 물리적으로 연결된 부품 쌍에 적용 |
| topology mode | mixed | 6개 topology에서 가중 추출 |
| topology 가중치 | chain 0.10, star 0.10, tree 0.10, two-module bridge 0.35, dense-clustered 0.60, sparse-random 0.10 | 합이 1이 아니므로 내부에서 정규화됨 |
| 정규화된 topology 확률 | 7.41%, 7.41%, 7.41%, 25.93%, 44.44%, 7.41% | 위 순서와 동일 |
| 관계 강도 `W_ij` | `(공통 이웃 수 + U(0,1))/(N-1)` | 연결된 쌍에만 양수, 대칭 행렬 |
| 노드 특징 차원 | 4 | material one-hot 2 + maintenance 1 + standard 1 |
| 엣지 특징 차원 | 4 | material variation, maintenance difference, relative motion, 관계 강도 `W` |
| 실행 가능성 제약 | standard-part 격리, material/maintenance/relative-motion 충돌 금지, 그룹 연결성 | 정책 action mask와 모든 비교 알고리즘에 공통 적용 |
| 목적함수 scale | 1000 | 학습 reward 및 평가 score에 동일 적용 |

목적함수는 다음과 같다.

\[
\mathrm{score}=1000\left[
\frac{1}{2m}\sum_{c}\sum_{i,j\in c}W_{ij}
-\gamma\frac{1}{(2m)^2}\sum_c\left(\sum_{i\in c}k_i\right)^2
\right],
\quad k_i=\sum_j W_{ij},\quad 2m=\sum_i k_i.
\]

코드에 기록되는 `Q_expected`는 gamma를 곱하기 전 값이 아니라 **이미 gamma가 곱해진 penalty 항**이다. 따라서 코드상의 관계는 `Q_gamma = Q_observed - Q_expected`이다.

## 표 2. RL-based NCO 모델 및 학습 하이퍼파라미터

| 구분 | 설정값 | 비고 |
|---|---:|---|
| 학습 알고리즘 | REINFORCE | policy-gradient |
| baseline | rollout baseline | 1 epoch exponential warm-up, `beta=0.8`, 갱신 검정 `alpha=0.05` |
| 정책 구조 | Attention Model + custom edge-aware encoder | RL4CO Attention Model 기반 |
| 정책 파라미터 수 | 989,699 | checkpoint로부터 확인한 trainable policy parameter 수 |
| embedding 차원 | 128 | 노드·엣지·decoder 공통 |
| 초기 노드 투영 | 256 -> 128 | 실제 4차원 노드 특징을 256차원까지 zero-padding 후 선형 투영 |
| encoder layer 수 | 3 | edge-aware message-passing layer |
| attention head 수 | 8 | Attention Model decoder |
| encoder aggregation | learned weighted aggregation | 이웃 message의 학습형 가중합 |
| encoder FFN | `128 -> 512 -> 128` | 각 layer의 residual FFN |
| activation | ReLU | message MLP 및 FFN |
| normalization | LayerNorm | 각 message-passing layer에 2회 |
| dropout | 0.0 | encoder 기본값 |
| SEP 처리 | encoder에서 제외 후 별도 learned embedding 결합 | `exclude_sep_from_encoder=true` |
| compatibility message mask | 사용 안 함 | `use_compat_mask=false` |
| physical-edge message mask | 사용 안 함 | `use_message_mask=false`; 모든 유효 부품 쌍 사이에서 message 전달 |
| connection indicator 추가 | 사용 안 함 | `include_connection_feature=false`; 관계 강도 `W`는 입력됨 |
| decoder context | 마지막 선택 token + 현재 열린 그룹의 평균 embedding | 두 벡터를 결합하여 context 생성 |
| softmax temperature | 1.0 | Attention Model 기본값 |
| tanh clipping | 10.0 | Attention Model 기본값 |
| train decoding | sampling | 확률적으로 해 생성 |
| validation/test decoding | greedy | 가장 높은 확률의 행동 선택 |
| entropy regularization | 없음 | entropy는 모니터링만 하며 loss에 직접 가산하지 않음 |
| batch size | 128 | train/validation/test loader 설정 |
| epoch당 학습 인스턴스 | 10,000 | 매 epoch 종료 후 새로 생성 |
| optimizer | Adam | PyTorch 기본 `betas=(0.9,0.999)`, `eps=1e-8` |
| 기본 learning rate | `1e-4` | gamma 0, 0.25, 0.5 및 모든 gamma의 Stage 1 |
| high-gamma retry learning rate | `1e-5` | 최종 비교에서 gamma 0.75와 1.0의 Stage 2~3에 사용 |
| weight decay | `1e-6` | Adam L2 penalty |
| learning-rate scheduler | 없음 | stage 내 고정 learning rate |
| gradient clipping | 1.0 | global norm 기준 Lightning 설정 |
| epoch 수 | stage당 100 | 총 3 stages |
| validation 주기 | 매 epoch | `check_val_every_n_epoch=1` |
| checkpoint 기준 | 최대 `val/reward` | best 1개 + `last.ckpt` 저장 |
| 계산 정밀도 | `16-mixed` | mixed precision |
| matrix-multiplication precision | medium | PyTorch float32 matmul 설정 |
| training seed | 1234 | Python/NumPy/PyTorch 및 worker seed |
| deterministic mode | false | 완전 결정론은 강제하지 않음 |
| model compilation | false | `torch.compile` 미사용 |

## 표 3. Curriculum learning 구성

| Stage | 학습 데이터 구성 | Epoch | 초기화 | validation 파일 | test 파일 |
|---|---|---:|---|---|---|
| Stage 1 | `N=10~20` 100% | 100 | 무작위 초기화 | 고정 `N=20`, seed 4321 | 고정 `N=20`, seed 1234 |
| Stage 2 | `N=10~20` 20% + `N=21~30` 80% | 100 | Stage 1 best policy weights | 고정 `N=30`, seed 4321 | 고정 `N=30`, seed 1234 |
| Stage 3 | `N=10~20` 10% + `N=21~30` 10% + `N=31~40` 80% | 100 | Stage 2 best policy weights | 고정 `N=40`, seed 4321 | 고정 `N=40`, seed 1234 |

- 각 범위가 선택된 뒤, 범위 안의 `N`은 inclusive discrete uniform distribution으로 추출한다.
- Stage 전환 시 checkpoint의 policy weights를 불러오지만 optimizer state와 epoch counter는 초기화한다.
- replay 비율은 Stage 2에서 20%, Stage 3에서 총 20%이며, 이전 두 범위가 각각 10%를 차지한다.
- gamma 0.75와 1.0은 Stage 1의 기존 best checkpoint를 재사용하고 Stage 2~3만 learning rate `1e-5`로 재학습하였다.

### 최종 비교에 사용되는 checkpoint

| gamma | Stage 1 LR | Stage 2~3 LR | 최종 Stage 3 best checkpoint | 출처 |
|---:|---:|---:|---|---|
| 0 | `1e-4` | `1e-4` | `epoch_069.ckpt` | original curriculum run |
| 0.25 | `1e-4` | `1e-4` | `epoch_062.ckpt` | original curriculum run |
| 0.5 | `1e-4` | `1e-4` | `epoch_088.ckpt` | original curriculum run |
| 0.75 | `1e-4` | `1e-5` | `epoch_098.ckpt` | high-gamma retry run |
| 1.0 | `1e-4` | `1e-5` | `epoch_087.ckpt` | high-gamma retry run |

## 표 4. 비교 알고리즘 및 최종 평가 설정

| 구분 | CP-SAT | CPCCD | RL-based NCO |
|---|---|---|---|
| 실행 장치 | CPU | CPU | CUDA GPU |
| 해 생성 | exact/anytime integer optimization | community detection 후 제약 기반 greedy merge | learned autoregressive greedy decoding |
| 시간 제한 | 인스턴스당 300 s | 없음 | 없음 |
| worker 수 | 1 | 1 process | batch inference |
| 목적계수 정수화 | `1,000,000` 배 scaling 후 반올림 | 해당 없음 | 해당 없음 |
| 주요 설정 | 연결성 flow constraints | `alpha=None`, R3 retry 최대 10 | batch size 100, greedy decode |
| gamma 사용 | 목적함수에 직접 반영 | 생성된 grouping은 gamma와 무관하고 평가 score에만 gamma 적용 | gamma별 별도 학습 checkpoint |

최종 curriculum 비교의 공통 평가 설정은 다음과 같다.

| 항목 | 설정값 |
|---|---:|
| test 범위 | `N=10~20`, `N=21~30`, `N=31~40` |
| 범위별 test 인스턴스 | 100 |
| test seed | 1234 |
| gamma | 0, 0.25, 0.5, 0.75, 1.0 |
| 표시 통계 | 평균 score 및 표준편차 (`mean +/- 1 SD`) |
| 전체 검정 | Friedman test |
| 사후 쌍대 검정 | paired two-sided Wilcoxon signed-rank test (`zero_method=wilcox`) |
| 다중비교 보정 | 각 `범위 x gamma` 조건 내부 Holm correction; 전체 45개 비교의 global Holm도 별도 계산 |
| 유의수준 | `alpha=0.05` |
| 효과크기 | Kendall's W, rank-biserial correlation |

## 표 5. 별도 GA 진단 실험 설정

| 항목 | 설정값 |
|---|---:|
| `N` | 20, 30, 40 |
| gamma | 0, 0.5, 1.0 |
| 인스턴스 수 | 시나리오당 10 |
| test seed | 1234 |
| population size | 100 |
| generations | 3,000 |
| elite size | 0 |
| tournament size | 2 |
| crossover rate | 0.9 |
| mutation rate | 0.3 |
| strong mutation probability | 0.5 |
| strong mutation steps | 5 |
| initial new-group bias | 0.6 |
| initial diverse fraction | 0.7 |
| post-merge repair | false |

## 표 6. 실행 환경

| 구분 | 확인된 값 |
|---|---|
| CPU | Intel Core i5-13600K, 14 cores / 20 logical processors |
| RAM | 31.8 GiB |
| GPU | NVIDIA GeForce RTX 4070, 12,282 MiB |
| OS | Windows NT 10.0 build 26200 (`Windows 10 Pro` registry 표기; 논문 제출 전 Windows 설정 화면의 제품명 재확인 권장) |
| Python | 3.12.13 |
| RL4CO | 0.6.0 기반 customized local fork |
| PyTorch | 2.12.1+cu126 |
| CUDA runtime | 12.6 |
| cuDNN | 9.10.2 |
| Lightning | 2.6.5 |
| OR-Tools | 9.15.6755 |
| TorchRL / TensorDict | 0.13.3 / 0.13.0 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| pandas / NetworkX | 3.0.3 / 3.6.1 |

## 코드 검증 과정에서 확인된 주의사항

1. `configs/experiment/pc/am_pc_edge.yaml`에는 `val_data_size=100`, `test_data_size=100`으로 기록되어 있지만, 고정 `.npz` 파일을 읽을 때 `batch_size` 인자를 사용해 자르지 않는다. 현재 `pc20/30/40_newdist_val/test` 파일에는 각각 1,000개가 들어 있으므로 **학습 중 validation과 학습 종료 직후 test에는 실제로 1,000개 전체가 사용된다.** 논문에는 실제값 1,000을 기재하는 것이 맞다. 별도의 최종 curriculum range 비교는 범위당 100개이다.
2. 환경 YAML의 기본 gamma는 0.3이지만, 최종 실험은 실행 script override로 `{0, 0.25, 0.5, 0.75, 1.0}`을 사용한다. 논문에는 0.3이 아니라 실험 gamma 집합을 적어야 한다.
3. trainer YAML의 기본 validation 주기는 5 epoch지만 curriculum script가 1 epoch로 덮어쓴다. 실제값은 매 epoch이다.
4. gamma 0.75와 1.0의 최종 모델만 Stage 2~3 learning rate가 `1e-5`이다. 모든 gamma에 같은 학습 설정을 적용했다고 쓰면 사실과 다르므로, 이를 gamma별 tuning 또는 안정화 재학습으로 명시해야 한다.
5. CP-SAT의 `OPTIMAL`은 `coefficient_scale=1,000,000`으로 정수화한 모델에 대한 optimal 상태다. 원래 부동소수점 목적함수와의 반올림 차이가 이론적으로 존재하므로 논문에서는 "scaled integer CP-SAT formulation의 최적해"로 표현하는 것이 가장 정확하다.
6. 저장소는 RL4CO 0.6.0의 customized fork이며 현재 다수의 로컬 변경이 commit되지 않은 상태다. 논문 재현성을 위해 제출 전 현재 코드를 별도 commit/tag로 고정해야 한다. 현재 기반 commit은 `f2124f3c6c934510346223182cd38c0e0c556764`이다.
7. `logs/train/curriculum_retry_gamma0p5_lr1e-5/2026-10-24_15-48-51`에는 현재 Stage 2 설정만 있고 완료된 Stage 3 checkpoint가 없다. 따라서 이 retry는 최종 모델 표와 최종 비교 결과에 포함하지 않았으며, gamma 0.5의 현재 최종 모델은 original curriculum의 `epoch_088.ckpt`이다.

## 논문 본문에 넣기 좋은 간결한 서술

본 연구의 RL-based NCO는 RL4CO 0.6.0의 Attention Model을 기반으로 구현하였으며, 부품 간 관계 특징을 처리하는 3-layer edge-aware message-passing encoder와 PC 그룹 생성 상태를 반영하는 decoder context를 적용하였다. 정책은 128차원 embedding과 8-head attention을 사용하고, rollout baseline을 사용하는 REINFORCE로 학습하였다. 각 curriculum stage에서는 epoch당 10,000개의 신규 인스턴스를 batch size 128로 100 epochs 학습하였으며, Adam optimizer, weight decay `1e-6`, gradient clipping 1.0 및 mixed-precision을 적용하였다. Curriculum은 `N=10~20`, `N=21~30`, `N=31~40`의 세 단계로 구성하였고, 이전 범위의 forgetting을 줄이기 위해 Stage 2와 Stage 3에 각각 20%의 replay 데이터를 포함하였다. 최종 평가는 각 범위와 gamma 조합마다 seed 1234로 생성한 100개의 동일 인스턴스에 대해 CP-SAT, CPCCD 및 RL-based NCO를 비교하였다.
