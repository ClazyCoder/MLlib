# MLlib

### uv 설치

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 의존성 설치

```bash
uv sync
```

## 사용법

```bash
bash entrypoint.sh MODE CONFIG_PATH
```

## Docker

Docker 이미지를 빌드하고 실행:

```bash
docker build -t mllib .
docker run mllib train default.yaml
```

## Config 예시

```yaml
model_config:
  name: ResNet18
  num_classes: 2
  pretrained: true
loss_config:
  name: CELoss
  weight: null
  ignore_index: -100
trainer_config:
  name: ClassificationTrainer
  lr: 0.001
  batch_size: 16
  epochs: 10
  save_dir: ./results
metric_config:
  name: Accuracy
dataset_config:
  name: ClassificationDataset
  train_dataset_path: ./test/data/train.json
  val_dataset_path: ./test/data/val.json
```

src/build_dataset.py으로 학습에 필요한 .json파일 생성.

### build_dataset

uv run src/build_dataset.py --mode [MODE] --data_path [PATH_TO_DATASET] --output_path [PATH_TO_SAVE_JSON]

#### mode

데이터 타입 선택 (image or text(NOT IMPLEMENTED))

#### data_path

데이터가 존재하는 경로. 폴더 형태는 다음과 같아야 함.:
```
root
├classnum
│   └files
├classnum
│  └files
└sub_folder
   └classnum
        └files
...
```

#### output_path

결과 json파일의 저장 경로. EX: ./train.json