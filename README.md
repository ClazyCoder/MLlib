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
python src/main.py --mode train --config default.yaml
```

## Docker

Docker 이미지를 빌드하고 실행:

```bash
docker build -t mllib .
docker run mllib train default.yaml
```