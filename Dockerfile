FROM pytorch/pytorch:12.1.1-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["bash", "entrypoint.sh"]