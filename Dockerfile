FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=user:user . /app

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]