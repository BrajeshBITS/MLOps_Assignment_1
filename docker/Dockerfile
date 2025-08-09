FROM python:3.9-slim

WORKDIR /app

RUN useradd --create-home appuser
USER appuser

COPY --chown=appuser:appuser ./requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser:appuser ./src ./src

EXPOSE 8000

ENV PATH="/home/appuser/.local/bin:${PATH}"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]