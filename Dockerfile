FROM python:3.9

# Install dependencies:
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
ENTRYPOINT ["python", "main.py"]