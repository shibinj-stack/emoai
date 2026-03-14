FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    tensorflow-cpu==2.15.0 \
    numpy \
    scikit-learn \
    pandas \
    gunicorn

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]
