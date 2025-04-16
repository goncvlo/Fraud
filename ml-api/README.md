# Deployment

#### :test_tube: Work

Follow the steps below to set up this project on your local machine.
Open the terminal and run the following lines of code.

```bash
# 1. Navigate into the project folder
>> cd ml-api

# 2. [Optional] Set up and activate virtual environment
>> python -m venv .venv  
>> .venv\Scripts\activate

# 3. Install project dependencies
>> pip install -r requirements.txt

# 4. Click on frontend/index.html
# cd ml-api/app/frontend
# python -m http.server 8080

# 5. Start server
uvicorn app.app:app --reload

```
#### :rocket: Future work

- Enhance frontend UI
- Dockerize everything

```Dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```