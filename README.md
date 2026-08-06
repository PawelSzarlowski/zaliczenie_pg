
created on the basis of https://fastapi.tiangolo.com/tutorial/

\#1. run development mode: **fastapi dev main.py**

\#2. run production mode: **fastapi run main.py**

\#3. Interactive API docs
Now go to http://127.0.0.1:8000/docs.
You will see the automatic interactive API documentation (provided by Swagger UI):

\#4. Alternative API docs¶
And now, go to http://127.0.0.1:8000/redoc.

You will see the alternative automatic documentation (provided by ReDoc):

\#5. FastAPI automatically generates a JSON (schema) with the descriptions of all your API.

You can see it directly at: http://127.0.0.1:8000/openapi.json.

\#6. Call API (ex. postman)
post -> http://127.0.0.1:8000/joboffersjson

body:

{
  "job_desc": "Szukam pracy jak Java developer, mieszkam w Gdańsku, więc w tym mieście chciałbym znaleźć jakąś posadę. 
  Moje doświadczenie to około 15 lat jako programista aplikacji webowych i desktopowych. Oczekiwania finansowe to około 15000 zł brutto"
}
