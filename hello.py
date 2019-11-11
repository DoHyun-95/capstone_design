from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

if __name__ == "__main__": #파이썬을 실행할때만 아래를 실행하라는 뜻임
    app.run(port=7654)