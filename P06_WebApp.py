# Web Aookucatuib
__author__='Barbu'

from flask import Flask
app = Flask(__name__) # '__main__'

@app.route('/') #www.snipermed.com/api/
def hello_method():
    return "Hello, Barbu. Today is Sep 5 2018. Salutations from Python Flask !!!"

if __name__ == '__main__':
    app.run(port=5000)
    
