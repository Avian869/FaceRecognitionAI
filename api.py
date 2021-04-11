from flask import Flask, render_template, request, redirect, url_for
from flask_restful import Resource, Api
import json, codecs
import main

app = Flask(__name__, template_folder='templates')
api = Api(app)

@app.route('/')
def index():
    return render_template('index.html')

class Image(Resource):
    def post(self):
        data = request.files["image_file"]
        if data.filename != '':
            results = main.ProcessImage(data)
            #return render_template('results.html', )
            return results

    def get(self):
        return("Worked")

class Camera(Resource):
    def get(self):
        return('WIP')

api.add_resource(Image, '/Image/')
api.add_resource(Camera, '/Camera/')

if __name__ == '__main__':
    app.run(debug='true')