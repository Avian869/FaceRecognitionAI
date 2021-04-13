from flask import Flask, render_template, request, make_response
from flask_restful import Resource, Api
import main

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
api = Api(app)

@app.route('/')
def index():
    return render_template('index.html')

class Image(Resource):
    def post(self):
        data = request.files["image_file"]
        if data.filename != '':
            results = main.process_image(data)
            headers = {'Content-Type': 'text/html', 'Cache-Control': 'no-store'}
            return make_response(render_template('results.html', images = results),200,headers)
            #return render_template('index.html')
            #return results
        #return "No file selected :("

    def get(self):
        return("Worked")

class Camera(Resource):
    def get(self):
        return('WIP')

api.add_resource(Image, '/Image/')
api.add_resource(Camera, '/Camera/')

if __name__ == '__main__':
    app.run(debug='true')