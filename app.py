from flask import Flask, render_template, request, make_response, Response
from flask_restful import Resource, Api
import main

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
api = Api(app)

@app.route('/video_feed')
def video_feed():
    """Method responsible for processing of a camera stream"""
    return Response(main.process_stream(),
          mimetype='multipart/x-mixed-replace; boundary=frame')

class MainPage(Resource):
    """The class which handles the main page of the web page"""
    def get(self):
        """This method renders the index.html when opening the web page"""
        return make_response(render_template('index.html'))

    def post(self):
        """This method renders either the image or camera web page depending on button pressed"""
        if "image" in request.form:
            return make_response(render_template('image.html'))
        return make_response(render_template('camera.html'))


class Image(Resource):
    """The class which handles the web page actions responsible for the processing of an image"""
    def get(self):
        """The Get method for the web page actions responsible for images"""
        return make_response(render_template('image.html'))

    def post(self):
        """The Post method for the web page actions responsible for images"""
        data = request.files["image_file"]
        if data.filename != '':
            results = main.process_image(data)
            headers = {'Content-Type': 'text/html', 'Cache-Control': 'no-store'}
            return make_response(render_template('results.html', images = results),200,headers)

class Camera(Resource):
    """The class which handles the web page actions responsible for a camera stream"""
    def get(self):
        """The Get method for the web page actions responsible for a camera stream"""
        return make_response(render_template('camera.html'))

api.add_resource(MainPage, '/')
api.add_resource(Image, '/Image/')
api.add_resource(Camera, '/Camera/')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
