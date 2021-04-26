from flask import Flask, render_template, request, make_response, Response
from flask_restful import Resource, Api
import main

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
api = Api(app)

@app.route('/video_feed')
def video_feed():
    return Response(main.process_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

class main_page(Resource):
    def index(self):
        return make_response(render_template('index.html'))

    def get(self):
        return make_response(render_template('index.html'))

    def post(self):
        if "image" in request.form:
            return make_response(render_template('image.html'))
        return make_response(render_template('camera.html'))


class Image(Resource):
    def get(self):
        return make_response(render_template('image.html'))

    def post(self):
        data = request.files["image_file"]
        if data.filename != '':
            results = main.process_image(data)
            headers = {'Content-Type': 'text/html', 'Cache-Control': 'no-store'}
            return make_response(render_template('results.html', images = results),200,headers)

class Camera(Resource):
    def get(self):
        return make_response(render_template('camera.html'))


api.add_resource(main_page, '/')
api.add_resource(Image, '/Image/')
api.add_resource(Camera, '/Camera/')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
