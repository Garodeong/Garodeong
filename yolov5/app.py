from flask import Flask, render_template, Response, request, redirect, url_for, session
from camera import VideoCamera
# from flask_login import LoginManager, UserMinin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'changeme'

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# users = {'admin': {'password': 'password'}}

# class User(UserMinin):
#     pass

# @login_manager.user_loader
# def user_loader(user_id):
#     if user_id not in users:
#         return None
#     user = User()
#     user.id = user_id
#     return user

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'GET':
#         return render_template('login.html')

#     username = request.form['username']
#     password = request.form['password']

#     if username in users and users[username]['password'] == password:
#         user = User()
#         user.id = username
#         login_user(user)
#         return redirect(url_for('index'))

#     return 'Incorrect username or password'

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))



@app.route('/')
# @login_required
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
# @login_required
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)