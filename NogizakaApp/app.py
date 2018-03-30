import os, re
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from detect import start_detect


app = Flask(__name__)

@app.route('/')
def index():
    name = "Hello World"
    return render_template('index.html', title='flask test', name=name)

@app.route('/good')
def good():
    name = "Good"
    return name

###########################
# 以下画像アップロード処理
##########################

#iPhoneの画像ファイル形式への対応

#画像アップロード
UPLOAD_FOLDER = './static/asset/uploads'
ALLOWED_EXTENSIONS = set(['jpg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

#アップロードを許可する画像ファイル
def allowed_file(filename):
    # ファイル名の拡張子より前を取得し, フォーマット後のファイル名に変更
    filename = re.search("(?<!\.)\w+", filename).group(0) + "." + 'jpg'
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


#画像POST処理
@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_url = '/uploads/' + filename
            predict_name, predict_enname, rate = start_detect(filename)

            if predict_name != None:
                if predict_enname != "Akimoto Yasushi":
                    predict_enname = predict_enname.lower()
                    enname = predict_enname.split()
                    url_enname = enname[1] + "." + enname[0]
                else:
                    url_enname = None

                print(predict_name)
                # return render_template('index.html', img_url=img_url)
                return render_template('result.html', name=predict_name, urlname=url_enname, rate=rate, filename=filename)
            else:
                return render_template('index.html', error="error")
        else:
            return ''' <p>許可されていない拡張子です</p> '''
    else:
        return redirect(url_for('index'))


#アップロードされた画像ファイル
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


## おまじない
if __name__ == "__main__":
    app.run()
