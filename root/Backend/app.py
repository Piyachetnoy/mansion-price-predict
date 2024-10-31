from flask import Flask, render_template, request, redirect, url_for, session
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np
import joblib
import pandas as pd
import config
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'



# 学習済みモデルを読み込み利用します
def predict(parameters):
    pred =0.1
    #データの整形
    new_data = pd.DataFrame({
        '築年数': [(parameters[0] -17.753359) / 14.589411],
        '階数': [(parameters[1] - 8.098852)/ 7.507385],
        '階': [(parameters[2] - 4.567270) / 4.736252],
        '間取り_label': [(parameters[3] - 7.605640) / 9.983787],
        '部屋数':[(parameters[4] - 1.298260) / 0.580308],
        'LDK':[(parameters[5] - 1.822961) / 1.107274],
        'S':[(parameters[6] - 0.02876) / 0.167133],
        '23区_label':[(parameters[7] - 10.869985) /  6.493723],
        '最寄駅_label':[(parameters[8] - 232.228634) / 130.911890]
        })
    #モデルの読み込み
    try:
        model = joblib.load(r'C:\Users\nyugo\team-a-2024-summer-08-26\root\Backend\model\lgb_model3.pkl')
        #scaler = StandardScaler()
        #std_scaler = StandardScaler()
        features = ['築年数', '階数', '階', '間取り_label', '部屋数', 'LDK', 'S','23区_label','最寄駅_label']
        X_new = new_data[features]
        #
        #std_scaler.fit(X_new)
        #print('x_new=',X_new)
        #X_std = pd.DataFrame(std_scaler.transform(X_new))
        #
        #X_new_scaled = scaler.transform(X_std)
        #print(X_new_scaled)
        #print('X_std',X_std)
        pred = model.predict(X_new,0)
    except Exception as e:
        print("error=" ,e)
    return pred

# データの読み込みとカラムの取り出し
df = pd.read_csv(r'C:\Users\nyugo\team-a-2024-summer-08-26\root\Backend\data\0829_train_2 - コピー.csv', encoding='utf-8')
y = df[config.TARGET]
X = df[config.COLUMNS]
col_names = X.columns

class PredictForm(Form):
    # 英語の変数名を使用
    age = FloatField("築年数:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    floor_count = FloatField("階数:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    floor = FloatField("階:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    layout_label = FloatField("間取り_label:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    rooms = FloatField("部屋数:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    ldk = FloatField("LDK:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    s = FloatField("S:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    zone_label = FloatField("23区_label:", [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])
    station = FloatField('最寄駅_label', [validators.InputRequired("この項目は入力必須です"), validators.NumberRange(min=0, max=1000)])

    submit = SubmitField("予測")

@app.route("/", methods=["GET", "POST"])
def index():
    form = PredictForm(request.form)
    if request.method == "POST":
        pred_list = []
        try:
            for col in col_names:
                # フィールド名をマッピングする辞書を使用
                field_map = {
                    '築年数': 'age',
                    '階数': 'floor_count',
                    '階': 'floor',
                    '間取り_label': 'layout_label',
                    '部屋数': 'rooms',
                    'LDK': 'ldk',
                    'S': 's',
                    '23区_label': 'zone_label',
                    '最寄駅_label': 'station',
                }

            
                form_field_name = field_map.get(col, col)  # フィールド名を変換
                pred_value = request.form.get(form_field_name)


                if pred_value is None or pred_value.strip() == "":
                    return render_template("index.html", form=form, error="すべてのフィールドを入力してください。")
                
                
                pred_list.append(float(pred_value))  # 入力値をfloat型に変換
            
            x = np.array(pred_list)
            pred = predict(x)
        except KeyError as e:
            return render_template("index.html", form=form, error=f"カラム '{e.args[0]}' が見つかりません。")
        except ValueError:
            return render_template("index.html", form=form, error="無効な入力が含まれています。")

        # セッションに予測結果を保存
        session['prediction'] = int(pred[0])
        return redirect("/results")
    else:
        return render_template("index.html", form=form)

@app.route("/results")
def results():
    pred = session.get('prediction')
    if pred is None:
        return redirect("/")
    return render_template("results.html", pred=pred)

if __name__ == "__main__":
    app.run(debug=True)

