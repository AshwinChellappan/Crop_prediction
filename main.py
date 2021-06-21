from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client
# import seaborn as sns

logreg = LogisticRegression()
standardscalar = StandardScaler()
account_sid = ["AC6942a93b8721a23530bd0127ae516b01"]
auth_token = ["fb2beaa51ccc7a262f61395113df324a"]
client = Client(account_sid, auth_token)
# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods=["GET", "POST"])
def gfg():
    if request.method == "POST":
        # uName=request.form.get("uName")
        # pNum=request.form.get("pNum")
        nitrogen = request.form.get("ncontent")
        potassium = request.form.get("potcontent")
        phosphorous=request.form.get("poscontent")
        temperature=request.form.get("temp")
        humidity=request.form.get("humidity")
        phval=request.form.get("phval")
        rainfall=request.form.get("rainfall")
        df_cust=pd.DataFrame(data=[nitrogen,potassium,phosphorous,temperature,humidity,phval,rainfall])
        print('Dataframe',df_cust)
        df_custscale=standardscalar.fit_transform(df_cust).reshape(1,-1)
        print('Scaled Dataframe',df_custscale)
        log_pred = logreg.predict(df_custscale)
        print("Predicted",log_pred)
        return "Crop Recommended for your soil condition is " +str(log_pred)
    # message = client.messages.create(from_=+14844699603, to=+916379508752, body="Hi there! How are you")
    # print(message.sid)
    return render_template("form.html")


if __name__ == '__main__':
    print('Line 1')
    # df=pd.read_csv('D:\\DataScience\\Projectspace\\Crop_recommendation.csv')
    # print(df.head())
    df = pd.read_csv('Crop_recommendation.csv')
    c = df.label.astype('category')
    targets = dict(enumerate(c.cat.categories))
    df['target'] = c.cat.codes
    x = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['target']
    df_scaled = standardscalar.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.3, random_state=50)
    logreg.fit(x_train, y_train)
    # log_pred = logreg.predict(x_test)
    app.run()
    print('Line 2')
