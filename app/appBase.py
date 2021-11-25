import app1
import app2
import app3

from multiapp import MultiApp

app = MultiApp()

app.add_app("３次元および２次元プロット", app1.app)
app.add_app("データの確認", app2.app)
app.add_app("モデルのプロット", app3.app)

app.run()
