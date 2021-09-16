import app1
import app2

from multiapp import MultiApp

app = MultiApp()

app.add_app("app1", app1.app)
app.add_app("app2", app2.app)

app.run()
