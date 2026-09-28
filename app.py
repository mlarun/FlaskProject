from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    # 2. Flask automatically searches for 'home.html' inside the 'templates' folder
    return render_template('home.html')

@app.route('/about')
def about():
    # 3. Renders the about page when visiting http://127.0.0
    return render_template('aboutus.html')

if __name__ == '__main__':
    
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
