from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path


app = Flask(__name__)


# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ml" / "student_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    fail_probability = None
    pass_probability = None
    total_marks = None
    percentage = None
    error = None

    if request.method == "POST":

        try:

            # ----------------------------------------
            # Read form values
            # ----------------------------------------

            attendance = float(request.form["attendance"])
            sessional = float(request.form["sessional"])
            assignment = float(request.form["assignment"])
            class_performance = float(
                request.form["class_performance"]
            )
            viva = float(request.form["viva"])
            mini_project = float(request.form["mini_project"])
            lab_record = float(request.form["lab_record"])
            study_hours = float(request.form["study_hours"])


            # ----------------------------------------
            # Validation
            # ----------------------------------------

            if not 0 <= attendance <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if not 0 <= sessional <= 30:
                raise ValueError("Sessional must be between 0 and 30.")

            if not 0 <= assignment <= 20:
                raise ValueError("Assignment must be between 0 and 20.")

            if not 0 <= class_performance <= 20:
                raise ValueError(
                    "Class Performance must be between 0 and 20."
                )

            if not 0 <= viva <= 20:
                raise ValueError("Viva must be between 0 and 20.")

            if not 0 <= mini_project <= 20:
                raise ValueError(
                    "Mini Project must be between 0 and 20."
                )

            if not 0 <= lab_record <= 10:
                raise ValueError(
                    "Lab Record must be between 0 and 10."
                )

            if not 0 <= study_hours <= 10:
                raise ValueError(
                    "Study Hours must be between 0 and 10."
                )


            # ----------------------------------------
            # Calculate Total Marks
            # ----------------------------------------

            total_marks = (
                sessional
                + assignment
                + class_performance
                + viva
                + mini_project
                + lab_record
            )

            percentage = (total_marks / 120) * 100


            # ----------------------------------------
            # Prepare ML Input
            # ----------------------------------------

            input_data = pd.DataFrame([{
                "attendance": attendance,
                "sessional": sessional,
                "assignment": assignment,
                "class_performance": class_performance,
                "viva": viva,
                "mini_project": mini_project,
                "lab_record": lab_record,
                "study_hours": study_hours
            }])


            # ----------------------------------------
            # Prediction
            # ----------------------------------------

            prediction = model.predict(input_data)[0]


            # ----------------------------------------
            # Prediction Probability
            # ----------------------------------------

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            for class_name, probability in zip(
                classes,
                probabilities
            ):

                if class_name == "Fail":
                    fail_probability = probability * 100

                elif class_name == "Pass":
                    pass_probability = probability * 100


        except Exception as e:

            error = str(e)


    # ----------------------------------------
    # Send result to HTML
    # ----------------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        fail_probability=fail_probability,
        pass_probability=pass_probability,
        total_marks=total_marks,
        percentage=percentage,
        error=error
    )

@app.route('/aboutMLProject')
def about():
    # 3. Renders the about page when visiting http://127.0.0.1/5000/aboutMLProject
    return render_template('aboutus.html')


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )