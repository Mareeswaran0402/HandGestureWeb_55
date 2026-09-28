from flask import Flask, render_template, request, jsonify

import joblib
import numpy as np
import os


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# FILES
# =========================================================

MODEL_FILE = "gesture_model.pkl"


# =========================================================
# SUPPORTED GESTURES
# =========================================================

GESTURES = [
    "HELLO",
    "STOP",
    "GOOD",
    "YES",
    "PEACE",
    "LOVE",
    "HELP",
    "THANKS",
    "COME",
    "BYE"
]


# =========================================================
# SETTINGS
# =========================================================

# Minimum confidence required
CONFIDENCE_THRESHOLD = 55.0

# Difference between top and second prediction
MARGIN_THRESHOLD = 8.0


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Missing model file: {MODEL_FILE}"
    )


try:

    model = joblib.load(
        MODEL_FILE
    )

except Exception as e:

    print()
    print("=" * 60)
    print("ERROR: COULD NOT LOAD MODEL")
    print("=" * 60)
    print()
    print(e)
    print()

    raise SystemExit(1)


# =========================================================
# MODEL CLASSES
# =========================================================

MODEL_CLASSES = [
    str(class_name)
    for class_name in model.classes_
]


print()
print("=" * 60)
print("        HAND GESTURE WEB APPLICATION")
print("=" * 60)
print()

print("Model classes:")

for class_name in MODEL_CLASSES:

    print(
        " -",
        class_name
    )

print()


# =========================================================
# CHECK FOR UNKNOWN
# =========================================================

if "UNKNOWN" in MODEL_CLASSES:

    print("=" * 60)
    print("WARNING: UNKNOWN FOUND IN MODEL")
    print("=" * 60)
    print()

    print(
        "Your gesture_model.pkl still contains UNKNOWN."
    )

    print(
        "Retrain the model using only the 10 gestures."
    )

    print()

    raise RuntimeError(
        "UNKNOWN is present in gesture_model.pkl. "
        "Retrain the model without UNKNOWN."
    )


# =========================================================
# CHECK REQUIRED CLASSES
# =========================================================

missing_gestures = [

    gesture

    for gesture in GESTURES

    if gesture not in MODEL_CLASSES

]


if missing_gestures:

    print(
        "Missing gesture classes:"
    )

    for gesture in missing_gestures:

        print(
            " -",
            gesture
        )

    raise RuntimeError(
        "gesture_model.pkl does not contain all 10 "
        "required gesture classes."
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "model": "loaded",

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "margin_threshold":
            MARGIN_THRESHOLD,

        "gestures": GESTURES

    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # =================================================
        # GET JSON
        # =================================================

        data = request.get_json(
            silent=True
        )


        if data is None:

            return jsonify({

                "success": False,

                "error": "No JSON data received."

            }), 400


        # =================================================
        # GET LANDMARKS
        # =================================================

        landmarks = data.get(
            "landmarks"
        )


        # -------------------------------------------------
        # NO LANDMARKS
        # -------------------------------------------------

        if not landmarks:

            return jsonify({

                "success": True,

                "gesture": "NO HAND",

                "confidence": 0.0,

                "margin": 0.0,

                "accepted": False

            })


        # =================================================
        # CHECK 21 LANDMARKS
        # =================================================

        if not isinstance(
            landmarks,
            list
        ):

            return jsonify({

                "success": False,

                "error": (
                    "Landmarks must be a list."
                )

            }), 400


        if len(landmarks) != 21:

            return jsonify({

                "success": False,

                "error": (
                    "Expected 21 landmarks, "
                    f"received {len(landmarks)}."
                )

            }), 400


        # =================================================
        # CREATE 63 FEATURES
        # =================================================

        features = []


        for index, point in enumerate(
            landmarks
        ):

            if not isinstance(
                point,
                dict
            ):

                return jsonify({

                    "success": False,

                    "error": (
                        f"Landmark {index} "
                        "is invalid."
                    )

                }), 400


            if (
                "x" not in point
                or "y" not in point
                or "z" not in point
            ):

                return jsonify({

                    "success": False,

                    "error": (
                        f"Landmark {index} "
                        "must contain x, y and z."
                    )

                }), 400


            try:

                x = float(
                    point["x"]
                )

                y = float(
                    point["y"]
                )

                z = float(
                    point["z"]
                )

            except (
                TypeError,
                ValueError
            ):

                return jsonify({

                    "success": False,

                    "error": (
                        f"Landmark {index} "
                        "contains invalid values."
                    )

                }), 400


            features.append(x)

            features.append(y)

            features.append(z)


        # =================================================
        # CHECK FEATURE COUNT
        # =================================================

        if len(features) != 63:

            return jsonify({

                "success": False,

                "error": (
                    "Expected 63 features."
                )

            }), 400


        # =================================================
        # NUMPY ARRAY
        # =================================================

        X = np.asarray(

            features,

            dtype=np.float32

        ).reshape(

            1,

            63

        )


        # =================================================
        # MODEL PROBABILITIES
        # =================================================

        probabilities = model.predict_proba(
            X
        )[0]


        # =================================================
        # TOP TWO CLASSES
        # =================================================

        sorted_indices = np.argsort(
            probabilities
        )[::-1]


        best_index = int(
            sorted_indices[0]
        )


        if len(sorted_indices) > 1:

            second_index = int(
                sorted_indices[1]
            )

        else:

            second_index = best_index


        # =================================================
        # PREDICTION
        # =================================================

        prediction = str(

            model.classes_[
                best_index
            ]

        )


        # =================================================
        # CONFIDENCE
        # =================================================

        confidence = (

            float(
                probabilities[
                    best_index
                ]
            )

            * 100.0

        )


        # =================================================
        # SECOND CONFIDENCE
        # =================================================

        second_confidence = (

            float(
                probabilities[
                    second_index
                ]
            )

            * 100.0

        )


        # =================================================
        # MARGIN
        # =================================================

        margin = (

            confidence
            - second_confidence

        )


        # =================================================
        # ACCEPTANCE
        # =================================================

        accepted = True


        # -------------------------------------------------
        # CONFIDENCE < 55%
        # -------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            accepted = False


        # -------------------------------------------------
        # TOP TWO TOO CLOSE
        # -------------------------------------------------

        if margin < MARGIN_THRESHOLD:

            accepted = False


        # =================================================
        # UNCERTAIN
        # =================================================

        if not accepted:

            display_prediction = (
                "UNCERTAIN"
            )

        else:

            display_prediction = (
                prediction
            )


        # =================================================
        # RETURN JSON
        # =================================================

        return jsonify({

            "success": True,

            "gesture":
                display_prediction,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "margin":
                round(
                    margin,
                    2
                ),

            "accepted":
                accepted

        })


    # =====================================================
    # INVALID DATA
    # =====================================================

    except ValueError as e:

        return jsonify({

            "success": False,

            "error": (
                "Invalid landmark data: "
                + str(e)
            )

        }), 400


    # =====================================================
    # SERVER ERROR
    # =====================================================

    except Exception as e:

        print()
        print(
            "Prediction error:",
            e
        )
        print()


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =========================================================
# 413 ERROR
# =========================================================

@app.errorhandler(413)
def request_too_large(error):

    return jsonify({

        "success": False,

        "error": "Request is too large."

    }), 413


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    # Render provides PORT.
    # Local development defaults to 5000.

    port = int(

        os.environ.get(
            "PORT",
            5000
        )

    )


    print()
    print("=" * 60)
    print("        HAND GESTURE TO WORDS")
    print("=" * 60)
    print()

    print(
        "Supported gestures:"
    )

    for gesture in GESTURES:

        print(
            " -",
            gesture
        )

    print()

    print(
        "Confidence threshold:",
        f"{CONFIDENCE_THRESHOLD}%"
    )

    print(
        "Margin threshold:",
        f"{MARGIN_THRESHOLD}%"
    )

    print(
        "Hold time:",
        "0.3 seconds"
    )

    print()

    print(
        f"Starting server on port {port}..."
    )

    print()


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )