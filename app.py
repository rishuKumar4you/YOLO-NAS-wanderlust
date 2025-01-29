import os
import warnings

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress warnings from specific modules
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="matplotlib")

# Disable crash tips in super_gradients
os.environ["CRASH_HANDLER"] = "FALSE"                  ## these settings declutter the log and supress warnings.


from flask import Flask, request, jsonify, send_file, Response
import subprocess
import io
import uuid
import json
import cv2
import base64

"""to use this script, use the command.
    
    !curl -X POST -F "file=@../create-process-flow-diagram-pid-ri-bfd-and-ufd.jpg" -F "confidence=0.3" http://127.0.0.1:8081/predict -o ../responses/response10.json

the structure of response will be:

{
    "json_data": {
        "predictions": [
            {
                "x": 182.22,
                "y": 172.12,
                "width": 72.52,
                "height": 366.56,
                "confidence": 0.779,
                "class": "symbol",
                "class_id": 0,
                "detection_id": "bd88df16-a480-460a-867d-0208b88f75cf"
            }, ..] 
            },
        "labeled_image": "base64encoded_image_string"
        }

"""



# Flask app
app = Flask(__name__)

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return "healthy", 200

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the request contains a file
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]  

        # Check if a file was provided
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
                
            
            # Parse the confidence value from the request
        # Default confidence value is 0.4 if not provided
        confidence = request.form.get("confidence", default=0.4, type=float)

        # Ensure the confidence value is valid
        if not (0 <= confidence <= 1):
            return jsonify({"error": "Invalid confidence value. It should be between 0 and 1."}), 400


        # Save the uploaded image to a temporary file
        input_dir = "temp_inputs"
        os.makedirs(input_dir, exist_ok=True)
        input_filename = f"{uuid.uuid4()}.jpg"
        input_filepath = os.path.join(input_dir, input_filename)

        file.save(input_filepath)

        # Run the visualise.py script
        try:
            subprocess.run(
                [
                    "python3", "visualise.py",
                    "--data", "/workspace/YOLO-NAS-pytorch/dataset/data.yaml",
                    "--model", "yolo_nas_m",
                    "--weight", "/workspace/artifacts/RUN_1_ep50b8dt_sym_only/ckpt_best.pth",
                    "--image", input_filepath,
                    "--conf", str(confidence),
                    "--save"
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Inference failed: {str(e)}"}), 500

        # Paths to the output files
        output_dir = os.path.join("outputs", f"{os.path.splitext(input_filename)[0]}")
        os.makedirs(output_dir, exist_ok=True)

        annotated_image_path = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}_labelled.jpg")
        json_path = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}_labelled.json")

        # Check if the output files exist
        if not os.path.exists(annotated_image_path) or not os.path.exists(json_path):
            return jsonify({"error": "Output files not generated"}), 500

        # Read the labeled image
        with open(annotated_image_path, "rb") as img_file:
            labeled_image_data = img_file.read()

        # Read the JSON file
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        
        # encode the image to base 64
        labeled_image_base64 = base64.b64encode(labeled_image_data).decode("utf-8")


        # Create a response with the labeled image and JSON
        response = {
            "json_data": json_data,
            "labeled_image": labeled_image_base64
        }

        # Return the response with labeled image and JSON
        return Response(
            response=json.dumps(response),
            content_type="application/json",
            status=200
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
