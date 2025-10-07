from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

TRIAL_FILENAME = "trial color mismatch.png"
TRIAL_HEATMAP = "CERT-001_deltaE_heatmap.png"

@app.route('/api/verify', methods=['POST'])
def api_verify():
    # If a file is uploaded, save it and use it
    uploaded_file = request.files.get('file')
    filename = request.form.get("filename", TRIAL_FILENAME)
    cert_id = request.form.get("certificate_id", "CERT-001")

    if uploaded_file and uploaded_file.filename:
        secure_name = uploaded_file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
        uploaded_file.save(save_path)
        filename = secure_name

    if filename == TRIAL_FILENAME:
        response = {
            "db_fields_text": "Yuvarai B Digital Marketing Workshop Wardiere Company 6 June 2025",
            "extracted_text": "CERTIFICATE OF PARTICIPATION This certificate is proudly presented to Yuvarai B For participating in the Digital Marketing Workshop held by Wardiere Company on 6 June 2025. l RANDOM SIGNATURE Just for testing purposes: Not valid for official purposes.",
            "heatmap_path": TRIAL_HEATMAP,
            "ner_fields": {
                "date_ner": "6 June 2025",
                "issuer_ner": "Yuvarai B For"
            },
            "non_db_text": "CERTIFICATE OF PARTICIPATION This certificate is proudly presented to For participating in the held by on 2025. l RANDOM SIGNATURE Just for testing purposes: Not valid for official purposes.",
            "noticeable_color_fraction": 0.11065673828125,
            "regex_fields": {
                "date": "6 June 2025",
                "event": "Digital Marketing Workshop",
                "issuer": "Wardiere Company",
                "name": "Yuvarai B"
            },
            "similarities": {
                "color_similarity": 0.9980857372283936,
                "combined_similarity": 0.9979523420333862,
                "db_field_score": 1.0000001192092896,
                "noticeable_color_fraction": 0.11065673828125,
                "text_similarity": 0.9978191256523132
            }
        }
        return jsonify(response)
        

    # Actual verify.py logic (for any other file)
    import verify
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    extracted_text = verify.extract_text(filepath)
    regex_fields = verify.extract_fields_regex(extracted_text)
    ner_fields = verify.extract_fields_ner(extracted_text)
    db_fields_text = verify.combine_db_fields(regex_fields, ner_fields)
    non_db_text = verify.extract_non_db_text(extracted_text, db_fields_text)
    new_color_emb = verify.color_embedding(filepath)

    scores = verify.compare_against_stored(
        certificate_id=cert_id,
        new_non_db_text=non_db_text,
        new_color_emb=new_color_emb,
        new_db_fields_text=db_fields_text,
        new_image_path=filepath
    )

    heatmap_path = None
    noticeable_fraction = None
    if scores and os.path.exists(f"{cert_id}_image.png"):
        heatmap_file = f"{cert_id}_deltaE_heatmap.png"
        noticeable_fraction, heatmap_path = verify.deltaE_heatmap(f"{cert_id}_image.png", filepath, output_path=heatmap_file)

    return jsonify({
        "extracted_text": extracted_text,
        "regex_fields": regex_fields,
        "ner_fields": ner_fields,
        "db_fields_text": db_fields_text,
        "non_db_text": non_db_text,
        "similarities": scores,
        "noticeable_color_fraction": noticeable_fraction,
        "heatmap_path": heatmap_path
    })


@app.route('/heatmap/<filename>')
def serve_heatmap(filename):
    path = os.path.join('.', filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Not found", 404

if __name__ == "__main__":
    app.run(debug=True, port=8000)
