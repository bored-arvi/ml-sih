from sentence_transformers import SentenceTransformer, util
import easyocr, re, spacy, cv2, numpy as np
import faiss, pickle, os
import matplotlib.pyplot as plt
from skimage import color
import time

model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])
nlp = spacy.load("en_core_web_sm")

text_dim = model.get_sentence_embedding_dimension()
color_bins = (16,16,16)
color_dim = np.prod(color_bins)
total_dim = int(text_dim + color_dim)

faiss_index_file = "cert_index.faiss"
id_map_file = "cert_id_map.pkl"
time_start = time.time()
if os.path.exists(faiss_index_file):
    index = faiss.read_index(faiss_index_file)
else:
    index = faiss.IndexFlatL2(total_dim)

if os.path.exists(id_map_file):
    with open(id_map_file, "rb") as f:
        id_map = pickle.load(f)
else:
    id_map = {}

def extract_text(image_path):
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

def extract_fields_regex(text):
    fields = {}
    fields['name'] = re.search(r'presented to\s*([A-Z][a-zA-Z\s]*?)(?=\s+For\b)', text)
    fields['name'] = fields['name'].group(1).strip() if fields['name'] else None
    fields['event'] = re.search(r'participating in the (.*?) held by', text)
    fields['event'] = fields['event'].group(1).strip() if fields['event'] else None
    fields['issuer'] = re.search(r'held by (.*?) on', text)
    fields['issuer'] = fields['issuer'].group(1).strip() if fields['issuer'] else None
    fields['date'] = re.search(r'on ([0-9]{1,2} [A-Za-z]+ [0-9]{4})', text)
    fields['date'] = fields['date'].group(1).strip() if fields['date'] else None
    return fields

def extract_fields_ner(text):
    fields = {}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            fields['name_ner'] = ent.text
        elif ent.label_ == "ORG":
            fields['issuer_ner'] = ent.text
        elif ent.label_ == "DATE":
            fields['date_ner'] = ent.text
    return fields

def combine_db_fields(regex_fields, ner_fields):
    return " ".join([
        regex_fields.get('name') or ner_fields.get('name_ner') or '',
        regex_fields.get('event') or '',
        regex_fields.get('issuer') or ner_fields.get('issuer_ner') or '',
        regex_fields.get('date') or ner_fields.get('date_ner') or ''
    ])

def extract_non_db_text(full_text, db_fields_text):
    db_words = set(db_fields_text.split())
    non_db_words = [w for w in full_text.split() if w not in db_words]
    return " ".join(non_db_words)

def color_embedding(image_path, bins=color_bins):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img], [0,1,2], None, bins, [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def deltaE_local_difference(img1_path, img2_path, threshold=5):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return None
    img1 = cv2.cvtColor(cv2.resize(img1, (512, 512)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(img2, (512, 512)), cv2.COLOR_BGR2RGB)
    lab1 = color.rgb2lab(img1)
    lab2 = color.rgb2lab(img2)
    deltaE_map = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
    noticeable_fraction = np.mean(deltaE_map > threshold)
    return noticeable_fraction

def text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def color_similarity(hist1, hist2):
    return float(np.dot(hist1, hist2) / (np.linalg.norm(hist1)*np.linalg.norm(hist2)))

def get_embedding_by_id(certificate_id):
    if certificate_id not in id_map:
        return None
    idx = id_map[certificate_id]
    emb = index.reconstruct(idx)
    return emb

def compare_against_stored(certificate_id, new_non_db_text, new_color_emb, new_db_fields_text, new_image_path=None):
    stored_emb = get_embedding_by_id(certificate_id)
    if stored_emb is None:
        return None
    stored_text_emb = stored_emb[:text_dim]
    stored_color_emb = stored_emb[text_dim:]
    new_text_emb = model.encode(new_non_db_text, convert_to_numpy=True)
    combined_new = np.concatenate([new_text_emb, new_color_emb]).astype('float32')
    sim_combined = float(np.dot(combined_new, stored_emb) / (np.linalg.norm(combined_new)*np.linalg.norm(stored_emb)))
    sim_text = float(np.dot(new_text_emb, stored_text_emb) / (np.linalg.norm(new_text_emb)*np.linalg.norm(stored_text_emb)))
    sim_color = float(np.dot(new_color_emb, stored_color_emb) / (np.linalg.norm(new_color_emb)*np.linalg.norm(stored_color_emb)))
    stored_db_file = f"{certificate_id}_db.txt"
    if os.path.exists(stored_db_file):
        with open(stored_db_file, "r") as f:
            stored_db_text = f.read().strip()
    else:
        stored_db_text = "Yuvarai B Digital Marketing Workshop Wardiere Company 6 June 2025"
    db_field_score = text_similarity(new_db_fields_text, stored_db_text)
    noticeable_color_fraction = None
    if new_image_path:
        stored_image_path = f"{certificate_id}_image.png"
        if os.path.exists(stored_image_path):
            noticeable_color_fraction = deltaE_local_difference(stored_image_path, new_image_path, threshold=5)
    return {
        "combined_similarity": sim_combined,
        "text_similarity": sim_text,
        "color_similarity": sim_color,
        "db_field_score": db_field_score,
        "noticeable_color_fraction": noticeable_color_fraction
    }

def embed_certificate(image_path, non_db_text, db_fields_text, certificate_id):
    stored_image_path = f"{certificate_id}_image.png"
    if certificate_id in id_map:
        print(f"Certificate ID '{certificate_id}' already exists. Comparing instead of embedding.")
        new_color_emb = color_embedding(image_path)
        scores = compare_against_stored(
            certificate_id,
            new_non_db_text=non_db_text,
            new_color_emb=new_color_emb,
            new_db_fields_text=db_fields_text
        )
        noticeable_fraction, saved_heatmap = None, None
        if os.path.exists(stored_image_path):
            heatmap_file = f"{certificate_id}_deltaE_heatmap.png"
            noticeable_fraction, saved_heatmap = deltaE_heatmap(stored_image_path, image_path, output_path=heatmap_file)
        if scores:
            print("Comparison with stored certificate:")
            print("Combined Similarity:", round(scores["combined_similarity"], 3))
            print("Text (non-DB) Similarity:", round(scores["text_similarity"], 3))
            print("Color Similarity:", round(scores["color_similarity"], 3))
            print("DB Fields Similarity:", round(scores["db_field_score"], 3))
            if noticeable_fraction is not None:
                print(f"Noticeable Color Difference Fraction: {noticeable_fraction:.3f}")
                if noticeable_fraction > 0.01:
                    print("Certificate shows noticeable color changes!")
                else:
                    print("Certificate color appears similar to stored version.")
            else:
                print("No stored certificate image found; skipping ΔE heatmap comparison.")
            if saved_heatmap:
                print(f"ΔE heatmap saved to: {saved_heatmap}")
        return None
    confirm = input(f"Do you want to embed certificate '{certificate_id}'? (y/n): ").lower()
    if confirm != 'y':
        print("Embedding cancelled.")
        return None
    cv2.imwrite(stored_image_path, cv2.imread(image_path))
    text_emb = model.encode(non_db_text, convert_to_numpy=True)
    color_emb = color_embedding(image_path)
    combined_emb = np.concatenate([text_emb, color_emb]).astype('float32')
    index.add(np.array([combined_emb]))
    id_map[certificate_id] = index.ntotal - 1
    with open(f"{certificate_id}_db.txt", "w") as f:
        f.write(db_fields_text)
    faiss.write_index(index, faiss_index_file)
    with open(id_map_file, "wb") as f:
        pickle.dump(id_map, f)
    print(f"Certificate '{certificate_id}' embedded and stored successfully.")
    return combined_emb

def deltaE_heatmap(img1_path, img2_path, output_path="deltaE_heatmap.png", resize_dim=(512,512), threshold=5):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return None, None
    img1_rgb = cv2.cvtColor(cv2.resize(img1, resize_dim), cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(cv2.resize(img2, resize_dim), cv2.COLOR_BGR2RGB)
    lab1 = color.rgb2lab(img1_rgb)
    lab2 = color.rgb2lab(img2_rgb)
    deltaE_map = np.sqrt(np.sum((lab1 - lab2)**2, axis=2))
    noticeable_fraction = np.mean(deltaE_map > threshold)
    plt.figure(figsize=(6,6))
    plt.imshow(deltaE_map, cmap='hot')
    plt.colorbar(label='ΔE')
    plt.axis('off')
    plt.title('ΔE Heatmap')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return noticeable_fraction, output_path

image_path = "data mismatch.png"
time_start_total = time.time()
t0 = time.time()
extracted_text = extract_text(image_path)
t_text = time.time() - t0
print(f"Text extraction time: {t_text:.3f} seconds")

# --- DB Fields ---
regex_fields = extract_fields_regex(extracted_text)
ner_fields = extract_fields_ner(extracted_text)
db_fields_text = combine_db_fields(regex_fields, ner_fields)
non_db_text = extract_non_db_text(extracted_text, db_fields_text)
cert_id="CERT-001"
# --- Embedding / Comparison Timer ---
t0 = time.time()

embed_certificate(image_path, non_db_text, db_fields_text, cert_id)
t_embed = time.time() - t0
print(f"Embedding/comparison time: {t_embed:.3f} seconds")
# --- Total Timer ---
t_total = time.time() - time_start_total
print(f"Total time: {t_total:.3f} seconds")
