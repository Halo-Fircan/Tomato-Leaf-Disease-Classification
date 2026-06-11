import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = "model.tflite"
LABEL_PATH = "label.txt"

DISEASE_INFO = {
    "Bacterial_spot": {
        "display": "Bacterial Spot", "severity": "Medium",
        "desc": "Penyakit bakteri Xanthomonas yang menyebabkan bercak-bercak kecil berair pada daun.",
        "treatment": "Semprot fungisida berbasis tembaga. Hindari penyiraman dari atas. Rotasi tanaman setiap musim.",
    },
    "Early_blight": {
        "display": "Early Blight", "severity": "Medium",
        "desc": "Infeksi jamur Alternaria solani — bercak konsentris coklat gelap pada daun tua.",
        "treatment": "Gunakan fungisida chlorothalonil atau mancozeb. Buang daun yang terinfeksi dan jaga sirkulasi udara.",
    },
    "Late_blight": {
        "display": "Late Blight", "severity": "High",
        "desc": "Phytophthora infestans — penyakit paling destruktif pada tomat, menyebar cepat di kondisi lembab.",
        "treatment": "Segera semprot fungisida sistemik (metalaxyl). Isolasi tanaman yang terinfeksi. Hindari kelembaban berlebih.",
    },
    "Leaf_Mold": {
        "display": "Leaf Mold", "severity": "Medium",
        "desc": "Jamur Cladosporium fulvum menyebabkan bercak kuning di permukaan atas dan spora coklat di bawah daun.",
        "treatment": "Kurangi kelembaban greenhouse. Gunakan fungisida berbahan aktif thiram atau chlorothalonil.",
    },
    "Septoria_leaf_spot": {
        "display": "Septoria Leaf Spot", "severity": "Medium",
        "desc": "Bercak kecil melingkar dengan pusat abu-abu dan tepi gelap akibat jamur Septoria lycopersici.",
        "treatment": "Semprot fungisida berbasis tembaga. Buang daun bagian bawah yang terinfeksi lebih awal.",
    },
    "Spider_mites Two-spotted_spider_mite": {
        "display": "Spider Mites", "severity": "Medium",
        "desc": "Tungau laba-laba menyebabkan stippling halus pada daun dan web tipis di bagian bawah daun.",
        "treatment": "Semprot insektisida/mitisida berbahan aktif abamectin. Semprotkan air bertekanan untuk membersihkan tungau.",
    },
    "Target_Spot": {
        "display": "Target Spot", "severity": "Medium",
        "desc": "Jamur Corynespora cassiicola membentuk pola cincin konsentris seperti sasaran tembak pada daun.",
        "treatment": "Gunakan fungisida berbahan aktif azoxystrobin atau difenoconazole. Jaga sirkulasi udara.",
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Yellow Leaf Curl Virus", "severity": "High",
        "desc": "Virus TYLCV ditularkan kutu kebul (Bemisia tabaci) — daun menggulung ke atas dan menguning.",
        "treatment": "Tidak ada obat langsung. Kendalikan populasi kutu kebul dengan insektisida sistemik. Gunakan varietas tahan.",
    },
    "Tomato_mosaic_virus": {
        "display": "Mosaic Virus", "severity": "High",
        "desc": "Virus ToMV menyebabkan pola mosaik kuning-hijau pada daun dan menghambat pertumbuhan.",
        "treatment": "Tidak ada obat. Cabut dan musnahkan tanaman terinfeksi. Sterilisasi alat. Gunakan benih bersertifikat.",
    },
    "healthy": {
        "display": "Healthy", "severity": "None",
        "desc": "Daun tomat dalam kondisi sehat tanpa tanda-tanda infeksi penyakit.",
        "treatment": "Lanjutkan perawatan rutin: pemupukan seimbang, penyiraman teratur, dan monitoring berkala.",
    },
    "powdery_mildew": {
        "display": "Powdery Mildew", "severity": "Low",
        "desc": "Lapisan putih tepung pada permukaan daun akibat jamur Leveillula taurica — berkembang di kondisi kering.",
        "treatment": "Semprot larutan baking soda (1%) atau fungisida sulfur. Kurangi kepadatan tanaman.",
    },
}

# ── Load model di module level (sebelum FastAPI init) ─────────────────────────
print("Loading model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

with open(LABEL_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Model loaded. Classes: {labels}")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TomatoScan API",
    description="API deteksi penyakit daun tomat menggunakan MobileNetV2 TFLite",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def run_inference(image_bytes: bytes) -> np.ndarray:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor = preprocess(image_bytes)
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "TomatoScan API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": interpreter is not None, "num_classes": len(labels)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG/PNG).")
    image_bytes = await file.read()
    try:
        probs = run_inference(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    top_idx   = int(np.argmax(probs))
    top_label = labels[top_idx]
    info      = DISEASE_INFO.get(top_label, {})
    return JSONResponse({
        "label":      top_label,
        "display":    info.get("display", top_label),
        "confidence": round(float(probs[top_idx]) * 100, 2),
        "severity":   info.get("severity", "Unknown"),
        "desc":       info.get("desc", ""),
        "treatment":  info.get("treatment", ""),
    })

@app.post("/predict/top-k")
async def predict_topk(file: UploadFile = File(...), k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG/PNG).")
    if k < 1 or k > len(labels):
        raise HTTPException(status_code=400, detail=f"k harus antara 1 dan {len(labels)}.")
    image_bytes = await file.read()
    try:
        probs = run_inference(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    sorted_idx = np.argsort(probs)[::-1][:k]
    results = []
    for rank, i in enumerate(sorted_idx):
        lbl  = labels[i]
        info = DISEASE_INFO.get(lbl, {})
        results.append({
            "rank":       rank + 1,
            "label":      lbl,
            "display":    info.get("display", lbl),
            "confidence": round(float(probs[i]) * 100, 2),
            "severity":   info.get("severity", "Unknown"),
            "desc":       info.get("desc", ""),
            "treatment":  info.get("treatment", ""),
        })
    return JSONResponse({"top_k": k, "predictions": results})