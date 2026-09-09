"""Serve a web page that predicts the Vietnamese dish in an uploaded photo.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/serve_predict.py \
        --checkpoint artifacts/vietnamese_resnet18/best.pt --port 8000
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from food_classifier.inference import get_predictor

CHECKPOINT = Path("artifacts/vietnamese_resnet18/best.pt")

app = FastAPI(title="Nhận diện món ăn Việt Nam", version="1.0.0")

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Nhận diện món ăn Việt Nam</title>
<style>
:root{--bg:#0f1216;--card:#171c23;--line:#262d38;--fg:#e8ecf1;--mut:#8b96a5;--accent:#ff7a45;}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:640px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 22px}
#drop{border:2px dashed var(--line);border-radius:14px;padding:34px 18px;text-align:center;cursor:pointer;transition:.15s;background:var(--card)}
#drop.hover{border-color:var(--accent);background:#1d232b}
#drop b{color:var(--accent)}
#prev{display:none;margin:18px auto 0;max-height:320px;border-radius:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-top:18px}
.row{display:flex;align-items:center;gap:10px;margin:9px 0}
.row .nm{width:180px;flex:0 0 auto;font-size:14px}
.row .nm small{color:var(--mut);display:block;font-size:11px}
.bar{flex:1;height:14px;background:#0d1014;border-radius:7px;overflow:hidden}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#ff7a45,#ffb347)}
.pct{width:52px;text-align:right;font-variant-numeric:tabular-nums;color:var(--mut)}
.top .bar>i{background:linear-gradient(90deg,#22c55e,#86efac)}
.top .pct{color:#86efac}
.muted{color:var(--mut);font-size:13px}
#err{color:#f87171;margin-top:12px}
</style></head><body><div class="wrap">
<h1>🍜 Nhận diện món ăn Việt Nam</h1>
<p class="sub">Kéo-thả hoặc chọn một ảnh món ăn. Mô hình ResNet18 (17 lớp) sẽ đoán top-5.</p>
<div id="drop">Kéo ảnh vào đây, hoặc <b>bấm để chọn ảnh</b><input id="file" type="file" accept="image/*" hidden/></div>
<img id="prev"/>
<div id="err"></div>
<div id="out"></div>
<p class="muted" id="meta"></p>
</div><script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),
prev=document.getElementById('prev'),out=document.getElementById('out'),err=document.getElementById('err');
drop.onclick=()=>file.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hover')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hover')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files[0])send(ev.dataTransfer.files[0])});
file.onchange=()=>{if(file.files[0])send(file.files[0])};
function send(f){
  err.textContent='';out.innerHTML='<p class="muted">Đang dự đoán…</p>';
  prev.src=URL.createObjectURL(f);prev.style.display='block';
  const fd=new FormData();fd.append('file',f);
  fetch('/api/predict',{method:'POST',body:fd}).then(r=>r.json().then(j=>{if(!r.ok)throw new Error(j.detail||'Lỗi');return j}))
  .then(render).catch(e=>{out.innerHTML='';err.textContent=e.message});
}
function render(j){
  out.innerHTML='<div class="card">'+j.predictions.map((p,i)=>{
    const pct=(p.probability*100).toFixed(1);
    return `<div class="row ${i===0?'top':''}"><div class="nm">${p.name}<small>${p.label}</small></div>`+
           `<div class="bar"><i style="width:${pct}%"></i></div><div class="pct">${pct}%</div></div>`;
  }).join('')+'</div>';
  document.getElementById('meta').textContent=`Thiết bị: ${j.device} · ${j.num_classes} lớp`;
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/health")
def health() -> dict[str, object]:
    predictor = get_predictor(str(CHECKPOINT))
    return {"status": "ok", "classes": predictor.classes, "device": str(predictor.device)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if not CHECKPOINT.exists():
        raise HTTPException(status_code=503, detail=f"Chưa có checkpoint: {CHECKPOINT}")
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File không phải ảnh hợp lệ.")
    predictor = get_predictor(str(CHECKPOINT))
    predictions = predictor.predict(image, top_k=5)
    return JSONResponse({"predictions": predictions, "device": str(predictor.device), "num_classes": len(predictor.classes)})


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint
    get_predictor(str(CHECKPOINT))  # warm the model before serving
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
