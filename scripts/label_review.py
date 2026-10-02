"""Local web tool for auditing validation/test labels against a model's predictions.

    .venv/Scripts/python scripts/label_review.py --port 8010
    -> open http://127.0.0.1:8010

Queues: "confident errors" (wrong predictions, most confident first) for finding mislabels
fast, and "random sample" (seeded, all images) for an unbiased estimate of label noise.
Every decision is written immediately to --reviews (JSON keyed by relative_path).
Keys: 1 label correct · 2 label wrong, the prediction is right · 3 wrong, other class ·
4 remove (not a dish / several dishes / unclear) · 0 skip · arrows navigate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_classifier.vietnamese_dishes import DISHES

DECISIONS = {"label_ok", "label_wrong_pred", "label_wrong_other", "remove", "skip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/vietnamese_food_merged"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("artifacts/merged50_resnet50"))
    parser.add_argument("--reviews", type=Path, default=Path("artifacts/label_review/reviews.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_state(args: argparse.Namespace) -> dict:
    items = []
    for split in ("validation", "test"):
        path = args.predictions_dir / f"predictions_{split}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["top5"] = [(c, float(p)) for c, p in (x.rsplit(":", 1) for x in row["top5"].split(";"))]
                for key in ("pred_prob", "label_prob"):
                    row[key] = float(row[key])
                row["correct"] = row["correct"] == "1"
                items.append(row)
    with (args.data_root / "manifests" / "train.csv").open(encoding="utf-8", newline="") as handle:
        train = list(csv.DictReader(handle))
    rng = random.Random(args.seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in train:
        by_class[row["class_name"]].append(row)
    references = {}
    for name, rows in by_class.items():
        clean = [r["relative_path"] for r in rows if r.get("provider") == "30vnfoods"]
        other = [r["relative_path"] for r in rows if r.get("provider") != "30vnfoods"]
        rng.shuffle(clean)
        rng.shuffle(other)
        references[name] = (clean + other)[:4]
    order = list(range(len(items)))
    rng.shuffle(order)
    for rank, index in enumerate(order):
        items[index]["random_rank"] = rank
    names = {d.canonical_label: d.vietnamese_name for d in DISHES}
    classes = sorted(by_class)
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews.exists() else {}
    return {"items": items, "references": references, "reviews": reviews,
            "names": {c: names.get(c, c) for c in classes}, "classes": classes}


PAGE = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duyệt nhãn</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--line:#e4e1da;--fg:#1d1c1a;--mut:#6d6a63;--acc:#b4481f;--ok:#2f7d4f;--bad:#b3261e;--chip:#f0ede6}
@media (prefers-color-scheme:dark){:root{--bg:#151412;--panel:#1e1d1a;--line:#34322d;--fg:#ece9e2;--mut:#a29e95;--acc:#f08a5d;--ok:#6fcf97;--bad:#ff8a80;--chip:#2a2824}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,"Segoe UI",Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2}
header h1{font-size:16px;margin:0 8px 0 0}select,button{font:inherit;color:inherit;background:var(--chip);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
button{cursor:pointer}button:hover{border-color:var(--acc)}.mut{color:var(--mut)}
main{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:16px;padding:16px;max-width:1400px;margin:0 auto}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
#img{width:100%;max-height:62vh;object-fit:contain;background:var(--chip);border-radius:8px;display:block}
.lab{font-size:20px;font-weight:650}.big{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.row{display:flex;gap:8px;align-items:center;margin:4px 0}.bar{flex:1;height:10px;background:var(--chip);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.row.is-label .nm{font-weight:700;color:var(--ok)}.nm{width:170px;flex:none}.pct{width:48px;text-align:right;font-variant-numeric:tabular-nums}
.acts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.acts button{padding:10px;text-align:left}
.acts kbd{display:inline-block;min-width:20px;text-align:center;border:1px solid var(--line);border-radius:4px;margin-right:6px;font-size:12px}
button.sel{outline:2px solid var(--acc)}
.refs{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.refs img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:var(--chip)}
h3{margin:12px 0 6px;font-size:13px}.stat{font-variant-numeric:tabular-nums}.tag{display:inline-block;padding:1px 8px;border-radius:10px;background:var(--chip);font-size:12px}
.tag.bad{color:var(--bad)}.tag.ok{color:var(--ok)}a{color:var(--acc)}
</style></head><body>
<header><h1>Duyệt nhãn val/test</h1>
<label>Hàng đợi <select id="mode"><option value="errors">Lỗi tự tin nhất</option><option value="random">Mẫu ngẫu nhiên</option></select></label>
<label>Split <select id="split"><option value="">tất cả</option><option>validation</option><option>test</option></select></label>
<label>Nguồn <select id="prov"><option value="">tất cả</option><option>web</option><option>pexels</option><option>wikimedia_commons</option></select></label>
<label>Lớp <select id="cls"><option value="">tất cả</option></select></label>
<label><input type="checkbox" id="hideDone" checked> ẩn ảnh đã duyệt</label>
<span class="mut" id="pos"></span></header>
<main><section class="card"><img id="img" alt="">
<div class="mut" id="meta" style="margin-top:6px"></div></section>
<section>
<div class="card"><div class="big">Nhãn hiện tại</div><div class="lab" id="label"></div>
<div id="status" style="margin:4px 0 10px"></div>
<div class="big">Model đoán (top-5)</div><div id="top5"></div>
<div class="acts">
<button data-d="label_ok"><kbd>1</kbd>Nhãn đúng (model sai)</button>
<button data-d="label_wrong_pred"><kbd>2</kbd>Nhãn sai → đúng là <b id="predname"></b></button>
<button data-d="label_wrong_other"><kbd>3</kbd>Nhãn sai → lớp khác: <select id="other"></select></button>
<button data-d="remove"><kbd>4</kbd>Loại bỏ (không phải món / nhiều món / không rõ)</button>
<button data-d="skip"><kbd>0</kbd>Bỏ qua (không chắc)</button>
<button id="prev">← Ảnh trước</button></div></div>
<div class="card" style="margin-top:12px"><h3 id="refLabelT"></h3><div class="refs" id="refLabel"></div>
<h3 id="refPredT"></h3><div class="refs" id="refPred"></div>
<div class="mut" style="font-size:12px;margin-top:6px">Ảnh mẫu lấy từ tập train, ưu tiên ảnh 30VNFoods (nhãn tay).</div></div>
<div class="card stat" style="margin-top:12px" id="stats"></div>
</section></main>
<script>
let S=null, queue=[], i=0;
const $=id=>document.getElementById(id), nm=c=>S.names[c]||c;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){S=await (await fetch('/api/state')).json();
 for(const c of S.classes){$('cls').add(new Option(nm(c),c));$('other').add(new Option(nm(c),c));}
 build();}
function build(){const m=$('mode').value,sp=$('split').value,pv=$('prov').value,cl=$('cls').value,hide=$('hideDone').checked;
 let q=S.items.filter(x=>(!sp||x.split==sp)&&(!pv||x.provider==pv)&&(!cl||x.label==cl||x.pred==cl));
 if(m=='errors'){q=q.filter(x=>!x.correct).sort((a,b)=>b.pred_prob-a.pred_prob)}else{q.sort((a,b)=>a.random_rank-b.random_rank)}
 if(hide)q=q.filter(x=>!S.reviews[x.relative_path]);
 queue=q;i=0;show();stats();}
function refs(el,c){el.innerHTML=(S.references[c]||[]).map(p=>`<img loading="lazy" alt="" src="/img/${encodeURI(p)}">`).join('')}
function show(){const x=queue[i];$('pos').textContent=queue.length?`${i+1} / ${queue.length}`:'Hết ảnh trong hàng đợi';
 if(!x){$('img').removeAttribute('src');return}
 $('img').src='/img/'+encodeURI(x.relative_path);$('label').textContent=nm(x.label);
 const link=/^https?:\/\//.test(x.source_url)?` · <a href="${esc(x.source_url)}" target="_blank" rel="noopener noreferrer">link gốc</a>`:'';
 $('meta').innerHTML=`${esc(x.split)} · nguồn <b>${esc(x.provider)}</b>${link} · <span class="mut">${esc(x.relative_path)}</span>`;
 $('status').innerHTML=x.correct?'<span class="tag ok">model đoán đúng</span>':`<span class="tag bad">model đoán sai</span> <span class="mut">xác suất cho nhãn hiện tại: ${(x.label_prob*100).toFixed(1)}%</span>`;
 $('top5').innerHTML=x.top5.map(([c,p])=>`<div class="row ${c==x.label?'is-label':''}"><span class="nm">${esc(nm(c))}</span><span class="bar"><i style="width:${p*100}%"></i></span><span class="pct">${(p*100).toFixed(0)}%</span></div>`).join('');
 $('predname').textContent=nm(x.pred);const r=S.reviews[x.relative_path];
 $('other').value=(r&&r.corrected_label)||(x.top5.find(([c])=>c!=x.label&&c!=x.pred)||[x.pred])[0];
 document.querySelectorAll('.acts button[data-d]').forEach(b=>b.classList.toggle('sel',!!r&&r.decision==b.dataset.d));
 $('refLabelT').textContent='Mẫu train: '+nm(x.label);refs($('refLabel'),x.label);
 $('refPredT').textContent='Mẫu train: '+nm(x.pred);refs($('refPred'),x.pred);}
async function decide(d){const x=queue[i];if(!x)return;
 const body={relative_path:x.relative_path,decision:d,split:x.split,label:x.label,pred:x.pred,provider:x.provider,
  corrected_label:d=='label_wrong_pred'?x.pred:d=='label_wrong_other'?$('other').value:null,queue:$('mode').value};
 const res=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!res.ok){alert('Lưu thất bại: '+await res.text());return}
 S.reviews[x.relative_path]=await res.json();
 if($('hideDone').checked){queue.splice(i,1);if(i>=queue.length)i=Math.max(0,queue.length-1)}else i=Math.min(i+1,queue.length-1);
 show();stats();}
function stats(){const all=Object.values(S.reviews),r=all.filter(v=>v.decision!='skip');
 const grp=k=>{const g={};for(const v of r){const e=(g[v[k]]=g[v[k]]||{n:0,bad:0,rm:0});e.n++;if(v.decision.startsWith('label_wrong'))e.bad++;if(v.decision=='remove')e.rm++}return g};
 const tbl=(t,g)=>`<h3>${t}</h3>`+Object.entries(g).map(([k,v])=>`<div>${esc(k)}: ${v.n} ảnh · nhãn sai <b>${(100*v.bad/v.n).toFixed(0)}%</b> · loại bỏ ${(100*v.rm/v.n).toFixed(0)}%</div>`).join('');
 $('stats').innerHTML=`<div>Đã duyệt: <b>${r.length}</b> (bỏ qua: ${all.length-r.length})</div>`+
  tbl('Theo hàng đợi — tỉ lệ nhãn bẩn thật xem ở dòng "random"',grp('queue'))+tbl('Theo nguồn',grp('provider'));}
document.querySelectorAll('.acts button[data-d]').forEach(b=>b.onclick=e=>{if(e.target.tagName!='SELECT')decide(b.dataset.d)});
$('prev').onclick=()=>{i=Math.max(0,i-1);show()};
['mode','split','prov','cls','hideDone'].forEach(id=>$(id).onchange=()=>build());
document.addEventListener('keydown',e=>{if(e.target.tagName=='SELECT')return;
 const k={'1':'label_ok','2':'label_wrong_pred','3':'label_wrong_other','4':'remove','0':'skip'}[e.key];
 if(k){e.preventDefault();decide(k)}else if(e.key=='ArrowRight'){i=Math.min(i+1,queue.length-1);show()}else if(e.key=='ArrowLeft'){i=Math.max(0,i-1);show()}});
load();
</script></body></html>"""


def make_handler(args: argparse.Namespace, state: dict, lock: threading.Lock) -> type[BaseHTTPRequestHandler]:
    data_root = args.data_root.resolve()
    valid_paths = {item["relative_path"] for item in state["items"]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if content_type == "image/jpeg" else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/state":
                with lock:
                    self.send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"), "application/json")
            elif path.startswith("/img/"):
                target = (data_root / path[5:]).resolve()
                if data_root not in target.parents or target.suffix.lower() not in {".jpg", ".jpeg"} or not target.is_file():
                    self.send(404, b"not found", "text/plain")
                else:
                    self.send(200, target.read_bytes(), "image/jpeg")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/review":
                self.send(404, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if body.get("relative_path") not in valid_paths or body.get("decision") not in DECISIONS:
                    raise ValueError("unknown image or decision")
                if body["decision"] == "label_wrong_other" and body.get("corrected_label") not in state["classes"]:
                    raise ValueError("corrected_label must be one of the classes")
            except (ValueError, json.JSONDecodeError) as error:
                self.send(400, str(error).encode("utf-8"), "text/plain; charset=utf-8")
                return
            record = {k: body.get(k) for k in ("decision", "corrected_label", "split", "label", "pred", "provider", "queue")}
            record["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with lock:
                state["reviews"][body["relative_path"]] = record
                args.reviews.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.reviews.with_suffix(".tmp")
                temporary.write_text(json.dumps(state["reviews"], ensure_ascii=False, indent=1), encoding="utf-8")
                temporary.replace(args.reviews)
            self.send(200, json.dumps(record, ensure_ascii=False).encode("utf-8"), "application/json")

    return Handler


def main() -> None:
    args = parse_args()
    state = load_state(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args, state, threading.Lock()))
    errors = sum(not item["correct"] for item in state["items"])
    print(f"{len(state['items'])} images ({errors} errors), {len(state['reviews'])} already reviewed")
    print(f"open http://{args.host}:{args.port}  (reviews -> {args.reviews})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
