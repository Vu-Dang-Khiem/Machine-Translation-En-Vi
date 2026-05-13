# -*- coding: utf-8 -*-
"""
Build standalone HTML file for Human Evaluation.
Embeds all sample data inline so it works as a single file.
Can be hosted on GitHub Pages for public access.

Run: python human_evaluation/build_html.py
Output: human_evaluation/index.html
"""

import json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_FILE = os.path.join(BASE_DIR, "eval_samples.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "index.html")

# Load samples
with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

samples_json = json.dumps(data["samples"], ensure_ascii=False)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Human Evaluation — MT EN↔VI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Inter,system-ui,sans-serif;background:#f8fafc;color:#1e293b;min-height:100vh}
.header{background:linear-gradient(135deg,#059669,#0d9488,#0891b2);color:#fff;text-align:center;padding:28px 20px;border-radius:0 0 20px 20px;box-shadow:0 8px 30px rgba(5,150,105,.3)}
.header h1{font-size:1.8em;margin-bottom:4px}
.header p{opacity:.85;font-size:.95em}
.container{max-width:900px;margin:20px auto;padding:0 16px}
.setup{background:#fff;border-radius:14px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.setup label{font-weight:600;font-size:.9em;color:#475569}
.setup input,.setup select{padding:8px 14px;border:1px solid #cbd5e1;border-radius:8px;font-size:.95em;outline:none}
.setup input:focus,.setup select:focus{border-color:#059669;box-shadow:0 0 0 3px rgba(5,150,105,.15)}
.progress-bar{background:#fff;border-radius:12px;padding:14px 20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06);display:flex;align-items:center;gap:14px}
.progress-track{flex:1;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,#059669,#0891b2);border-radius:5px;transition:width .3s}
.progress-text{font-weight:700;color:#059669;min-width:100px;text-align:right}
.card{background:#fff;border-radius:14px;padding:24px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:8px}
.badge{background:#e0f2fe;color:#0369a1;padding:4px 12px;border-radius:20px;font-size:.8em;font-weight:600}
.badge.scored{background:#d1fae5;color:#065f46}
.text-label{font-weight:700;font-size:.78em;text-transform:uppercase;letter-spacing:.5px;color:#64748b;margin-bottom:6px}
.text-label.pred{color:#d97706}
.text-box{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin-bottom:14px;font-size:1.02em;line-height:1.65;min-height:50px;word-wrap:break-word}
.text-box.pred{background:#fef3c7;border-color:#fbbf24}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.two-col{grid-template-columns:1fr}}
.score-section{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:16px 0}
@media(max-width:700px){.score-section{grid-template-columns:1fr}}
.score-group label{display:block;font-weight:600;margin-bottom:8px;font-size:.95em}
.score-btns{display:flex;gap:6px}
.score-btn{width:48px;height:48px;border:2px solid #cbd5e1;border-radius:10px;background:#fff;font-size:1.15em;font-weight:700;cursor:pointer;transition:all .15s}
.score-btn:hover{border-color:#059669;background:#ecfdf5}
.score-btn.active{background:#059669;color:#fff;border-color:#059669}
.score-desc{font-size:.75em;color:#94a3b8;margin-top:4px}
.note-input{width:100%;padding:10px 14px;border:1px solid #cbd5e1;border-radius:8px;font-size:.95em;margin:10px 0;outline:none}
.note-input:focus{border-color:#059669}
.nav{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.btn{padding:10px 20px;border:none;border-radius:10px;font-size:.95em;font-weight:600;cursor:pointer;transition:all .15s}
.btn-primary{background:linear-gradient(135deg,#059669,#0891b2);color:#fff;flex:2}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(5,150,105,.3)}
.btn-secondary{background:#e2e8f0;color:#475569;flex:1}
.btn-secondary:hover{background:#cbd5e1}
.btn-download{background:linear-gradient(135deg,#7c3aed,#a855f7);color:#fff}
.btn-download:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(124,58,237,.3)}
.toast{position:fixed;bottom:20px;right:20px;background:#059669;color:#fff;padding:12px 20px;border-radius:10px;font-weight:600;transform:translateY(100px);opacity:0;transition:all .3s;z-index:999}
.toast.show{transform:translateY(0);opacity:1}
.guide{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:14px 18px;margin-bottom:16px;font-size:.88em;line-height:1.7}
.guide h4{color:#059669;margin-bottom:6px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:16px}
.stat-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center}
.stat-card .val{font-size:1.4em;font-weight:800;color:#059669}
.stat-card .lbl{font-size:.75em;color:#64748b;text-transform:uppercase}
.footer{text-align:center;padding:20px;color:#94a3b8;font-size:.82em}
</style>
</head>
<body>

<div class="header">
<h1>🧑‍🔬 Human Evaluation</h1>
<p>Machine Translation EN ↔ VI &nbsp;|&nbsp; Adequacy + Fluency (1-5)</p>
</div>

<div class="container">

<div class="setup">
<div><label>👤 Tên của bạn:</label><br><input id="nameInput" value="" placeholder="VD: Khiem, Minh..."></div>
<div><label>🔄 Đánh giá chiều:</label><br>
<select id="dirSelect"><option value="envi">EN → VI</option><option value="vien">VI → EN</option></select></div>
</div>

<div class="guide">
<h4>📋 Hướng dẫn chấm điểm</h4>
<b>Adequacy:</b> 5=Đầy đủ nghĩa | 4=Hầu hết | 3=Nhiều nhưng thiếu | 2=Một phần | 1=Sai hết<br>
<b>Fluency:</b> 5=Hoàn hảo | 4=Tốt | 3=Hiểu được | 2=Khó đọc | 1=Không đọc được<br>
<em>So sánh <b>Model Prediction</b> (ô vàng) với <b>Source</b> để chấm. Reference chỉ để tham khảo.</em>
</div>

<div class="progress-bar">
<div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
<div class="progress-text" id="progressText">0/100</div>
</div>

<div class="nav">
<button class="btn btn-secondary" onclick="go(-1)">⬅️ Trước</button>
<button class="btn btn-secondary" onclick="go(1)">➡️ Sau</button>
<button class="btn btn-primary" onclick="submitAndNext()">✅ Chấm & Tiếp theo</button>
<button class="btn btn-secondary" onclick="goUnscored()">⏭️ Chưa chấm</button>
</div>

<div class="card" id="sampleCard">
<div class="card-header">
<span id="sampleInfo" style="font-weight:700;font-size:1.1em"></span>
<span class="badge" id="statusBadge">⏳ Chưa chấm</span>
</div>
<div class="text-label" id="srcLabel">📝 SOURCE</div>
<div class="text-box" id="srcText"></div>
<div class="two-col">
<div><div class="text-label" id="refLabel">📖 REFERENCE</div><div class="text-box" id="refText"></div></div>
<div><div class="text-label pred">🤖 MODEL PREDICTION — CẦN ĐÁNH GIÁ</div><div class="text-box pred" id="predText"></div></div>
</div>

<div class="score-section">
<div class="score-group"><label>⭐ Adequacy (Đầy đủ nghĩa)</label>
<div class="score-btns" id="adeqBtns"></div>
<div class="score-desc">1=Sai hết → 5=Đầy đủ</div></div>
<div class="score-group"><label>✨ Fluency (Trôi chảy)</label>
<div class="score-btns" id="fluBtns"></div>
<div class="score-desc">1=Không đọc được → 5=Hoàn hảo</div></div>
</div>

<input class="note-input" id="noteInput" placeholder="📝 Ghi chú (tùy chọn): VD sai tên riêng, thiếu từ...">
</div>

<div class="nav">
<button class="btn btn-download" onclick="downloadResults()">📥 Tải kết quả (JSON)</button>
<button class="btn btn-secondary" onclick="showStats()">📊 Xem thống kê</button>
</div>

<div id="statsArea" style="display:none" class="card">
<h3 style="margin-bottom:12px">📊 Thống kê</h3>
<div class="stats" id="statsCards"></div>
</div>

<div class="footer">Built by <b>Khiem</b> | Transformer Machine Translation EN ↔ VI</div>
</div>

<div class="toast" id="toast"></div>

<script>
const SAMPLES=__SAMPLES_DATA__;
let idx=0,adeq=4,flu=4;
const key=()=>`he_${document.getElementById('dirSelect').value}`;
function getAnn(){try{return JSON.parse(localStorage.getItem(key()))||{}}catch(e){return{}}}
function setAnn(a){localStorage.setItem(key(),JSON.stringify(a))}
function render(){
const s=SAMPLES[idx],d=document.getElementById('dirSelect').value,ann=getAnn();
const scored=ann[s.id];
document.getElementById('sampleInfo').textContent=`Câu ${idx+1}/${SAMPLES.length} — #${s.id} | ${s.length_group.toUpperCase()} | ${s.source_word_count}w | ${d==='envi'?'EN→VI':'VI→EN'}`;
const badge=document.getElementById('statusBadge');
if(scored){badge.textContent='✅ Đã chấm';badge.className='badge scored'}else{badge.textContent='⏳ Chưa chấm';badge.className='badge'}
if(d==='envi'){
document.getElementById('srcLabel').textContent='📝 SOURCE (English)';
document.getElementById('refLabel').textContent='📖 REFERENCE (Vietnamese)';
document.getElementById('srcText').textContent=s.source_en;
document.getElementById('refText').textContent=s.reference_vi;
document.getElementById('predText').textContent=s.prediction_envi||'[N/A]';
}else{
document.getElementById('srcLabel').textContent='📝 SOURCE (Vietnamese)';
document.getElementById('refLabel').textContent='📖 REFERENCE (English)';
document.getElementById('srcText').textContent=s.reference_vi;
document.getElementById('refText').textContent=s.source_en;
document.getElementById('predText').textContent=s.prediction_vien||'[N/A]';
}
adeq=scored?scored.adequacy:4;flu=scored?scored.fluency:4;
document.getElementById('noteInput').value=scored?scored.note||'':'';
renderBtns();updateProgress();
}
function renderBtns(){
['adeqBtns','fluBtns'].forEach((id,i)=>{
const v=i===0?adeq:flu;const c=document.getElementById(id);c.innerHTML='';
for(let n=1;n<=5;n++){const b=document.createElement('button');b.className='score-btn'+(n===v?' active':'');b.textContent=n;
b.onclick=()=>{if(i===0)adeq=n;else flu=n;renderBtns()};c.appendChild(b)}});
}
function updateProgress(){
const ann=getAnn(),done=Object.keys(ann).length,total=SAMPLES.length;
document.getElementById('progressFill').style.width=(done/total*100)+'%';
document.getElementById('progressText').textContent=`${done}/${total} (${Math.round(done/total*100)}%)`;
}
function go(d){idx=Math.max(0,Math.min(SAMPLES.length-1,idx+d));render()}
function goUnscored(){const ann=getAnn();for(let i=idx+1;i<SAMPLES.length;i++){if(!ann[SAMPLES[i].id]){idx=i;render();return}}
for(let i=0;i<idx;i++){if(!ann[SAMPLES[i].id]){idx=i;render();return}}toast('🎉 Đã chấm hết!')}
function submitAndNext(){
const s=SAMPLES[idx],ann=getAnn(),name=document.getElementById('nameInput').value.trim()||'anonymous';
ann[s.id]={adequacy:adeq,fluency:flu,note:document.getElementById('noteInput').value.trim(),annotator:name,time:new Date().toISOString()};
setAnn(ann);toast(`✅ #${s.id}: Adeq=${adeq}, Flu=${flu}`);
if(idx<SAMPLES.length-1){const a2=getAnn();let ni=idx+1;for(let i=idx+1;i<SAMPLES.length;i++){if(!a2[SAMPLES[i].id]){ni=i;break}}idx=ni}render();
}
function downloadResults(){
const ann=getAnn(),name=document.getElementById('nameInput').value.trim()||'anonymous',dir=document.getElementById('dirSelect').value;
const out={annotator:name,direction:dir,total:SAMPLES.length,completed:Object.keys(ann).length,
timestamp:new Date().toISOString(),annotations:ann};
const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`${name}_${dir}.json`;a.click();
toast('📥 File đã tải!');}
function showStats(){
const area=document.getElementById('statsArea'),ann=getAnn(),vals=Object.values(ann);
if(!vals.length){area.style.display='block';document.getElementById('statsCards').innerHTML='<p>Chưa có điểm nào</p>';return}
const aa=vals.map(v=>v.adequacy),ff=vals.map(v=>v.fluency);
const avg=a=>a.length?(a.reduce((s,v)=>s+v,0)/a.length).toFixed(2):'-';
area.style.display='block';
document.getElementById('statsCards').innerHTML=`
<div class="stat-card"><div class="val">${vals.length}</div><div class="lbl">Đã chấm</div></div>
<div class="stat-card"><div class="val">${avg(aa)}</div><div class="lbl">Adequacy TB</div></div>
<div class="stat-card"><div class="val">${avg(ff)}</div><div class="lbl">Fluency TB</div></div>
<div class="stat-card"><div class="val">${avg([...aa,...ff])}</div><div class="lbl">Overall</div></div>`;
}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2500)}
document.getElementById('dirSelect').onchange=render;
render();
</script>
</body>
</html>"""

# Inject data
html = HTML_TEMPLATE.replace("__SAMPLES_DATA__", samples_json)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(OUTPUT_FILE) / 1024
print(f"Created: {OUTPUT_FILE}")
print(f"Size: {size_kb:.0f} KB")
print(f"Samples: {len(data['samples'])}")
