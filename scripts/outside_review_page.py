"""注釈外の枠の手検査用の点検表（index.html）を作る。1 件ずつ切り出した画像に Claude の一次判定を添え、違うものだけ選び直す。
使い方: python3 outside_review_page.py <判定の tsv（outside_sheet_main.py の出力に judge を書いたもの）>
→ tsv と同じフォルダに crops/<番号>.jpg と index.html。選び直した結果は「TSV をコピー」で取り出し、tsv の judge 欄に author: として書き戻す"""
import sys,os,csv,collections,html
from PIL import Image,ImageDraw
tsv=sys.argv[1]; d=os.path.dirname(tsv); os.makedirs(f"{d}/crops",exist_ok=True)
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): rows[(r["item"],r["ci"])].append(r)
import json
S={(x["item"],str(x["ci"])):x["scale"] for x in map(json.loads,open("../data/main_cv_all.jsonl"))}
R=list(csv.DictReader(open(tsv),delimiter="\t"))
LAB=[("scene","場面の中の文字（注釈の付け漏れ）"),("margin","余白の出版事項・署名・届"),("picture","絵・文字でないもの"),("unclear","判断できない")]
cards=[]
for r in R:
    k=(r["item"],r["ci"]); s=S[k]; x1,y1,x2,y2=(float(r[c])/s for c in ["x1","y1","x2","y2"])
    im=Image.open(f"../data/img/{k[0]}_{k[1]}.jpg").convert("RGB"); D=ImageDraw.Draw(im)
    for a in rows[k]:
        ax,ay,aw,ah=(float(a[c])/s for c in "xywh"); D.rectangle([ax,ay,ax+aw,ay+ah],outline=(230,0,0),width=2)
    D.rectangle([x1,y1,x2,y2],outline=(0,80,255),width=4)
    m=max(max(x2-x1,y2-y1)*1.0+60,200); c=im.crop((int(x1-m),int(y1-m),int(x2+m),int(y2+m))); c.thumbnail((520,520)); c.save(f"{d}/crops/{r['no']}.jpg",quality=85)
    j=r["judge"].split(":")[-1]
    opts="".join(f'<option value="{v}"{" selected" if v==j else ""}>{t}</option>' for v,t in LAB)
    cards.append(f'<div class="c" data-no="{r["no"]}" data-cl="{j}"><div class="h">{r["no"]}　<a href="https://dcb-one.vercel.app/ja/s/seinan/item/{k[0]}/" target="_blank">{html.escape(k[0])}</a> p{k[1]}　確信度 {float(r["conf"]):.2f}</div><img src="crops/{r["no"]}.jpg" loading="lazy"><select>{opts}</select></div>')
page=f"""<!doctype html><meta charset="utf-8"><title>注釈外の枠の点検</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fafafa}}.g{{display:flex;flex-wrap:wrap;gap:12px}}.c{{background:#fff;border:2px solid #ddd;padding:6px;width:300px}}.c.ch{{border-color:#e69500}}.c img{{max-width:300px;max-height:300px;display:block;margin:4px 0}}.h{{font-size:12px}}select{{width:100%;font-size:14px}}#bar{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ccc;margin-bottom:12px}}</style>
<div id="bar">青の枠が検出、赤の枠が正解の注釈。初期値は Claude の一次判定。違うものだけ選び直す（直したものは橙の枠になる）。資料番号を押すと、その資料のページが別のタブで開く（何面目かは番号の後の p）。　<button onclick="cp()">TSV をコピー</button> <span id="n"></span></div>
<div class="g">{''.join(cards)}</div>
<script>
const K='outside-check-'+location.pathname;
function sv(){{try{{localStorage.setItem(K,JSON.stringify([...document.querySelectorAll('.c')].map(c=>c.querySelector('select').value)))}}catch(e){{}}}}
function up(){{let n=0;document.querySelectorAll('.c').forEach(c=>{{const ch=c.querySelector('select').value!==c.dataset.cl;c.classList.toggle('ch',ch);n+=ch}});document.getElementById('n').textContent='直した件数: '+n}}
try{{const v=JSON.parse(localStorage.getItem(K)||'null');if(v)document.querySelectorAll('.c select').forEach((s,i)=>s.value=v[i])}}catch(e){{}}
document.querySelectorAll('.c select').forEach(s=>s.onchange=()=>{{sv();up()}});up();
function cp(){{const t=[...document.querySelectorAll('.c')].map(c=>c.dataset.no+'\\t'+c.querySelector('select').value).join('\\n');navigator.clipboard.writeText(t).then(()=>alert('コピーしました'))}}
</script>"""
open(f"{d}/index.html","w").write(page); print(len(R),"件",f"{d}/index.html")
