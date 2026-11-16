"""位置合わせの確かめ: register_all.py の確かめ用画像（左がコマ k の重なりの帯、右がコマ k+1 を変換して同じ範囲を切り出したもの）を
Gemini 4（agy の gemini-4-argon-high、読み取りだけの plan モード）に見せ、同じ内容が同じ位置に写っているかを答えさせる。
→ ../data/register/gemini_check.jsonl（途中から再開できる、raw に返答をそのまま残す）。使い方: python3 check_register_agy.py [並列数（既定 2）]"""
import sys,os,json,re,subprocess,threading,glob
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data","register")
MODEL="gemini-4-argon-high"
PROMPT=("Open the image file {p} . It shows two crops side by side, separated by a thin white bar. Both come from photographs of "
        "adjacent sheets of a Japanese multi-sheet woodblock print, where the sheets overlap. Left: the left edge strip of one photograph. "
        "Right: the same area taken from the neighbouring photograph after automatic alignment (black where that photograph has no pixels). "
        "Some photographs show only one sheet on a background (cutting mat, ruler, colour chart) with no part of the neighbouring sheet; then the alignment may match only the background. Judge ONLY the printed picture, not the background, ruler or colour chart. aligned = true only if the same printed content (figures, text labels, lines) appears in both halves at the same height and position. overlap = true if the printed picture content is shared between the halves at all. "
        'Return ONLY JSON: {{"aligned": true or false, "overlap": true or false, "shift": "none" or "small" or "large", "note": "one short sentence"}}. No other text.')
OUT=f"{D}/gemini_check.jsonl"; done=set()
if os.path.exists(OUT): done={json.loads(l)["file"] for l in open(OUT)}
files=[p for p in sorted(glob.glob(f"{D}/check/*.jpg")) if os.path.basename(p) not in done]
lock=threading.Lock()
def run(p):
    try:
        r=subprocess.run(["agy","--model",MODEL,"--mode","plan","--add-dir",os.path.dirname(p),"--print-timeout","600s","-p",PROMPT.format(p=p)],capture_output=True,text=True,timeout=700)
        raw=r.stdout
    except Exception as e: raw=f"ERROR {e}"
    m=re.search(r"\{[^{}]*\"aligned\"[^{}]*\}",raw,re.S); ans=None
    if m:
        try: ans=json.loads(m.group(0))
        except Exception: pass
    with lock:
        open(OUT,"a").write(json.dumps({"file":os.path.basename(p),"answer":ans,"raw":raw[-2000:]},ensure_ascii=False)+"\n")
print(len(files),"件"); 
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv)>1 else 2) as ex: list(ex.map(run,files))
