import csv,random,json,urllib.parse
rows=list(csv.DictReader(open("rows.tsv"),delimiter="\t"))
random.seed(20261030)
by={k:[r for r in rows if r["pred"]==k] for k in ["札","札?","詞書"]}
short=[r for r in by["詞書"] if len(r["text"])<=10]
long_=[r for r in by["詞書"] if len(r["text"])>10]
S=[("札",r) for r in random.sample(by["札"],80)]+[("札?",r) for r in random.sample(by["札?"],50)]+[("詞書(短)",r) for r in random.sample(short,40)]+[("詞書(長)",r) for r in random.sample(long_,30)]
random.shuffle(S)
out=[]
for n,(stratum,r) in enumerate(S,1):
    x,y,w,h=float(r["x"]),float(r["y"]),float(r["w"]),float(r["h"]); W,H=int(r["W"]),int(r["H"])
    pad=max(w,h)*0.6
    cx,cy=max(0,x-pad),max(0,y-pad); cx2,cy2=min(W,x+w+pad),min(H,y+h+pad)
    reg=f"{int(cx)},{int(cy)},{int(cx2-cx)},{int(cy2-cy)}"
    base=r["img"].replace(" ","%20")
    out.append(dict(n=n,id=r["id"],stratum=stratum,pred=r["pred"],text=r["text"],item=r["item"],ci=r["ci"],user=r["user"],
      url=f"{base}/{reg}/!400,400/0/default.jpg",
      box=[(x-cx)/(cx2-cx),(y-cy)/(cy2-cy),w/(cx2-cx),h/(cy2-cy)],
      region=[int(x),int(y),int(w),int(h)], canvas=r["canvas"]))
json.dump(out,open("sample.json","w"),ensure_ascii=False,indent=1)
print(len(out)); print(out[0]["url"])
