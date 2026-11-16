"""隣り合うコマの位置合わせ（試し）。コマ k の左端の帯と、コマ k+1 の右端の帯で特徴点を対応づけ、相似変換を求める。
コマの番号は右から左へ（コマ 1 が右端の紙）。使い方: python3 register_pair.py <資料> <コマ k> <出力.jpg>
→ 対応点の数・変換・重なりの幅を表示し、コマ k+1 の注釈をコマ k に写した図を書く"""
import sys,csv,collections
import numpy as np, cv2
item,k,outp=sys.argv[1],sys.argv[2],sys.argv[3]; k2=str(int(k)+1)
A=cv2.imread(f"../data/img/{item}_{k}.jpg"); B=cv2.imread(f"../data/img/{item}_{k2}.jpg")
ga,gb=(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) for x in (A,B)); wa,wb=A.shape[1],B.shape[1]
F=0.3; ma=np.zeros_like(ga); ma[:,:int(wa*F)]=255; mb=np.zeros_like(gb); mb[:,int(wb*(1-F)):]=255
s=cv2.SIFT_create(4000); ka,da=s.detectAndCompute(ga,ma); kb,db=s.detectAndCompute(gb,mb)
m=[x for x,y in cv2.BFMatcher().knnMatch(db,da,k=2) if x.distance<0.75*y.distance]
P=np.float32([kb[x.queryIdx].pt for x in m]); Q=np.float32([ka[x.trainIdx].pt for x in m])
M,inl=cv2.estimateAffinePartial2D(P,Q,ransacReprojThreshold=5); n=int(inl.sum())
sc=np.hypot(M[0,0],M[1,0]); ang=np.degrees(np.arctan2(M[1,0],M[0,0]))
print(f"対応 {len(m)} 本、正しい対応 {n}、倍率 {sc:.3f}、回転 {ang:.2f}°、ずらし ({M[0,2]:.0f},{M[1,2]:.0f})、重なり（B の右端が A の x={M[0,0]*wb+M[0,2]:.0f}、A の幅 {wa}）")
rows=collections.defaultdict(list)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): rows[(r["item"],r["ci"])].append(r)
# 注釈の座標は元画像の座標。data/img の縮尺との比は幅の比で出す
import json
D={(d["item"],str(d["ci"])):d for d in map(json.loads,open("../data/main_cv_all.jsonl"))}
sa,sb=D[(item,k)]["scale"],D[(item,k2)]["scale"]
for r in rows[(item,k)]:
    x,y,w,h=(float(r[c])/sa for c in "xywh"); cv2.rectangle(A,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,230),3)
t=0
for r in rows[(item,k2)]:
    x,y,w,h=(float(r[c])/sb for c in "xywh")
    p=cv2.transform(np.float32([[[x,y],[x+w,y+h]]]),M)[0]
    if p[1][0]>0 and p[0][0]<wa:
        t+=1; cv2.rectangle(A,tuple(int(v) for v in p[0]),tuple(int(v) for v in p[1]),(0,200,0),4)
print("写した注釈",t)
cv2.imwrite(outp,cv2.resize(A[:, :int(wa*0.35)],None,fx=0.5,fy=0.5),[cv2.IMWRITE_JPEG_QUALITY,80])
