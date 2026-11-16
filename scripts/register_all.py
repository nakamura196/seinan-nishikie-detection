"""隣り合うコマの位置合わせを全部の組で行う。コマの番号は右から左（コマ k の左端にコマ k+1 の右端が重なる）。
コマ k の左 35% とコマ k+1 の右 35% で SIFT を対応づけ、相似変換（RANSAC 5px）を求める。
→ ../data/register/pairs.tsv（組ごとの対応の数・倍率・回転・ずらし・重なりの幅）と、
   ../data/register/check/<資料>_<k>-<k+1>.jpg（確かめ用。左がコマ k の重なりの帯、右がコマ k+1 の同じ範囲を変換して切り出したもの）。
変換は data/img の画素の座標（コマ k+1 → コマ k）。使い方: python3 register_all.py"""
import os,csv,collections
import numpy as np, cv2
O="../data/register"; os.makedirs(f"{O}/check",exist_ok=True)
cs=collections.defaultdict(set)
for r in csv.DictReader(open("../data/rows.tsv"),delimiter="\t"): cs[r["item"]].add(int(r["ci"]))
s=cv2.SIFT_create(4000); F=0.35
f=open(f"{O}/pairs.tsv","w"); f.write("item\tk\tk2\tmatches\tinliers\tscale\tangle\ttx\tty\toverlap_px\toverlap_frac\n")
for item in sorted(cs):
    c=sorted(cs[item])
    for k in c[:-1]:
        k2=k+1
        if k2 not in cs[item]: continue
        A=cv2.imread(f"../data/img/{item}_{k}.jpg"); B=cv2.imread(f"../data/img/{item}_{k2}.jpg")
        ga,gb=(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) for x in (A,B)); ha,wa=ga.shape; wb=gb.shape[1]
        ma=np.zeros_like(ga); ma[:,:int(wa*F)]=255; mb=np.zeros_like(gb); mb[:,int(wb*(1-F)):]=255
        ka,da=s.detectAndCompute(ga,ma); kb,db=s.detectAndCompute(gb,mb)
        m=[] if da is None or db is None else [p[0] for p in cv2.BFMatcher().knnMatch(db,da,k=2) if len(p)==2 and p[0].distance<0.75*p[1].distance]
        M=None
        if len(m)>=4:
            P=np.float32([kb[x.queryIdx].pt for x in m]); Q=np.float32([ka[x.trainIdx].pt for x in m])
            M,inl=cv2.estimateAffinePartial2D(P,Q,ransacReprojThreshold=5)
        if M is None:
            f.write(f"{item}\t{k}\t{k2}\t{len(m)}\t0\t\t\t\t\t\t\n"); continue
        n=int(inl.sum()); sc=float(np.hypot(M[0,0],M[1,0])); ang=float(np.degrees(np.arctan2(M[1,0],M[0,0])))
        ov=float(M[0,0]*wb+M[0,1]*ha/2+M[0,2])  # コマ k+1 の右端が、コマ k のどの x に来るか
        f.write(f"{item}\t{k}\t{k2}\t{len(m)}\t{n}\t{sc:.4f}\t{ang:.3f}\t{M[0,2]:.1f}\t{M[1,2]:.1f}\t{ov:.0f}\t{ov/wa:.3f}\n")
        x1=int(max(min(ov,wa*F),40)); Bw=cv2.warpAffine(B,M,(wa,ha))
        a,b=A[:,:x1],Bw[:,:x1]; sep=np.full((ha,12,3),255,np.uint8)
        im=np.hstack([a,sep,b]); sc2=900/ha; cv2.imwrite(f"{O}/check/{item}_{k}-{k2}.jpg",cv2.resize(im,None,fx=sc2,fy=sc2),[cv2.IMWRITE_JPEG_QUALITY,80])
f.close()
