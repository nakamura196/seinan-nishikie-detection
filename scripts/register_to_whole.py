"""各コマを、三枚続を並べて 1 枚に撮った全体の写真（../data/stitched_manual/<資料>.jpg、8272×6200）に合わせる。
SIFT で対応を取り、相似変換（RANSAC 5px）を求める。全体の写真は 1/2 に縮めて使い、変換は「data/img のコマの画素 → 全体の写真の原寸の画素」。
→ ../data/register/whole_reg.tsv（コマごと）と whole_check/<資料>.jpg（全体の写真の上に各コマの外形を描いたもの）。
さらに、2 つのコマの変換から「コマ k+1 → コマ k」を出し、pairs.tsv（コマどうしの直接の位置合わせ）と比べる。使い方: python3 register_to_whole.py"""
import os,glob,csv
import numpy as np, cv2
D="../data"; O=f"{D}/register"; os.makedirs(f"{O}/whole_check",exist_ok=True); R=0.5
s=cv2.SIFT_create(8000); bf=cv2.BFMatcher()
pc={(r["item"],int(r["k"])):r for r in csv.DictReader(open(f"{O}/pair_class.tsv"),delimiter="\t")}
pr={(r["item"],int(r["k"])):r for r in csv.DictReader(open(f"{O}/pairs.tsv"),delimiter="\t")}
f=open(f"{O}/whole_reg.tsv","w"); f.write("item\tk\tmatches\tinliers\tscale\tangle\ttx\tty\n")
g=open(f"{O}/whole_vs_pairs.tsv","w"); g.write("item\tk\tk2\tclass\tdx_whole\tdy_whole\tdx_pair\tdy_pair\tdiff_px\n")
for p in sorted(glob.glob(f"{D}/stitched_manual/*.jpg")):
    it=os.path.basename(p)[:-4]; W=cv2.imread(p); Ws=cv2.resize(W,None,fx=R,fy=R,interpolation=cv2.INTER_AREA)
    kw,dw=s.detectAndCompute(cv2.cvtColor(Ws,cv2.COLOR_BGR2GRAY),None); T={}; vis=cv2.resize(W,None,fx=0.15,fy=0.15)
    for q in sorted(glob.glob(f"{D}/img/{it}_*.jpg")):
        k=int(q.rsplit("_",1)[1][:-4]); A=cv2.imread(q); ka,da=s.detectAndCompute(cv2.cvtColor(A,cv2.COLOR_BGR2GRAY),None)
        m=[x[0] for x in bf.knnMatch(da,dw,k=2) if len(x)==2 and x[0].distance<0.75*x[1].distance]
        if len(m)<4: f.write(f"{it}\t{k}\t{len(m)}\t0\t\t\t\t\n"); continue
        P=np.float32([ka[x.queryIdx].pt for x in m]); Q=np.float32([kw[x.trainIdx].pt for x in m])/R
        M,inl=cv2.estimateAffinePartial2D(P,Q,ransacReprojThreshold=5/R)
        n=int(inl.sum()); sc=float(np.hypot(M[0,0],M[1,0])); ang=float(np.degrees(np.arctan2(M[1,0],M[0,0])))
        f.write(f"{it}\t{k}\t{len(m)}\t{n}\t{sc:.4f}\t{ang:.3f}\t{M[0,2]:.1f}\t{M[1,2]:.1f}\n"); T[k]=(M,A.shape)
        h,w=A.shape[:2]; c=cv2.transform(np.float32([[[0,0],[w,0],[w,h],[0,h]]]),M)[0]*0.15
        cv2.polylines(vis,[c.astype(np.int32)],True,(0,0,255) if k%2 else (255,0,0),4); cv2.putText(vis,str(k),tuple(c.mean(0).astype(int)),0,3,(0,0,255),6)
    cv2.imwrite(f"{O}/whole_check/{it}.jpg",vis,[cv2.IMWRITE_JPEG_QUALITY,80])
    for k in sorted(T):
        if k+1 not in T: continue
        Mk=np.vstack([T[k][0],[0,0,1]]); Mk1=np.vstack([T[k+1][0],[0,0,1]]); X=np.linalg.inv(Mk)@Mk1  # コマ k+1 → コマ k
        r=pr.get((it,k)); cl=pc.get((it,k),{}).get("class","")
        if r and r["tx"]:
            Y=np.array([[float(r["scale"])*np.cos(np.radians(float(r["angle"]))),0,float(r["tx"])],[0,0,float(r["ty"])]])
            # 比べる量: コマ k+1 の中心が、コマ k のどこに来るか
            h,w=T[k+1][1][:2]; cxy=np.array([w/2,h/2,1]); a=X[:2]@cxy
            Mp=np.zeros((2,3)); sc,an=float(r["scale"]),np.radians(float(r["angle"]))
            Mp[:,:2]=[[sc*np.cos(an),-sc*np.sin(an)],[sc*np.sin(an),sc*np.cos(an)]]; Mp[:,2]=[float(r["tx"]),float(r["ty"])]; b=Mp@cxy
            g.write(f"{it}\t{k}\t{k+1}\t{cl}\t{X[0,2]:.1f}\t{X[1,2]:.1f}\t{r['tx']}\t{r['ty']}\t{np.hypot(*(a-b)):.1f}\n")
        else: g.write(f"{it}\t{k}\t{k+1}\t{cl}\t{X[0,2]:.1f}\t{X[1,2]:.1f}\t\t\t\n")
    print(it,{k:int(v[0][0,2]) for k,v in T.items()},flush=True)
f.close(); g.close()
