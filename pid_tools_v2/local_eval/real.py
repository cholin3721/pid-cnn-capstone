"""실제 도면(PID2Graph OPEN100)에서 연결 규칙 평가. 사용법: PID_DEV=... python local_eval/real.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import statistics
OP=f'{P2G}/PID2Graph OPEN100'
def run_real(i, dp, gp, ref=86.0, dbg=None):
    G=load_gt(f'{OP}/{i}.graphml'); ids=[n for n,d in G.nodes(data=True) if d['sym']]
    syms=[{'cls':0,'box':G.nodes[n]['box'],'label':G.nodes[n]['label']} for n in ids]
    med=statistics.median(max(s['box'][2]-s['box'][0], s['box'][3]-s['box'][1]) for s in syms)
    ps=med/ref
    gray=cv2.imread(f'{OP}/{i}.png',0)
    segs=detect_lines(gray,[s['box'] for s in syms],px_scale=ps,**dp)
    H=build_graph(syms,segs,px_scale=ps,**gp)
    ren={f'S{k}':n for k,n in enumerate(ids)}
    P={frozenset(ren[x] for x in p) for p in symbol_pairs(H,'pipe')}; T=gt_pairs(G,'pipe')
    if dbg:
        img=cv2.imread(f'{OP}/{i}.png'); vis=img.copy()
        for s in segs: cv2.line(vis,tuple(map(int,s['p1'])),tuple(map(int,s['p2'])),(0,140,255) if s['dashed'] else (0,190,0),2)
        for s in syms: cv2.rectangle(vis,tuple(map(int,s['box'][:2])),tuple(map(int,s['box'][2:])),(255,80,0),1)
        c=lambda n:(int((G.nodes[n]['box'][0]+G.nodes[n]['box'][2])/2),int((G.nodes[n]['box'][1]+G.nodes[n]['box'][3])/2))
        for p in T-P: a,b=tuple(p); cv2.line(vis,c(a),c(b),(255,0,255),2)
        for p in P-T: a,b=tuple(p); cv2.line(vis,c(a),c(b),(0,0,255),2)
        cv2.imwrite(dbg,vis)
    return len(P&T),len(P-T),len(T-P),len(segs),ps,len(syms)
V1=(dict(stub_len=0,char_max=60),dict(cross=False,keep_tjunc=False,ext_dashed=0,graze=False,pad=8))
if __name__=='__main__':
    # 실제 도면(OPEN100 12장)에서 정답 기호를 주고 연결 쌍만 채점
    for name,(dp,gp) in (('v1',V1),('v2',({},{})),('v2+touch',({},dict(touch=0)))):
        tot=[0,0,0]
        for i in range(12):
            tp,fp,fn,*_=run_real(i,dp,gp); tot[0]+=tp; tot[1]+=fp; tot[2]+=fn
        P=tot[0]/max(tot[0]+tot[1],1); R=tot[0]/max(tot[0]+tot[2],1)
        print(f'{name:9} OPEN100 연결 쌍: P {P:.3f} R {R:.3f} F1 {2*P*R/max(P+R,1e-9):.3f}  {tot}')
