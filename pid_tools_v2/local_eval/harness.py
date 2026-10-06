"""규칙 실험: 시험 도면 100장 (조정용 50 / 검증용 50), 정답 기호 사용.
사용법: PID_DEV=<dev_data 폴더> PIDQA=<PIDQA 폴더> python local_eval/harness.py [--quick] [--pred pred_val_base.json]
지표: PID2Graph 정답 그래프의 기호 쌍(pipe) 대비 P/R/F1 + PIDQA 정답률"""
import sys, os, json, pickle, hashlib, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from multiprocessing import Pool
from gtlib import *
from detect_lines import detect_lines
from build_graph import build_graph, symbol_pairs
from pidqa import load_pidqa, make_index, score_sheet
MP=json.load(open(os.path.join(HERE,'p2g_map.json')))   # HF 도면 번호 -> PID2Graph 'Dataset PID' 번호
SHEETS=sorted(int(s) for s in MP)
DEV_SHEETS, TEST = SHEETS[0::2], SHEETS[1::2]   # 조정용 / 검증용
QS=None
CACHE=os.environ.get('PID_CACHE', os.path.join(HERE,'cache')); os.makedirs(CACHE,exist_ok=True)

def segs_for(sh, boxes_key, boxes, dp):
    # 기본값이 바뀌어도 예전 캐시를 쓰지 않도록, 생략한 인자까지 채워서 열쇠를 만든다
    full={k:v.default for k,v in inspect.signature(detect_lines).parameters.items() if v.default is not inspect.Parameter.empty}
    full.update(dp)
    key=hashlib.md5(json.dumps([sh,boxes_key,full],sort_keys=True).encode()).hexdigest()
    f=f'{CACHE}/{key}.pkl'
    if os.path.exists(f): return pickle.load(open(f,'rb'))
    gray=cv2.imread(f'{HF}/images/val/{sh}.jpg',0)
    segs=detect_lines(gray, boxes, **dp); pickle.dump(segs,open(f,'wb')); return segs

def run_sheet(args):
    sh, dp, gp, pred = args
    global QS
    if QS is None: QS=load_pidqa(PIDQA_DIR)
    labels,_=sheet_labels(sh)
    syms = labels if pred is None else pred
    segs=segs_for(sh, 'gt' if pred is None else 'pred'+str(len(pred)), [s['box'] for s in syms], dp)
    G=build_graph(syms, segs, img_w=7168, **gp)
    Gt=load_gt(f"{P2G}/Dataset PID/{MP[str(sh)]}.graphml")
    mp,_=match_gt_to_labels(Gt,labels)
    T={frozenset(f'S{mp[x]}' for x in p) for p in gt_pairs(Gt,'pipe')}
    out={'sheet':sh}
    for mode in ('pipe',):
        P=symbol_pairs(G,mode)
        if pred is not None:   # 예측 기호 -> 라벨 인덱스로 바꿔 비교
            from common import iou_matrix
            M=iou_matrix([s['box'] for s in syms],[l['box'] for l in labels])
            ren={f'S{i}':(f'S{int(M[i].argmax())}' if M[i].max()>=0.5 else f'Z{i}') for i in range(len(syms))}
            Pm={frozenset(ren[x] for x in p) for p in P}
        else: Pm=P
        out['tp']=len(Pm&T); out['fp']=len(Pm-T); out['fn']=len(T-Pm)
        cls={f'S{i}':s['cls'] for i,s in enumerate(syms)}
        c,a=make_index(cls,P,1); sc=score_sheet(QS,sh,c,a)
        out.update({k+'_ok':v[0] for k,v in sc.items()}); out.update({k+'_n':v[1] for k,v in sc.items()})
    out['nseg']=len(segs)
    return out

def evaluate(dp, gp, sheets, pred=None, workers=2):
    jobs=[(sh,dp,gp,None if pred is None else pred[f'{sh}.jpg']) for sh in sheets]
    with Pool(workers) as p: rows=p.map(run_sheet, jobs)
    tp=sum(r['tp'] for r in rows); fp=sum(r['fp'] for r in rows); fn=sum(r['fn'] for r in rows)
    P=tp/max(tp+fp,1); R=tp/max(tp+fn,1)
    return {'P':round(P,4),'R':round(R,4),'F1':round(2*P*R/max(P+R,1e-9),4),
            'count':round(sum(r['count_ok'] for r in rows)/sum(r['count_n'] for r in rows),4),
            'connect':round(sum(r['connect_ok'] for r in rows)/sum(r['connect_n'] for r in rows),4),
            'tp':tp,'fp':fp,'fn':fn,'rows':rows}

V1_D=dict(stub_len=0,char_max=60); V1_G=dict(cross=False,keep_tjunc=False,ext_dashed=0,graze=False,pad=8)
STEPS=[('v1 (기존)',V1_D,V1_G),
       ('+십자 연결점',V1_D,dict(V1_G,cross=True)),
       ('+기호 옆 분기 유지',V1_D,dict(V1_G,cross=True,keep_tjunc=True)),
       ('+점선 끝 늘리기',V1_D,dict(V1_G,cross=True,keep_tjunc=True,ext_dashed=30)),
       ('+짧은 선(stub)',dict(stub_len=20,char_max=60),dict(V1_G,cross=True,keep_tjunc=True,ext_dashed=30)),
       ('+글자 기준 40',dict(stub_len=20,char_max=40),dict(V1_G,cross=True,keep_tjunc=True,ext_dashed=30)),
       ('+스침 제외·pad6 (v2)',{},{})]

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description='연결 규칙 단계별 효과: 조정용 50장 / 검증용 50장')
    ap.add_argument('--pred',default='',help='predict_sheets.py 결과 json (YOLO 기호로도 평가)')
    ap.add_argument('--quick',action='store_true',help='v1과 v2만, 검증용 50장')
    a=ap.parse_args()
    steps=[STEPS[0],STEPS[-1]] if a.quick else STEPS
    splits=(('TEST',TEST),) if a.quick else (('DEV',DEV_SHEETS),('TEST',TEST))
    for name,dp,gp in steps:
        for split,sheets in splits:
            r=evaluate(dp,gp,sheets); r.pop('rows')
            print(f"{name:22}{split:5} P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} | PIDQA count {r['count']:.3f} connect {r['connect']:.3f} | fp {r['fp']} fn {r['fn']}",flush=True)
    if a.pred:
        pred=json.load(open(a.pred))['pred']
        for name,dp,gp in (STEPS[0],STEPS[-1]):
            r=evaluate(dp,gp,SHEETS,pred=pred); r.pop('rows')
            print(f"YOLO 기호 {name:16}ALL   P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} | PIDQA count {r['count']:.3f} connect {r['connect']:.3f}",flush=True)
