"""PID2Graph 정답 그래프 읽기 + YOLO 라벨과 맞추기 + 기호 쌍 뽑기

경로는 환경 변수로 지정한다.
  PID_DEV : dev_data.tgz 를 푼 폴더 (data/hf/..., data/pid2graph/... 가 들어 있음). 기본 ./dev_data
  PIDQA   : github.com/mgupta70/PIDQA 를 clone 한 폴더. 기본 ./PIDQA
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
DEV = os.environ.get('PID_DEV', 'dev_data')
PIDQA_DIR = os.environ.get('PIDQA', 'PIDQA')
import networkx as nx, numpy as np
from common import read_yolo_labels, iou_matrix, image_size
HF=f'{DEV}/data/hf/DigitizePID_Dataset'
P2G=f'{DEV}/data/pid2graph/PID2Graph/Complete'
NONSYM={'connector','crossing','background'}

def load_gt(path):
    G=nx.Graph(nx.read_graphml(path))
    for n,d in G.nodes(data=True):
        d['box']=[float(d['xmin']),float(d['ymin']),float(d['xmax']),float(d['ymax'])]
        d['sym']=d['label'] not in NONSYM
    G.remove_nodes_from([n for n,d in G.nodes(data=True) if d['label']=='background'])
    return G

def gt_pairs(G, mode='pipe'):
    syms=[n for n,d in G.nodes(data=True) if d['sym']]
    pairs=set()
    for s0 in syms:
        seen,stack={s0},[s0]
        while stack:
            u=stack.pop()
            for v in G.neighbors(u):
                if v in seen: continue
                seen.add(v)
                if G.nodes[v]['sym']: pairs.add(frozenset((s0,v))); continue
                if mode=='pipe' and G.degree(v)>=3: continue
                stack.append(v)
    return pairs

def sheet_labels(sheet):
    ip=f'{HF}/images/val/{sheet}.jpg'; W,H=image_size(ip)
    return read_yolo_labels(f'{HF}/labels/val/{sheet}.txt',W,H), ip

def match_gt_to_labels(G, labels):
    """정답 그래프의 기호 노드 -> 라벨 인덱스 (IoU 최대). 반환 {node: idx}"""
    syms=[n for n,d in G.nodes(data=True) if d['sym']]
    M=iou_matrix([G.nodes[n]['box'] for n in syms],[l['box'] for l in labels])
    out={}
    for i,n in enumerate(syms):
        j=int(np.argmax(M[i]))
        if M[i,j]>=0.5: out[n]=j
    return out, len(syms)
