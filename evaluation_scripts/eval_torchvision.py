# 5-class fruit identity
import argparse, json, time
from pathlib import Path
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets as D
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, itertools, csv

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]

def get_model(name, num_classes, ckpt_path=None):
    name=name.lower()
    if name=="resnet50":
        m=torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc=torch.nn.Linear(m.fc.in_features, num_classes)
    elif name in ["efficientnet_b0","efficientnet-b0","effnet_b0"]:
        m=torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1]=torch.nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError("resnet50 | efficientnet_b0")
    if ckpt_path:
        st=torch.load(ckpt_path, map_location="cpu")
        sd=st.get("state_dict", st.get("model_state", st))
        m.load_state_dict(sd, strict=False)
    return m

def get_loader(root, split="test", img_size=224, batch=64, workers=4):
    tfm=T.Compose([T.Resize(256),T.CenterCrop(img_size),T.ToTensor(),T.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    ds=D.ImageFolder(Path(root)/split, transform=tfm)  # labels = top-level folders: apple, banana, ...
    dl=DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return ds, dl

def plot_confusion(cm, names, out):
    import matplotlib.pyplot as plt, numpy as np, itertools
    plt.figure(figsize=(8,6)); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    ticks=np.arange(len(names)); plt.xticks(ticks,names,rotation=45,ha="right"); plt.yticks(ticks,names)
    thr=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v=cm[i,j]; 
        if v>0: plt.text(j,i,str(v),ha="center",color="white" if v>thr else "black",fontsize=8)
    plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()

def save_report(y, yhat, names, out_csv):
    rep=classification_report(y,yhat,target_names=names,output_dict=True,zero_division=0)
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["class","precision","recall","f1","support"])
        for c in names:
            r=rep[c]; w.writerow([c,f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1-score']:.4f}",int(r['support'])])
        w.writerow(["accuracy","","",f"{rep['accuracy']:.4f}",sum(int(rep[c]['support']) for c in names)])

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval().to(device)
    P=[]; H=[]; Y=[]
    for x,y in dl:
        x=x.to(device,non_blocking=True)
        p=torch.softmax(model(x),dim=1).cpu().numpy()
        h=p.argmax(1)
        P.append(p); H.append(h); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(H), np.concatenate(Y)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_prepared")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--model", default="resnet50", choices=["resnet50","efficientnet_b0"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="eval_outputs")
    args=ap.parse_args()

    ds, dl = get_loader(args.data, args.split, args.img_size, args.batch, args.workers)
    names=ds.classes; n=len(names)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=get_model(args.model, n, args.ckpt or None)

    t=time.time(); probs,preds,targets=evaluate(model,dl,device); dt=time.time()-t
    from pathlib import Path; out=Path(args.outdir)/f"{args.model}_{args.split}"; out.mkdir(parents=True,exist_ok=True)

    acc=accuracy_score(targets,preds); cm=confusion_matrix(targets,preds)
    plot_confusion(cm,names,out/"confusion_matrix.png"); save_report(targets,preds,names,out/"class_report.csv")
    np.save(out/"probs.npy",probs); np.save(out/"preds.npy",preds); np.save(out/"targets.npy",targets)
    with open(out/"summary.json","w",encoding="utf-8") as f:
        json.dump({"accuracy":acc,"elapsed_sec":dt,"num_samples":len(ds),"classes":names},f,ensure_ascii=False,indent=2)
    print(f"[{args.model}] {args.split} acc={acc:.4f} | n={len(ds)} | {dt:.1f}s -> {out.resolve()}")

if __name__=="__main__":
    main()
