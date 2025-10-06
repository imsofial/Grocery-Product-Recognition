# eval_freshness.py
import argparse, json, time, os
from pathlib import Path
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, itertools, csv

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
CLASSES=["fresh","rotten"]; LABELS={"fresh":0,"rotten":1}
IMG_EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

class FreshnessDataset(Dataset):
    # expects: split/<fruit>/{fresh,rotten}/image.jpg
    def __init__(self, root, split="test", img_size=224):
        self.root=Path(root)/split
        self.items=[]
        for fruit_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for status in ["fresh","rotten"]:
                sdir=fruit_dir/status
                if sdir.is_dir():
                    for img in sdir.rglob("*"):
                        if img.suffix.lower() in IMG_EXTS:
                            self.items.append((img, LABELS[status]))
        self.tfm=T.Compose([T.Resize(256),T.CenterCrop(img_size),T.ToTensor(),
                            T.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = Image.open(p).convert("RGB")
        return self.tfm(im), y

def get_model(name, num_classes=2, ckpt_path=None):
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

def plot_confusion(cm, names, out):
    plt.figure(figsize=(5,4)); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    ticks=np.arange(len(names)); plt.xticks(ticks,names,rotation=0); plt.yticks(ticks,names)
    thr=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v=cm[i,j]; 
        if v>0: plt.text(j,i,str(v),ha="center",color="white" if v>thr else "black",fontsize=10)
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

    ds=FreshnessDataset(args.data, split=args.split, img_size=args.img_size)
    dl=DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=get_model(args.model, num_classes=2, ckpt_path=args.ckpt or None)

    t=time.time(); probs,preds,targets=evaluate(model,dl,device); dt=time.time()-t
    out=Path(args.outdir)/f"{args.model}_freshness_{args.split}"; out.mkdir(parents=True,exist_ok=True)

    names=CLASSES; acc=accuracy_score(targets,preds); cm=confusion_matrix(targets,preds)
    plot_confusion(cm,names,out/"confusion_matrix.png"); save_report(targets,preds,names,out/"class_report.csv")
    np.save(out/"probs.npy",probs); np.save(out/"preds.npy",preds); np.save(out/"targets.npy",targets)
    with open(out/"summary.json","w",encoding="utf-8") as f:
        json.dump({"accuracy":acc,"elapsed_sec":dt,"num_samples":len(ds),"classes":names},f,ensure_ascii=False,indent=2)
    print(f"[{args.model} freshness] {args.split} acc={acc:.4f} | n={len(ds)} | {dt:.1f}s -> {out.resolve()}")

if __name__=="__main__":
    main()
