# app.py
from pathlib import Path
from typing import List, Tuple
import json

import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T

# <-- NEW: import your recipe module
import recipe_api  # expects recipe_api.py in the same folder

# -----------------------------
# Basic page setup
# -----------------------------
st.set_page_config(page_title="Grocery Recognition", layout="centered")
st.title("Grocery Product Recognition")
st.caption("Upload a photo → predict fruit + freshness → fetch and display recipes for that fruit.")

# -----------------------------
# Constants
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Joint 10-class labels (MUST match your training order)
JOINT_CLASSES = [
    "apple/fresh", "apple/rotten",
    "banana/fresh", "banana/rotten",
    "orange/fresh", "orange/rotten",
    "potato/fresh", "potato/rotten",
    "tomato/fresh", "tomato/rotten",
]

DEFAULT_CKPT = "finetuned_models/resnet50_fruits.pth"

# -----------------------------
# Helpers
# -----------------------------
def load_image(file) -> Image.Image:
    return Image.open(file).convert("RGB")

def preprocess(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(img)

def predict_top1(model: torch.nn.Module, device: str, x: torch.Tensor, class_names: List[str]) -> Tuple[str, float]:
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx])

def parse_fruit_and_freshness(joint_label: str) -> Tuple[str, str]:
    # supports "/" or "\" separators
    if "/" in joint_label:
        fruit, state = joint_label.split("/", 1)
    elif "\\" in joint_label:
        fruit, state = joint_label.split("\\", 1)
    else:
        fruit, state = joint_label, "unknown"
    return fruit, state

def load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        sd = state.get("state_dict", state.get("model_state", state))
    else:
        sd = state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("Checkpoint (.pth)", value=DEFAULT_CKPT)
    img_size  = st.slider("Image size (CenterCrop)", 128, 512, 224, step=16)

# -----------------------------
# Model (lazy-load, robust)
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_model(num_classes: int, ckpt_path: str):
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)

    p = Path(ckpt_path)
    if not p.exists():
        return None, f"Checkpoint not found: {ckpt_path}", None

    missing, unexpected = load_checkpoint_flex(m, ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m.eval().to(device)
    info = f"Model ready. missing={len(missing)}, unexpected={len(unexpected)} (strict=False)."
    return m, None, info

model, model_err, model_info = build_model(len(JOINT_CLASSES), ckpt_path)

st.subheader("Upload an image")
file = st.file_uploader("JPG/PNG/WEBP/BMP", type=["jpg", "jpeg", "png", "webp", "bmp"])

if model_err:
    st.warning(f"⚠️ {model_err}")
elif model_info:
    st.caption(model_info)

# -----------------------------
# Main prediction
# -----------------------------
if file is None:
    st.info("Upload an image to get a prediction.")
else:
    img = load_image(file)
    st.image(img, caption=file.name, use_column_width=True)

    if model is None:
        st.error("Model not loaded. Fix the checkpoint path in the sidebar.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = preprocess(img, img_size=img_size).unsqueeze(0)
        top1_joint, top1_prob = predict_top1(model, device, x, JOINT_CLASSES)
        fruit, state = parse_fruit_and_freshness(top1_joint)

        st.markdown("### Result")
        st.write(f"**Fruit:** `{fruit}`")
        st.write(f"**Freshness:** `{state}`")
        st.write(f"**Confidence:** {top1_prob*100:.2f}%")
        st.caption(f"(Joint class: `{top1_joint}`)")

        st.divider()
        st.markdown("### Recipes")
        st.caption("Click to fetch up to 5 recipes for the detected fruit (Dessert category by default).")

        # Let user change the category filter if desired
        category = st.text_input("Category filter", value="Dessert")

        if st.button(f"Get recipes for **{fruit}**"):
            with st.spinner("Requesting recipes..."):
                # Call your recipe API function (this writes recipes.json)
                msg = recipe_api.get_recipes_by_ingredient(fruit, category_filter=category)

            st.success(msg)

            # Try to read recipes.json and display nicely
            json_path = Path("recipes.json")
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(data, list) and len(data) > 0:
                        st.markdown("#### Results")
                        for rec in data:
                            with st.container(border=True):
                                # Title + image
                                st.subheader(rec.get("name", ""))
                                if rec.get("image"):
                                    st.image(rec["image"], use_column_width=True)
                                # Meta
                                st.write(f"**Category:** {rec.get('category','')}")
                                st.write(f"**Area:** {rec.get('area','')}")
                                # Ingredients
                                ings = rec.get("ingredients", [])
                                if ings:
                                    st.markdown("**Ingredients:**")
                                    for line in ings:
                                        st.write(f"- {line}")
                                # Instructions (collapsible)
                                instr = rec.get("instructions", "")
                                if instr:
                                    with st.expander("Instructions"):
                                        st.write(instr)
                        # Offer download of the JSON the function produced
                        st.download_button(
                            label="Download recipes.json",
                            data=json.dumps(data, indent=2, ensure_ascii=False),
                            file_name="recipes.json",
                            mime="application/json"
                        )
                    else:
                        st.info("No recipes found in recipes.json.")
                except Exception as e:
                    st.warning(f"Could not parse recipes.json: {e}")
            else:
                st.info("recipes.json not found after the call.")
