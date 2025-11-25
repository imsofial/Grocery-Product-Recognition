# app.py
from pathlib import Path
from typing import Dict, Any
from tempfile import NamedTemporaryFile
import json

import streamlit as st

from pipeline import detect_and_classify_image
from recipe_api import get_recipes_by_ingredient_raw
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Basic page setup
# -----------------------------
st.set_page_config(page_title="Grocery Recognition", layout="centered")
st.title("Grocery Product Recognition")
st.caption(
    "Upload a photo → detect multiple groceries → classify fruit + freshness → "
    "optionally fetch recipes for all detected fruits."
)


def parse_label(label: str):
    """
    'apple/fresh' -> ('apple', 'fresh')
    """
    if "/" in label:
        fruit, state = label.split("/", 1)
    elif "\\" in label:
        fruit, state = label.split("\\", 1)
    else:
        fruit, state = label, "unknown"
    return fruit, state


def draw_boxes(image: Image.Image, detections):
    """
    Draw rectangles and labels on a copy of the image.
    detections: list of dicts with "box" and "label".
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for det in detections:
        box = det.get("box")
        label = det.get("label", "")
        score = det.get("score", 0.0)

        if not box:
            continue

        x0, y0, x1, y1 = box
        # rectangle
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)

        # label text
        text = f"{label} ({score*100:.1f}%)"

        # Get text size using textbbox (works in new Pillow)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for very old Pillow versions
            text_w, text_h = font.getsize(text)

        # small background rect for text
        # put it slightly above the box (if there's space), otherwise inside
        text_x0 = x0
        text_y0 = max(0, y0 - text_h - 4)
        text_x1 = text_x0 + text_w + 4
        text_y1 = text_y0 + text_h + 4

        draw.rectangle((text_x0, text_y0, text_x1, text_y1), fill="red")
        draw.text((text_x0 + 2, text_y0 + 2), text, fill="white", font=font)

    return img


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    # if not ULTRALYTICS_AVAILABLE:
    #     st.error(
    #         "Ultralytics YOLO is not installed.\n\n"
    #         "Install with: `pip install ultralytics` in your .venv and restart the app."
    #     )

    # Predefined TheMealDB categories
    CATEGORY_OPTIONS = [
        "Any",        # new: no category filtering
        "Dessert",
        "Breakfast",
        "Beef",
        "Chicken",
        "Goat",
        "Lamb",
        "Miscellaneous",
        "Pasta",
        "Pork",
        "Seafood",
        "Side",
        "Starter",
        "Vegan",
        "Vegetarian",
    ]

    category_filter = st.selectbox(
        "Recipe category (TheMealDB)",
        options=CATEGORY_OPTIONS,
        index=0,  # default = "Any"
        help='Choose "Any" for all categories, or a specific one like "Dessert", "Seafood", etc.',
    )

    st.markdown("---")
    st.caption(
        "Models used in the pipeline:\n"
        "- YOLOv8n for detection\n"
        "- ResNet50 (10 classes: fruit × freshness) for classification"
    )


# -----------------------------
# Main UI
# -----------------------------
st.subheader("Upload an image")
file = st.file_uploader(
    "Upload a photo of your fridge / bag (JPG/PNG/WEBP/BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if file is None:
    st.info("Upload an image to run detection and (optionally) recipes.")
else:
    # Load and show original image
    img = Image.open(file).convert("RGB")
    st.image(img, caption=file.name, use_container_width=True)

    if not ULTRALYTICS_AVAILABLE:
        st.warning("Detection is disabled because ultralytics is not installed.")
    else:
        st.divider()
        st.markdown("### Detection & Classification")

        with st.spinner("Running YOLO detection + ResNet classification..."):
            result: Dict[str, Any] = detect_and_classify_image(img, topk=1)

        detections = result.get("detections", [])
        counts = result.get("counts", {})
        ingredients = result.get("ingredients", [])

        if len(detections) == 0:
            st.warning("No objects detected on the image.")
        else:
            # 1) Show annotated image with boxes + labels
            annotated = draw_boxes(img, detections)
            st.image(
                annotated,
                caption="Detected objects with predicted fruit/freshness",
                use_container_width=True,
            )

            # 2) Show summary counts
            st.markdown("#### Detected items (summary)")
            for label, c in counts.items():
                fruit, state = parse_label(label)
                st.write(f"- **{fruit}** ({state}) × **{c}**")

            # 3) Per-object details
            st.markdown("#### Per-object predictions")
            for i, det in enumerate(detections, start=1):
                label = det.get("label", "")
                score = det.get("score", 0.0)
                fruit, state = parse_label(label)
                with st.container(border=True):
                    st.write(f"**Object #{i}**")
                    st.write(f"- Label: `{label}`")
                    st.write(f"- Fruit: `{fruit}`")
                    st.write(f"- Freshness: `{state}`")
                    st.write(f"- Confidence (ResNet): {score * 100:.2f}%")
                    st.write(f"- Detection conf (YOLO): {det.get('det_conf', 0.0) * 100:.2f}%")

            st.divider()
            st.markdown("### Recipes for detected fruits")

            if not ingredients:
                st.info("No fruits recognized, so no recipes to fetch.")
            else:
                st.write(
                    "Detected fruits (ignoring freshness): "
                    + ", ".join(f"`{ing}`" for ing in ingredients)
                )

                # 🔘 Only fetch recipes on button press
                if st.button("Get recipes for all detected fruits"):
                    combined_recipes = []

                    with st.spinner("Fetching recipes from TheMealDB..."):
                        for ingr in ingredients:
                            st.markdown(f"#### Recipes for **{ingr}**")
                            recs = get_recipes_by_ingredient_raw(
                                ingr, category_filter=category_filter
                            )

                            if isinstance(recs, str):
                                # Error / not found message
                                st.info(recs)
                                continue

                            if not recs:
                                st.info(f"No recipes found for '{ingr}'.")
                                continue

                            for rec in recs:
                                combined_recipes.append(rec)
                                with st.container(border=True):
                                    st.subheader(rec.get("name", ""))

                                    if rec.get("image"):
                                        st.image(rec["image"], use_container_width=True)

                                    st.write(f"**Category:** {rec.get('category', '')}")
                                    st.write(f"**Area:** {rec.get('area', '')}")

                                    ings = rec.get("ingredients", [])
                                    if ings:
                                        st.markdown("**Ingredients:**")
                                        for line in ings:
                                            st.write(f"- {line}")

                                    instr = rec.get("instructions", "")
                                    if instr:
                                        with st.expander("Instructions"):
                                            st.write(instr)

                    if combined_recipes:
                        st.download_button(
                            label="Download all recipes as JSON",
                            data=json.dumps(combined_recipes, indent=2, ensure_ascii=False),
                            file_name="all_recipes.json",
                            mime="application/json",
                        )
