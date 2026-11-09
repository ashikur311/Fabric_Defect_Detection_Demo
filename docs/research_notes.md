# Research Notes – Computer Vision for Fabric Defect Detection (Denim)

*Phase 2 of the Fabric Defect Detection prototype.*

---

## 1. What is denim?

**Denim** is a strong cotton fabric used mainly for jeans, jackets and work wear.

- **Weave:** *Twill weave* – the threads cross in a way that creates visible **diagonal lines** on the surface.
- **Two threads:**
  - **Warp** (lengthwise threads) – dyed **blue** with indigo.
  - **Weft** (crosswise threads) – usually left **white**.
  - That's why denim is blue on the front and lighter on the back.
- **Weight:** measured in ounces per square yard (oz). Light denim ≈ 8–10 oz, standard jeans ≈ 12–14 oz.

### Denim production (simplified)

```
Cotton → Spinning (yarn) → Indigo dyeing (warp) → Weaving (loom) → Finishing → INSPECTION → Cutting & sewing → Washing → Jeans
```

Our project sits at the **INSPECTION** step – checking the woven fabric roll **before** it is cut. A defect missed here becomes a defective garment later, which is far more expensive.

---

## 2. What is a fabric defect?

A **fabric defect** is any flaw on the fabric surface that reduces its quality or value. Industry estimates say defects can reduce the price of fabric by **45–65%**, so finding them early matters.

### Defect types in this project

| Class | What it is | Typical cause | How it looks in an image |
|---|---|---|---|
| **good** | Defect-free fabric | – | Regular, repeating texture |
| **hole** | Opening/tear in the fabric | Broken needles, sharp machine parts, snagging | Dark or bright spot that breaks the texture |
| **stain** | Oil, dirt, rust or colour spot | Machine oil, dirty handling, dye problems | Patch with different colour/brightness, texture still intact |
| **missing_yarn** | One or more threads missing | Yarn breaks during weaving and loom continues | Thin straight line (horizontal or vertical) across the fabric |
| **needle_mark** | Small punctures or marks | Damaged/blunt needles in knitting or sewing | Small dots, often in a line |

### Other common denim defects (not in our dataset – future work)
- **Slub** – thick uneven spot in the yarn
- **Shade variation** – uneven indigo colour across the roll
- **Broken end / broken pick** – a single warp/weft thread broken
- **Float** – thread skips over others where it shouldn't
- **Crease marks** – permanent fold lines

---

## 3. How fabric is inspected today

### Manual inspection
- Fabric rolls run over an **inspection table** with back-lighting.
- A human inspector watches and marks defects.
- Common grading standard: the **4-Point System** – each defect gets 1–4 penalty points depending on its size; a roll with too many points per 100 square yards is rejected.

### Problems with manual inspection
| Problem | Detail |
|---|---|
| Slow | ~10–20 metres/minute |
| Tiring | Accuracy drops after 20–30 minutes of watching |
| Inconsistent | Different inspectors judge differently |
| Misses defects | Humans typically catch only ~60–75% of defects |

➡️ This is why factories are moving to **automatic (computer vision) inspection**.

---

## 4. What is computer vision?

**Computer vision (CV)** = teaching a computer to understand images.

A digital image is just a grid of numbers (pixels). For a colour image each pixel has 3 numbers (Red, Green, Blue, from 0 to 255). A 224×224 photo = 224 × 224 × 3 = **150,528 numbers**. CV finds patterns in these numbers.

### Three types of CV tasks

| Task | Question it answers | Output | Our project? |
|---|---|---|---|
| **Classification** | *What* defect is in this image? | One label: `hole` | ✅ **Yes** |
| **Object detection** | What defect and *where*? | Label + box around it | Future work |
| **Segmentation** | Exactly *which pixels* are defective? | Coloured mask | Future work |

We start with **classification** because it is the simplest and needs the least labelling.

---

## 5. How the model learns (simple explanation)

### Convolutional Neural Network (CNN)
A CNN is the standard model for images. It works in layers:

```
Image → [Edges & lines] → [Textures & shapes] → [Defect patterns] → Decision: "hole 94%"
         early layers       middle layers         deep layers
```

- **Early layers** detect simple things: edges, lines, dots.
- **Middle layers** combine them into textures (like the diagonal denim weave).
- **Deep layers** recognise complete patterns (a hole breaking the weave).

### Training
1. Show the model an image + its correct label.
2. Model makes a guess.
3. Measure how wrong it was (**loss**).
4. Adjust the model slightly to be less wrong.
5. Repeat thousands of times. One full pass through all images = **1 epoch**.

### Transfer learning – why we don't need millions of images
Training a CNN from zero needs millions of images. Instead we use **MobileNetV2**, a model already trained on **ImageNet** (1.4 million everyday images). It already knows edges, textures and shapes. We only:
1. Keep its "seeing" layers (frozen or lightly fine-tuned).
2. Replace the last layer with a new one that outputs our **5 fabric classes**.
3. Train on ~2,400 fabric images.

**Why MobileNetV2?** Small (~3.5 million parameters, ~14 MB), fast, runs on a laptop CPU, and accurate enough for a prototype.

---

## 6. Dataset choice

| Dataset | Images | Classes | Notes |
|---|---|---|---|
| **FabriD** (chosen) | ~2,407 | needle mark, hole, stain, missing yarn, defect-free | Ready for classification, light size |
| Own denim photos | 20–50 | same 5 | Real-world test only |

**Limitation:** FabriD is general woven fabric, not denim-specific. Defect *shapes* (holes, lines, spots) look similar across woven fabrics, so the model should transfer reasonably – our own denim photos will measure how well.

### Data split
| Split | Share | Purpose |
|---|---|---|
| Train | 70% | Model learns from these |
| Validation | 15% | Check progress during training, tune settings |
| Test | 15% | Final score – model never sees these during training |

### Preprocessing (OpenCV)
- Load images with `cv2.imread`
- Convert BGR → RGB (OpenCV loads colours in reverse order)
- Resize to **224×224** (MobileNetV2 input size)
- Normalise pixel values the way ImageNet models expect

### Data augmentation
To make the model robust with a small dataset, training images get small random changes: flips, small rotations, brightness changes. Defects look the same flipped, so the label stays correct.

---

## 7. How we measure success

| Metric | Meaning |
|---|---|
| **Accuracy** | % of test images labelled correctly |
| **Precision** | When the model says "hole", how often is it really a hole? |
| **Recall** | Of all real holes, how many did the model find? (most important in factories – a missed defect is costly) |
| **Confusion matrix** | Table showing which classes get mixed up (e.g. stain ↔ good) |

**Target:** 85–95% accuracy on FabriD test set. Lower accuracy is expected on own denim photos.

---

## 8. Expected challenges

- **Denim texture is busy** – the diagonal twill pattern can hide small defects.
- **Small defects** – needle marks and missing yarn may be tiny compared to the whole image.
- **Lighting** – phone photos differ a lot from dataset photos.
- **Similar classes** – faint stains may look like good fabric.
- **Domain gap** – training on general fabric, testing on denim.

---

## 9. Future improvements (beyond this prototype)

1. Build a **denim-specific dataset** from a real factory.
2. Move to **object detection (YOLO)** to show *where* the defect is.
3. Real-time inspection with a camera above the fabric roll.
4. Add more denim defect types (slub, shade variation, float).

---

## References
- FabriD: Fabric Defect Detection Dataset – https://data.mendeley.com/datasets/vncdwps24r
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018
- PyTorch transfer learning tutorial – https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- ASTM D5430 – Standard Test Methods for Visually Inspecting and Grading Fabrics (4-Point System)
