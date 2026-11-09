# Fabric Defect Detection – Denim (Prototype)

Basic image classification prototype that identifies fabric defects from a photo.

## Classes
| Class | Description |
|---|---|
| good | Clean, defect-free fabric |
| hole | Tear / opening in the fabric |
| stain | Oil, dirt or colour spots |
| missing_yarn | A thread is missing, leaving a line or gap |
| needle_mark | Small punctures from damaged needles |

## Tech stack
- Python 3.14
- PyTorch + torchvision (MobileNetV2 transfer learning, CPU)
- OpenCV (image loading & resizing)
- scikit-learn, Matplotlib, Seaborn (evaluation)
- Streamlit (demo app)

## Dataset
- Training: [FabriD – Fabric Defect Detection Dataset (Mendeley)](https://data.mendeley.com/datasets/vncdwps24r)
- Real-world test: own denim photos in `data/my_denim_samples/`

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project progress
- [x] Phase 1 – Setup
- [x] Phase 2 – Research notes → [docs/research_notes.md](docs/research_notes.md)
- [x] Phase 3 – Collect data (FabriD downloaded into `data/raw/`)
- [x] Phase 4 – Explore & prepare (OpenCV) → `python src/prepare_data.py`, `python src/explore_data.py`
- [x] Phase 5 – Train → `python src/train.py` (best val acc 85.4%, ~65 min on laptop CPU)
- [ ] Phase 6 – Evaluate → `python src/evaluate.py` (test acc **81.1%** ✅) · own denim photos: `python src/predict.py data\my_denim_samples` (pending)
- [x] Phase 7 – Demo app → `streamlit run app.py` (opens http://localhost:8501)
- [ ] Phase 8 – Report
