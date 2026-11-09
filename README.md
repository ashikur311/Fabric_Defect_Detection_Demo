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
- [ ] Phase 2 – Research notes
- [ ] Phase 3 – Collect data
- [ ] Phase 4 – Explore & prepare (OpenCV)
- [ ] Phase 5 – Train
- [ ] Phase 6 – Evaluate
- [ ] Phase 7 – Demo app
- [ ] Phase 8 – Report
