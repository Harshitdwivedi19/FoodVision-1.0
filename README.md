# FoodVision 1.0 🥗🔍🤖

**FoodVision 1.0** is an end-to-end, production-ready computer vision and nutritional intelligence platform. It allows users on mobile phones, tablets, and desktop computers to point their camera (or upload photos) at food to instantly classify the dish, view detailed caloric and macronutrient breakdowns, calculate exercise burn equivalents, and receive in-depth clinical dietary diagnoses powered by a Local LLM.

---

## 🌟 Key Features

1. **Camera & Device Universal Support**:
   - **Mobile Camera**: Built-in camera flip button smoothly switches between high-res rear cameras (`environment`) pointing down at a plate and front selfie cameras (`user`).
   - **Desktop Webcam**: Interactive WebRTC viewfinder with reticle alignment.
   - **File Upload & Drag-and-Drop**: Supports JPEG, PNG, WEBP.
   - **Sample Foods Showcase**: Test dishes (Pizza, Sushi, Salad, Burger, Salmon, etc.) instantly without needing a camera.

2. **Deep Learning Vision Classifier**:
   - Classifies across **101 food classes** (Food-101 dataset).
   - High-confidence Top-1 detection and Top-K alternatives ranking.
   - Fast CPU and GPU inference pipeline with PyTorch and Vision Transformers.

3. **Nutritional & Calorie Intelligence**:
   - **Energy**: Calories (kcal) scaled dynamically to portion size (0.5x, 1.0x, 1.5x, 2.0x).
   - **Macronutrients**: Protein, Carbohydrates, Total Fats, Saturated Fats, Fiber, Sugar.
   - **Micronutrients**: Sodium, Potassium, Calcium, Iron, Vitamin C.
   - **Allergen Alerts**: Highlighting Gluten, Dairy, Tree Nuts, Peanuts, Shellfish, Soy, Eggs.
   - **Physical Activity Calculator**: Estimated minutes required to burn calories through Walking, Running, Cycling, or Swimming.

4. **Local LLM Health & Clinical Diagnosis**:
   - Connects to **Ollama** (`http://localhost:11434`) using models such as `llama3`, `mistral`, `phi3`, or `qwen2.5`.
   - **Zero-Downtime Fallback**: Includes a deterministic, evidence-based Clinical Nutritionist Expert System that guarantees accurate medical-grade diagnoses even when an LLM server is offline.
   - **Interactive Nutrition Chat**: Ask natural language questions ("Can I eat this for late dinner?", "How do I balance this meal?").

5. **Production Architecture & Tooling**:
   - **`uv` Package Management**: Fast dependency resolution and isolated execution.
   - **FastAPI Backend**: Asynchronous REST endpoints with high throughput.
   - **Dual Logging**: Structured console output and rotating disk logs in `logs/foodvision.log`.
   - **Kaggle Pipeline**: Dedicated downloader (`data/kaggle_downloader.py`) with automatic fallback and PyTorch training loop (`src/foodvision/models/train.py`).

---

## 🏗️ System Architecture

```
                                +-------------------+
                                |   Camera / User   |
                                +---------+---------+
                                          |
                                          v
                               +---------------------+
                               |   FastAPI Backend   |
                               +----+-----------+----+
                                    |           |
            +-----------------------+           +-----------------------+
            |                                                           |
            v                                                           v
+-----------------------+                                   +-----------------------+
| DL Vision Classifier  |                                   |  Nutrition Knowledge  |
|  (PyTorch / ViT / CNN)|                                   |       Database        |
+-----------+-----------+                                   +-----------+-----------+
            |                                                           |
            +-----------------------+           +-----------------------+
                                    |           |
                                    v           v
                               +---------------------+
                               | Local LLM Diagnosis |
                               | (Ollama / Clinical) |
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                               |  Interactive WebUI  |
                               | (Dashboard & Chat)  |
                               +---------------------+
```

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+
- `uv` installed (`pip install uv` or via standalone installer)

### 2. Install Dependencies
Dependencies are managed via `uv`. Run:
```bash
uv sync
```

### 3. Launch the Production Server
```bash
uv run python main.py
```
The server will boot at:
👉 **`http://localhost:8000`**

Open this URL in any desktop browser, or connect from a mobile phone on the same local network (`http://<your-ip>:8000`) to test the mobile camera with rear/front switching!

---

## 📷 Camera & Mobile Usage

- **Camera Flip**: On smartphones, tap the **Switch Camera** button in the top-right of the viewport to toggle between the **rear camera** (for scanning a plate of food) and the front camera.
- **Reticle Guide**: Center the plate inside the green viewport brackets for optimal classification accuracy.
- **Portion Multiplier**: Adjust portion size (0.5x, 1.0x, 1.5x, 2.0x) to automatically recompute calories, macros, and workout burn times in real time.

---

## 🧠 Local LLM Setup (Optional)

FoodVision 1.0 connects automatically to **Ollama** if it is running on your system:
```bash
# 1. Start Ollama
ollama serve

# 2. Pull a recommended model (e.g. llama3 or qwen2.5)
ollama pull llama3:latest
```
If Ollama is not installed or not running, FoodVision 1.0 seamlessly activates its **Local Clinical Nutrition Engine**, providing instant evidence-based diagnoses with zero configuration.

---

## 📦 Kaggle Dataset & Model Training

To download the full Food-101 dataset from Kaggle and train a custom PyTorch model:

### 1. Kaggle API Credentials
Place your `kaggle.json` key in `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).

### 2. Download Dataset
```bash
uv run python data/kaggle_downloader.py
```

### 3. Train the PyTorch Model
```bash
uv run python -m foodvision.models.train --epochs 10 --batch_size 32 --arch mobilenet_v3_small
```

### 4. Evaluate Model Metrics
```bash
uv run python -m foodvision.models.evaluate --model_path models/food_classifier.pt
```

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:
```bash
uv run pytest -v tests/
```

---

## 📝 Logging & Monitoring

All events, image classifications, latencies, and diagnostic requests are logged with rotation:
- Log file: `logs/foodvision.log`
- Configuration: Rotating handler (10MB max file size, 5 backups preserved)
