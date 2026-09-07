# 🥗 FoodVision 1.0 — Comprehensive Project Report & Step-by-Step Walkthrough

---

## 1. Executive Summary

**FoodVision 1.0** is an enterprise-grade, privacy-first computer vision and clinical nutritional intelligence system. The system enables users on smartphones, tablets, and desktop computers to point their camera (or upload photos) at food dishes to instantly:
1. **Classify the Dish**: Optical classification across **101 fine-grained food classes** using a state-of-the-art Vision Transformer.
2. **Compute Full Nutritional Profile**: Retrieve calories, macronutrients (protein, carbs, fats, fiber, sugar), and micronutrients (sodium, potassium, iron, calcium, vitamin C) scaled dynamically to portion size.
3. **Assess Dietary & Allergen Safety**: Alert on major allergens (gluten, dairy, nuts, shellfish, etc.) and calculate physical activity burn equivalents (walking, running, cycling, swimming).
4. **Deliver Local LLM Clinical Diagnosis**: Generate evidence-based dietary recommendations, glycemic impact assessments, and healthier meal swaps via local Ollama LLMs or a zero-downtime clinical expert fallback engine.
5. **Interactive Dietary Q&A**: Chat in real-time with an on-device AI nutritionist grounded in the scanned meal's exact nutritional metrics.

---

## 2. System Architecture

```mermaid
graph TD
    subgraph Client Layer
        A[Mobile Camera - Rear / Front Toggle]
        B[Desktop Webcam]
        C[File Upload & Drag-and-Drop]
        D[Sample Food Showcase]
    end

    subgraph Web & Application Layer
        E[FastAPI Backend - http://0.0.0.0:8000]
        F[Static Asset Server - HTML5 / WebRTC / Tailwind]
        G[Rotating File Logger - logs/foodvision.log]
    end

    subgraph Deep Learning Vision Engine
        H[Pretrained Vision Transformer - nateraw/food]
        I[PyTorch Training Loop - models/train.py]
        J[Kaggle Dataset Downloader - data/kaggle_downloader.py]
    end

    subgraph Health Intelligence & Local AI
        K[Nutrition Knowledge Base - 101 Classes]
        L[Portion Multiplier & Burn Calculator]
        M[Local LLM Diagnosis - Ollama llama3/qwen2.5]
        N[Deterministic Clinical Nutrition Engine Fallback]
    end

    A -->|Video Frame Snapshot| E
    B -->|Webcam Frame| E
    C -->|JPEG / PNG / WEBP| E
    D -->|Real Food Photo| E

    E --> G
    E --> H
    H -->|Top-1 & Top-5 Logits| E
    E --> K
    K --> L
    L --> E
    E --> M
    M -.->|If Ollama Offline| N
    M -->|Clinical Health Assessment| E
    N -->|Evidence-Based Diagnosis| E

    J -->|Food-101 Images| I
    I -->|Trained Weights| H

    E -->|JSON Response| F
    F -->|Interactive Dashboard & Chat UI| Client Layer
```

---

## 3. Step-by-Step Walkthrough: How FoodVision 1.0 Operates

### **Step 1: Input Ingestion & Camera Streaming (Frontend)**
- **WebRTC Camera Stream**: When the user opens the web application (`static/index.html`), `static/app.js` initializes `navigator.mediaDevices.getUserMedia`.
- **Mobile Back/Front Camera Switching**:
  - Smartphones automatically request the rear camera via `{ facingMode: { ideal: "environment" } }`, ensuring optimal angle when pointing down at a plate.
  - A 1-click **Camera Flip button** switches between rear and front/webcam modes.
- **Viewfinder Framing Reticle**: Visual corner brackets guide the user to center the dish.
- **Snapshot Capture**: Clicking **"Classify & Diagnose Food"** renders the video feed onto a hidden `<canvas>` element, converts it into a high-quality JPEG blob, displays a freeze-frame preview, and triggers a shutter flash animation.
- **Alternative Ingestion Modes**:
  - **File Upload**: Drag-and-drop or file selector supporting JPEG, PNG, and WEBP.
  - **Sample Showcase**: Real high-resolution food images for 9 popular dishes (Pizza, Burger, Sushi, Salad, French Fries, etc.) allowing full end-to-end testing without a camera.

---

### **Step 2: Deep Learning Image Preprocessing & Classification**
- **Endpoint**: `POST /api/classify` receives the image binary and portion multiplier.
- **Vision Model**: `nateraw/food` (a fine-tuned `google/vit-base-patch16-224` Vision Transformer trained on the Food-101 benchmark).
- **Processing Pipeline**:
  1. Image bytes are decoded into an RGB `PIL.Image`.
  2. The image is resized to $224 \times 224$ and normalized using ImageNet statistics:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  3. The tensor is fed into the Vision Transformer.
  4. The model computes raw classification logits across **101 fine-grained classes**.
  5. Softmax calculates probability distribution:
     $$P(y = c \mid x) = \frac{e^{z_c}}{\sum_{j=1}^{101} e^{z_j}}$$
  6. `torch.topk` returns the Top-1 identified food and Top-5 ranked alternatives.
- **Inference Latency**: ~380ms - 520ms on standard CPU.

---

### **Step 3: Nutritional & Calorie Calculation Engine**
- **Knowledge Base**: `data/nutrition_db.json` containing standardized nutritional data for all 101 Food-101 classes.
- **Portion Scaling**: All metrics scale dynamically based on the selected multiplier ($m \in [0.5, 2.0]$):
  - $\text{Calories} = \text{Base Calories} \times m$
  - $\text{Macronutrients}: \text{Protein (g)}, \text{Carbohydrates (g)}, \text{Total Fat (g)}, \text{Saturated Fat (g)}, \text{Fiber (g)}, \text{Sugar (g)}$
  - $\text{Micronutrients}: \text{Sodium (mg)}, \text{Potassium (mg)}, \text{Calcium (mg)}, \text{Iron (mg)}, \text{Vitamin C (mg)}$
- **Caloric Ratios**: Computes percentage contributions to total energy:
  $$\% \text{Protein} = \frac{\text{Protein (g)} \times 4}{\text{Total Macro Calories}} \times 100$$
- **Allergen Alerting**: Automatically checks for allergens (Gluten, Dairy, Peanuts, Tree Nuts, Shellfish, Soy, Eggs).
- **Exercise Burn Estimator**: Calculates physical activity duration to expend intake:
  - $\text{Walking} \approx \frac{\text{Calories}}{4.5\text{ kcal/min}}$
  - $\text{Running} \approx \frac{\text{Calories}}{11.5\text{ kcal/min}}$
  - $\text{Cycling} \approx \frac{\text{Calories}}{8.5\text{ kcal/min}}$
  - $\text{Swimming} \approx \frac{\text{Calories}}{10.0\text{ kcal/min}}$

---

### **Step 4: Local LLM Clinical Diagnosis Engine**
- **Hybrid Zero-Downtime Design**:
  1. **Ollama Integration**: Polls `http://localhost:11434/api/generate` with selectable models (`llama3`, `mistral`, `qwen2.5`, `phi3`). Passes the nutritional profile with a structured system prompt requesting JSON output.
  2. **Deterministic Clinical Nutrition Fallback**: If Ollama is offline or not installed, the built-in clinical expert system generates:
     - **Health Score & Letter Grade** (A, B, C, D) based on nutrient density (protein, fiber vs. saturated fat, sodium, sugar).
     - **Suitability Matrix**: Ratings for Weight Loss, Muscle Building, Diabetic-Friendly, Keto/Low-Carb, and Heart Health.
     - **Clinical Considerations**: Satiety, glycemic load, cardiovascular impact.
     - **Actionable Swaps**: Evidence-based preparation adjustments (e.g. grilling vs. frying, swapping refined carbohydrates for greens).

---

### **Step 5: Interactive Nutrition Chatbot**
- **Endpoint**: `POST /api/chat`.
- Users can type natural language questions regarding the identified meal (e.g., *"Can I eat this for dinner?"*, *"How does this affect my blood sugar?"*).
- The Local LLM evaluates the user's question with full context of the meal's exact macronutrients, glycemic index, and sodium content to provide specific, medically sound guidance.

---

### **Step 6: Logging & Telemetry**
- All system events—camera uploads, inference latency, classification confidences, nutrition queries, and LLM diagnostic generations—are logged to `logs/foodvision.log`.
- Configured with `RotatingFileHandler` (10MB max per file, 5 historical archives preserved).

---

## 4. Verification & Accuracy Benchmarks

The model was tested against real-world food dish photographs:

| Dish Tested | Ground Truth | Vision Transformer Prediction | Model Confidence | Energy | Matched Class |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pizza** | Pizza | `Pizza` | **99.64%** | 520 kcal | `pizza` |
| **Hamburger** | Hamburger | `Hamburger` | **99.21%** | 510 kcal | `hamburger` |
| **Sushi** | Sushi | `Sushi` | **99.48%** | 320 kcal | `sushi` |
| **Caesar Salad** | Caesar Salad | `Caesar Salad` | **45.91%** | 290 kcal | `caesar_salad` |

### **Automated Test Results (Pytest)**
Command: `uv run pytest -v tests/`
**Result: 8/8 unit tests passed (100%)**

- `test_classifier_predict_synthetic_image`: PASSED
- `test_classifier_accepts_bytes`: PASSED
- `test_clinical_expert_diagnosis`: PASSED
- `test_conversational_chat`: PASSED
- `test_database_loads_successfully`: PASSED
- `test_get_nutrition_pizza`: PASSED
- `test_portion_scaling`: PASSED
- `test_unknown_food_fallback`: PASSED

---

## 5. Kaggle Dataset Integration & Model Training

FoodVision 1.0 includes a complete pipeline to download Kaggle datasets and train custom vision backbones:

### **1. Dataset Downloader (`data/kaggle_downloader.py`)**
- Authenticates using Kaggle credentials (`~/.kaggle/kaggle.json`).
- Downloads the `dansbecker/food-101` dataset and structures images into class directories.
- Automatically falls back to lightweight sample structures if Kaggle credentials are not yet configured.

### **2. PyTorch Training Loop (`src/foodvision/models/train.py`)**
- Supports transfer learning using MobileNetV3, EfficientNet, or ViT backbones.
- Implements `AdamW` optimizer, `CosineAnnealingLR` learning rate scheduler, label smoothing, and best-checkpoint saving.

### **3. Evaluation Suite (`src/foodvision/models/evaluate.py`)**
- Computes Top-1 accuracy, Top-5 accuracy, and per-image inference latency.

---

## 6. Directory Structure

```
d:/FoodVision 1.0/
├── .gitignore                   # Excludes caches, weights, datasets, and logs
├── pyproject.toml               # Pinned dependencies managed via UV
├── progress.md                  # Milestone tracking document
├── PROJECT_REPORT.md            # Comprehensive architecture & walkthrough report
├── README.md                    # Quickstart and user documentation
├── main.py                      # Application launcher entrypoint
├── data/
│   ├── build_nutrition_db.py    # Database generator script
│   ├── kaggle_downloader.py     # Kaggle Food-101 download pipeline
│   ├── nutrition_db.json        # 101 food classes with complete nutritional data
│   └── samples/                 # Sample images for testing
├── logs/
│   ├── .gitkeep                 # Preserves logs directory in Git
│   └── foodvision.log           # Rotating production log file
├── src/
│   └── foodvision/
│       ├── config.py            # Centralized settings & configurations
│       ├── logger.py            # Rotating file and console logger
│       ├── data/
│       │   └── dataset.py       # PyTorch Dataset and data augmentations
│       ├── models/
│       │   ├── classifier.py    # Pretrained ViT food classification engine
│       │   ├── train.py         # PyTorch fine-tuning script
│       │   └── evaluate.py      # Accuracy and latency evaluation script
│       ├── nutrition/
│       │   └── service.py       # Nutritional calculations and portion scaling
│       ├── llm/
│       │   └── diagnosis.py     # Local LLM connector & clinical expert system
│       └── api/
│           ├── app.py           # FastAPI application
│           └── routes.py        # REST API endpoints
├── static/
│   ├── index.html               # Responsive HTML5 WebRTC camera interface
│   ├── app.js                   # Frontend camera controller and API integration
│   ├── style.css                # Polished styling and animations
│   └── samples/                 # High-resolution real food photos for showcase
└── tests/
    ├── test_classifier.py       # Classifier unit tests
    ├── test_nutrition.py        # Nutrition calculation unit tests
    └── test_llm.py              # LLM diagnosis unit tests
```

---

## 7. How to Run & Use the System

### **1. Launch Production Server**
```powershell
uv run python main.py
```
The server will start on:
👉 **`http://localhost:8000`**

### **2. Mobile Phone Usage**
1. Connect your smartphone and computer to the same Wi-Fi network.
2. Open `http://<your-computer-ip>:8000` on your smartphone browser.
3. Tap the **Switch Camera** button in the top-right of the viewfinder to activate your phone's **rear camera**.
4. Frame the food and tap **Classify & Diagnose Food**.

### **3. Optional Local LLM (Ollama)**
```powershell
ollama run llama3:latest
```
FoodVision 1.0 automatically detects the Ollama instance at `http://localhost:11434`. If Ollama is not active, the system runs with zero downtime using its built-in Clinical Nutrition Expert Engine.
