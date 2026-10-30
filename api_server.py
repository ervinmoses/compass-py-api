import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import warnings
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import re

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings("ignore")

app = FastAPI()

FEATURES = ['BM', 'BI', 'Math', 'Sejarah', 'Avg_STEM', 'Avg_Art_Biz', 'Elective_Count', 'Avg_Elective']
TARGET = 'Status_Code'
TARGET_MAP = {0: 'Dropped Out / Failed', 1: 'Late Graduation', 2: 'Graduation On Time'}

class TrainingData(BaseModel):
    model_code: str
    test_size: float
    data: List[Dict[str, Any]]

def balance_with_smote(X_train, y_train, step_log):
    dist = Counter(y_train)
    step_log.append(f"Original class distribution: {dict(dist)}")

    if dist.get(0, 0) < 3:
        step_log.append("WARNING: Very few dropout samples — injecting 5 synthetic profiles to enable SMOTE.")
        synth = pd.DataFrame({
            'BM': [0, 0, 1, 0, 1], 'BI': [0, 1, 0, 0, 0],
            'Math': [0, 0, 0, 1, 0], 'Sejarah': [0, 0, 0, 0, 1],
            'Avg_STEM': [0.0, 0.5, 0.0, 1.0, 0.0], 'Avg_Art_Biz': [0.0, 0.0, 0.5, 0.0, 0.0],
            'Elective_Count': [0, 1, 0, 2, 0], 'Avg_Elective': [0.0, 0.5, 0.0, 1.0, 0.0]
        })
        synth_labels = pd.Series([0, 0, 0, 0, 0])
        X_train = pd.concat([X_train, synth], ignore_index=True)
        y_train = pd.concat([y_train, synth_labels], ignore_index=True)
        dist = Counter(y_train)

    if SMOTE_AVAILABLE:
        min_class_count = min(dist.values())
        k = min(5, min_class_count - 1)
        if k >= 1:
            try:
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_res, y_res = smote.fit_resample(X_train, y_train)
                step_log.append(f"SMOTE balanced distribution: {dict(Counter(y_res))} [k_neighbors={k}]")
                return X_res, y_res
            except Exception as e:
                step_log.append(f"SMOTE failed ({e}), using original distribution.")
        else:
            step_log.append("Not enough samples for SMOTE, skipping.")
    else:
        step_log.append("imblearn not installed — SMOTE skipped.")
    return X_train, y_train


def train_decision_tree(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        criterion='gini', max_depth=8, min_samples_split=10,
        min_samples_leaf=4, class_weight='balanced',
        ccp_alpha=0.005, random_state=42
    )
    model.fit(X_train, y_train)
    return model, None

def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=8,
        min_samples_leaf=3, max_features='sqrt', bootstrap=True,
        oob_score=True, class_weight='balanced_subsample',
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return model, None

def train_svm(X_train, y_train):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SVC(
        kernel='rbf', C=5.0, gamma='scale',
        probability=True, class_weight='balanced', random_state=42
    )
    model.fit(X_scaled, y_train)
    return model, scaler

def train_knn(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    k = int(np.sqrt(len(y_train)))
    if k % 2 == 0: k += 1
    k = max(3, min(15, k))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = KNeighborsClassifier(
        n_neighbors=k, weights='distance',
        metric='minkowski', p=2, algorithm='ball_tree'
    )
    model.fit(X_scaled, y_train)
    return model, scaler

def get_feature_weights(model, model_code, X_test, y_test, scaler=None, step_log=None):
    if step_log is None: step_log = []
    weights = []
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        weights = [{"feature": f, "weight": round(float(w), 4)} for f, w in zip(FEATURES, importances)]
    else:
        try:
            from sklearn.inspection import permutation_importance
            X_eval = scaler.transform(X_test) if scaler is not None else X_test
            result = permutation_importance(model, X_eval, y_test, n_repeats=30, random_state=42, n_jobs=-1)
            importances = result.importances_mean
            importances = np.clip(importances, 0, None)
            total = np.sum(importances)
            if total > 0: importances = importances / total
            weights = [{"feature": f, "weight": round(float(w), 4)} for f, w in zip(FEATURES, importances)]
        except Exception as e:
            w = round(1.0 / len(FEATURES), 4)
            weights = [{"feature": f, "weight": w} for f in FEATURES]
            
    return sorted(weights, key=lambda x: x['weight'], reverse=True)


@app.post("/train")
async def train_endpoint(payload: TrainingData):
    step_log = []
    try:
        # Load and clean data
        df = pd.DataFrame(payload.data)
        
        # Ensure numeric types
        for col in FEATURES + [TARGET]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=FEATURES + [TARGET], inplace=True)
        df[TARGET] = df[TARGET].astype(int)
        df = df[df[TARGET] >= 0].copy()
        
        X = df[FEATURES]
        y = df[TARGET]
        
        student_ids = df['student_id'] if 'student_id' in df.columns else pd.Series(range(len(df)))

        unique_classes = y.unique()
        can_stratify = all(sum(y == c) >= 2 for c in unique_classes) and len(unique_classes) > 1
        stratify_arg = y if can_stratify else None

        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, student_ids, test_size=payload.test_size, random_state=42, stratify=stratify_arg
        )

        X_train_bal, y_train_bal = balance_with_smote(X_train, y_train, step_log)

        # Select and train model
        if payload.model_code == 'dt':
            model, scaler = train_decision_tree(X_train_bal, y_train_bal)
        elif payload.model_code == 'rf':
            model, scaler = train_random_forest(X_train_bal, y_train_bal)
        elif payload.model_code == 'svm':
            model, scaler = train_svm(X_train_bal, y_train_bal)
        elif payload.model_code == 'knn':
            model, scaler = train_knn(X_train_bal, y_train_bal)
        else:
            raise HTTPException(status_code=400, detail="Invalid model code")

        # Evaluate accuracy
        X_test_eval = scaler.transform(X_test) if scaler is not None else X_test
        y_pred = model.predict(X_test_eval)
        acc = accuracy_score(y_test, y_pred) * 100

        # Save model
        saved_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')
        os.makedirs(saved_dir, exist_ok=True)
        joblib.dump(model, os.path.join(saved_dir, f'model_{payload.model_code}.pkl'))
        if scaler is not None:
            joblib.dump(scaler, os.path.join(saved_dir, f'scaler_{payload.model_code}.pkl'))

        # Prepare sample predictions
        sample_size = len(X_test)
        X_test_arr = X_test.values[:sample_size]
        y_test_arr = y_test.values[:sample_size]
        y_pred_arr = y_pred[:sample_size]
        id_test_arr = id_test.values[:sample_size]
        
        X_test_display = scaler.transform(X_test_arr) if scaler is not None else X_test_arr

        predictions_sample = []
        for i in range(sample_size):
            prob_val = 100.0
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test_display[i:i+1])[0]
                prob_val = round(float(max(probs)) * 100, 1)

            predictions_sample.append({
                "student_id": str(id_test_arr[i]),
                "index": int(i + 1),
                "actual": TARGET_MAP.get(int(y_test_arr[i]), "Unknown"),
                "predicted": TARGET_MAP.get(int(y_pred_arr[i]), "Unknown"),
                "correct": bool(y_test_arr[i] == y_pred_arr[i]),
                "confidence": prob_val
            })

        weights = get_feature_weights(model, payload.model_code, X_test, y_test, scaler=scaler, step_log=step_log)

        return {
            "status": "success",
            "accuracy": round(acc, 2),
            "training_records": len(df),
            "test_size": payload.test_size,
            "weights": weights,
            "predictions_sample": predictions_sample,
            "log": step_log
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }


class PredictData(BaseModel):
    model_code: str
    features: List[float]

@app.post("/predict")
async def predict_endpoint(payload: PredictData):
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'saved', f'model_{payload.model_code}.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'saved', f'scaler_{payload.model_code}.pkl')
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model not trained yet.")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        X_input = np.array([payload.features])
        X_scaled = scaler.transform(X_input) if scaler else X_input
            
        probs = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        
        prob_dict = {TARGET_MAP[c]: round(float(p)*100, 1) for c, p in zip(classes, probs)}
        survival = round(100.0 - prob_dict.get('Dropped Out / Failed', 0.0), 1)
        
        # Determine predicted class (highest probability class)
        pred_class_idx = int(np.argmax(probs))
        pred_class = int(classes[pred_class_idx])
        pred_label = TARGET_MAP.get(pred_class, 'Unknown')
        
        fc = []
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            for i, f in enumerate(FEATURES):
                fc.append({"feature": f, "importance": round(float(imps[i]), 4), "value": payload.features[i], "impact": round(float(imps[i]) * payload.features[i], 4)})
        else:
            w = 1.0 / len(FEATURES)
            for i, f in enumerate(FEATURES):
                fc.append({"feature": f, "importance": round(w, 4), "value": payload.features[i], "impact": round(w * payload.features[i], 4)})
                
        # Sort feature contributions by importance descending
        fc = sorted(fc, key=lambda x: x['importance'], reverse=True)
                
        return {
            "status": "success",
            "prediction": pred_class,
            "label": pred_label,
            "survival_probability": survival,
            "probabilities": prob_dict,
            "feature_contributions": fc
        }
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


def parse_spm_grades(text: str) -> Dict[str, str]:
    grades_map = {}
    
    # Map of possible OCR strings (lowercase) to standard System Subject Names
    subject_aliases = {
        "bahasa melayu": "Bahasa Melayu",
        "bahasa inggeris": "Bahasa Inggeris",
        "english": "Bahasa Inggeris",
        "mathematics": "Mathematics",
        "matematik": "Mathematics",
        "sejarah": "Sejarah",
        "physics": "Physics",
        "fizik": "Physics",
        "chemistry": "Chemistry",
        "kimia": "Chemistry",
        "biology": "Biology",
        "biologi": "Biology",
        "additional mathematics": "Additional Mathematics",
        "matematik tambahan": "Additional Mathematics",
        "prinsip perakaunan": "Prinsip Perakaunan",
        "ekonomi asas": "Ekonomi Asas",
        "ekonomi": "Ekonomi Asas",
        "pendidikan islam": "Pendidikan Islam",
        "pendidikan agama islam": "Pendidikan Islam",
        "science": "Science",
        "sains": "Science",
        "pendidikan moral": "Pendidikan Moral",
        "geografi": "Geografi",
        "tasawwur islam": "Tasawwur Islam",
        "pendidikan seni visual": "Pendidikan Seni Visual",
        "bahasa arab": "Bahasa Arab",
        "pendidikan al-quran": "Pendidikan Al-Quran",
        "pendidikan syariah": "Pendidikan Syariah",
        "kesusasteraan melayu": "Kesusasteraan Melayu"
    }
    
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        for alias, std_name in subject_aliases.items():
            if alias in line_lower:
                # Try to extract grade
                # We look for standard SPM grades: A+, A, A-, B+, B, C+, C, D, E, G
                # Using negative lookbehind/lookahead to avoid matching inside words
                match = re.search(r'(?<![a-zA-Z])(A\+|A-|A|B\+|B|C\+|C|D|E|G)(?![a-zA-Z])', line, re.IGNORECASE)
                if match:
                    # Prevent overwriting if we already matched this standard subject with a different alias
                    if std_name not in grades_map:
                        grades_map[std_name] = match.group(1).upper()
                    break # Move to next line once a subject is found in this line
                    
    return grades_map

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()
        text = ""

        # Check if PDF
        if filename.endswith(".pdf"):
            try:
                # Try extracting digital text
                pdf_document = fitz.open(stream=content, filetype="pdf")
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text()
                
                # If very little text, might be scanned PDF, fallback to image (Skipping for now due to Tesseract dependency)
            except Exception as pdf_e:
                print(f"PyMuPDF error: {pdf_e}")
                pass
        
        # If image or text wasn't extracted well
        if not text.strip():
            try:
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(io.BytesIO(content))
                    text = pytesseract.image_to_string(img)
            except Exception as img_e:
                # Give a specific error if tesseract is missing
                if "tesseract is not installed" in str(img_e).lower() or "tesseract" in str(img_e).lower():
                    return {"status": "error", "message": "Tesseract-OCR is required for images but is not installed on the server. Please install Tesseract-OCR."}
                else:
                    return {"status": "error", "message": f"Image OCR failed: {img_e}"}
        
        if not text.strip():
             return {"status": "error", "message": "Could not extract text from the provided file."}

        grades = parse_spm_grades(text)
        
        # If we successfully parsed text but no grades were found, tell the user
        if not grades:
             return {"status": "error", "message": "Text was extracted, but no recognized SPM subjects or grades were found."}
        
        return {
            "status": "success",
            "extracted_text": text,
            "parsed_grades": grades
        }
        
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}
