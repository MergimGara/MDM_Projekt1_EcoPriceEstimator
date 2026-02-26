# 🚗 Eco-Price Estimator (ModelOps Projekt 1)

[![ModelOps Pipeline](https://github.com/microsoft/onnxruntime/raw/master/docs/images/ORT_logo_for_white_bg.png)](https://onnxruntime.ai/)

## 📖 Projektübersicht
Dieses Repository enthält eine vollständige End-to-End Inferenz-Lösung für die Preisvorhersage von Gebrauchtwagen. Das Projekt wurde im Rahmen des Moduls **Model Deployment & Maintenance** (ZHAW) entwickelt und folgt dem **Maturity Level 1** des ModelOps-Ansatzes.

Es umfasst den gesamten Lifecycle:
1. **Daten:** Automatisierte Erzeugung synthetischer Marktdaten.
2. **Modell:** Training einer Scikit-Learn Pipeline mit automatisiertem Export in das **ONNX-Format**.
3. **Deployment:** Hochverfügbare **FastAPI** mit integrierter **Pydantic**-Validierung.
4. **Monitoring:** Echtzeit-Überwachung von Datenqualität und Drift via Admin-Dashboard.

---

## 🛠️ Technischer Stack & Architektur
- **Core:** Python 3.13
- **ML-Format:** ONNX (für maximale Portabilität und Performance)
- **API:** FastAPI (Asynchron, inkl. Swagger UI unter `/docs`)
- **Validation:** Pydantic (Type-Safety & Input-Screener)
- **Frontend:** Responsive HTML5 Dashboard mit Live-Monitoring
- **DevOps:** Docker (Multi-Stage-fähig) & GitHub Actions

---

## 🚀 Schnellstart (Lokal)

### 1. Umgebung aufsetzen
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Bash:
source .venv/Scripts/activate

pip install -r requirements.txt
```

### 2. Lifecycle ausführen (Training & Start)
```bash
# Datensatz generieren
python data/generate_dataset.py

# Modell trainieren und als ONNX exportieren
python model/train.py

# Web-App und API starten
python app/main.py
```
Öffne danach: **[http://localhost:8000](http://localhost:8000)**

---

## 🎖️ Bonus-Features (Grade 6.0 Kriterien)

### 📊 Integriertes Monitoring Dashboard
Statt nur Ergebnisse zu liefern, überwacht die API jede Anfrage:
- **Drift Detection:** Erkennt automatisch, wenn Eingabedaten (z.B. Baujahr, KM) nicht zum Modell-Scope passen.
- **Outlier Logging:** Jede Inferenz wird protokolliert und als "OK" oder "OUTLIER" markiert.
- **Visualisierung:** Ein interaktives Admin-Panel im Frontend visualisiert diese Metriken live.

### 🛡️ Robuste Input-Validierung
Dank Pydantic werden fehlerhafte Anfragen (z.B. negative Kilometerstände oder zukünftige Baujahre) sofort abgefangen, bevor sie das Modell belasten. Dies erhöht die Stabilität und Sicherheit der API.

---

## 🐳 Docker Deployment
Das Image ist für Cloud-Umgebungen (z.B. Azure Web Apps) optimiert:
```bash
# Build
docker build -t ecoprice-estimator .

# Run
docker run -p 8000:8000 ecoprice-estimator
```

---
**Autor:** Mergim Gara
**Modul:** Model Deployment & Maintenance (FS2026)
**Dozent:** Adrian Moser
