# Eco-Price Estimator ????
**Modul: Model Deployment & Maintenance (Projekt 1)**
**Dozent: Adrian Moser**

## ?? Projektbeschreibung
Dieses Projekt demonstriert den vollständigen ModelOps-Lebenszyklus einer KI-basierten Preisvorhersage für Gebrauchtwagen. Das System nimmt Fahrzeugdaten (Marke, Kilometer, Baujahr, Kraftstoff) entgegen und schätzt den aktuellen Marktwert mittels eines optimierten Gradient Boosting Regressors.

## ?? Technischer Stack
- **Model:** Scikit-learn Pipeline (StandardScaler, OneHotEncoder, GradientBoosting)
- **Format:** ONNX (Open Neural Network Exchange) für plattformunabhängige Inferenz
- **Backend:** FastAPI (Python) mit Pydantic-Datenvalidierung
- **Frontend:** HTML5/JavaScript (Responsive UI)
- **Container:** Docker (Python 3.13-slim)
- **CI/CD:** GitHub Actions (Automatisierte Tests & GHCR Image Build)

## ??? Installation & Lokaler Start

1. **Umgebung vorbereiten:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Lifecycle ausführen:**
   - **Datengenerierung:** `python data/generate_dataset.py`
   - **Training & ONNX Export:** `python model/train.py`
   - **API/Frontend starten:** `python app/main.py`

3. **URL öffnen:** [http://localhost:8000](http://localhost:8000)

## ?? Bonus: Monitoring & Maintenance
Das System verfügt über einen integrierten Monitoring-Service (Level 1 Maturity):
- **Endpoint:** `/monitoring`
- **Funktion:** Erkennt Drift und Outlier in den User-Anfragen (z.B. unrealistische Kilometerstände oder Baujahre). 
- **Health-Check:** `/health` liefert den Status des geladenen ONNX-Modells und dessen R2-Score.

## ?? Docker Deployment
Das Image ist optimiert für Azure Web Apps:
```bash
docker build -t ecoprice-estimator .
docker run -p 8000:8000 ecoprice-estimator
```
