# FinalProject

## Setup rapido (Windows PowerShell)

1. Crea ambiente virtuale (se non esiste):

```powershell
python -m venv .venv
```

2. Attiva il virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se PowerShell blocca gli script, imposta una volta sola:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Dipendenze

- `requirements.txt` → dipendenze **dirette/minimali** del progetto (più leggibile, utile in sviluppo).
- `requirements-full.txt` → snapshot **completo** (`pip freeze`) per riprodurre esattamente l'ambiente.

Installazione:

```powershell
pip install -r requirements.txt
```

oppure (riproducibilità completa):

```powershell
pip install -r requirements-full.txt
```

## Esecuzione script

```powershell
python demographic.py
python ogrin.py
```
