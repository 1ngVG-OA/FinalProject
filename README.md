# Final Project

## Opertional Analytics, a.a. 2024/2025

The exam consists of the presentation of a project related to the elements introduced
during the course. 

It is essential to consider data preprocessing, implementation of at
least three forecasting algorithms (one each in the statistical, neural and regression trees
sets), and statistical comparison of the relative quality of the forecasts. 

>The project will be a Python solution. No Jupyter notebooks >will be accepted, and only the
>libraries used in the classes will be accepted, including >pandas, numpy, matplotlib, and
>those included in the slides. Additional libraries must be >explicitly agreed upon before the
>project is submitted. 

The standard project requires the three forecasting algorithms to be applied to two series
taken from the M3 dataset and relating to two different areas (microeconomics,
macroeconomics, industry, other). \
Standard projects, and projects using datasets widely
available on the internet, typically score around 27.
>Alternatively, higher scores can be achieved by working with >data of personal interest, on
>case studies related to the course topics and proposed by >the candidates themselves.

Complex solutions can be developed in groups of up to three students. However, the
discussion will be individual and the proposed solution must be able to run on the
machines in the labs and therefore on my server.

detail, although some implementation will be required.
Judging will be based on the interest of the case study (the more personal interest it is, the
better), the quality of the software solution, the size of the group and the quality of the
presentation, plus any participation awards earned during the year.


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

