# -*- coding: utf-8 -*-
"""RouteAlert — recenser, comprendre et suivre les dégradations routières.

Toutes les ressources de l'application sont regroupées dans ce fichier.
Les données SQLite sont conservées hors du dossier livré. Aucun compte requis.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import sqlite3
from datetime import date, datetime, timedelta, timezone
import uuid
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from PIL import Image, ImageDraw, ImageOps
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage, KeepTogether,
)

VERSION = "1.0"
STATUSES = ["Signalé", "Vérifié", "Intervention prévue", "Réparé"]
SEVERITIES = ["Faible", "Modérée", "Élevée"]
EXPOSURES = ["Faible", "Moyenne", "Forte"]
STATUS_COLORS = {"Signalé": "#EB773C", "Vérifié": "#4777A3",
                 "Intervention prévue": "#987128", "Réparé": "#278367"}
PALETTE = ["#E17734", "#4777A3", "#BF941D", "#278367"]
MAX_RECORDS = 500
MAX_STORE_BYTES = 32 * 1024 * 1024
CSV_COLUMNS = ["route", "secteur", "latitude", "longitude", "date_signalement",
               "longueur_cm", "largeur_cm", "profondeur_cm", "gravite", "exposition", "description"]


def today() -> date:
    """Dates civiles selon le jour du serveur ; aucune date figée dans les essais."""
    return date.today()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def uid() -> str:
    return "RA-" + uuid.uuid4().hex[:10].upper()


def text_value(value, label, limit, required=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        value = ""
    value = str(value).strip()
    if required and not value:
        raise ValueError(f"{label} : ce champ est obligatoire.")
    if len(value) > limit:
        raise ValueError(f"{label} : maximum {limit} caractères.")
    return value


def number(value, label, low, high, strict_low=False):
    try:
        result = float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        raise ValueError(f"{label} : saisir un nombre valide.") from None
    if not math.isfinite(result) or not low <= result <= high or (strict_low and result == low):
        sign = ">" if strict_low else "≥"
        raise ValueError(f"{label} : valeur {sign} {low} et ≤ {high} attendue.")
    return result


def iso_day(value, label):
    try:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        raise ValueError(f"{label} : date attendue au format AAAA-MM-JJ.") from None


def normalize_photo(raw: bytes) -> str:
    """Vérifie l'image, enlève les métadonnées, redimensionne et compresse."""
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Photo trop volumineuse : maximum 5 Mo.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                if probe.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError("Utiliser une photo JPEG, PNG ou WebP.")
                if probe.width * probe.height > 20_000_000:
                    raise ValueError("Photo trop grande : maximum 20 millions de pixels.")
                probe.load()
                image = ImageOps.exif_transpose(probe).convert("RGB")
    except ValueError:
        raise
    except Exception:
        raise ValueError("Cette image est illisible ou endommagée.") from None
    image.thumbnail((1400, 1000))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=78, optimize=True)
    if buffer.tell() > 850_000:
        raise ValueError("Photo encore trop lourde après compression. Réduire sa taille.")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def validate_record(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Un signalement doit être un objet structuré.")
    r = {}
    r["id"] = text_value(raw.get("id", uid()), "Identifiant", 40, True)
    r["route"] = text_value(raw.get("route"), "Route / rue", 100, True)
    r["sector"] = text_value(raw.get("sector"), "Secteur / quartier", 100, True)
    r["latitude"] = number(raw.get("latitude"), "Latitude", -85, 85)
    r["longitude"] = number(raw.get("longitude"), "Longitude", -180, 180)
    r["length_cm"] = number(raw.get("length_cm"), "Longueur (cm)", 0, 2000, True)
    r["width_cm"] = number(raw.get("width_cm"), "Largeur (cm)", 0, 2000, True)
    r["depth_cm"] = number(raw.get("depth_cm"), "Profondeur (cm)", 0, 200, True)
    r["reported"] = iso_day(raw.get("reported"), "Date du signalement").isoformat()
    if iso_day(r["reported"], "Date") > today():
        raise ValueError("La date de signalement ne peut pas être dans le futur.")
    for field, options, default in (("severity", SEVERITIES, "Modérée"),
                                     ("exposure", EXPOSURES, "Moyenne"),
                                     ("status", STATUSES, "Signalé")):
        r[field] = raw.get(field, default)
        if r[field] not in options:
            raise ValueError(f"Valeur inconnue pour {field}.")
    r["description"] = text_value(raw.get("description"), "Description", 2000)
    r["team"] = text_value(raw.get("team"), "Équipe", 100)
    r["work_notes"] = text_value(raw.get("work_notes"), "Compte rendu", 2000)
    for field in ("planned", "repaired"):
        value = raw.get(field)
        r[field] = iso_day(value, "Date d'intervention").isoformat() if value else None
        if value and r[field] < r["reported"]:
            raise ValueError("Une intervention ne peut pas précéder le signalement.")
    if r["status"] == "Intervention prévue" and not r["planned"]:
        raise ValueError("Renseigner la date prévue de l'intervention.")
    if r["status"] == "Réparé":
        if not r["repaired"] or r["repaired"] > today().isoformat():
            raise ValueError("Une réparation terminée exige une date réelle, au plus tard aujourd'hui.")
    elif r["repaired"]:
        raise ValueError("La date de réparation est réservée au statut Réparé.")
    r["budget"] = number(raw.get("budget", 0), "Budget estimé (FCFA)", 0, 1e12)
    r["spent"] = number(raw.get("spent", 0), "Dépense réelle (FCFA)", 0, 1e12)
    for field in ("photo", "after_photo"):
        value = raw.get(field, "")
        if value:
            try:
                blob = base64.b64decode(value, validate=True)
            except Exception:
                raise ValueError("Photo sauvegardée invalide.") from None
            if len(blob) > 850_000:
                raise ValueError("Photo sauvegardée trop volumineuse.")
            try:
                with Image.open(io.BytesIO(blob)) as saved_image:
                    if saved_image.format != "JPEG" or saved_image.width * saved_image.height > 20_000_000:
                        raise ValueError("Photo sauvegardée invalide.")
                    saved_image.verify()
            except Exception:
                raise ValueError("Photo sauvegardée illisible.") from None
            r[field] = value
        else:
            r[field] = ""
    history = raw.get("history", [])
    if not isinstance(history, list) or len(history) > 200:
        raise ValueError("Historique invalide ou trop long (maximum 200 événements).")
    r["history"] = [{"at": text_value(e.get("at"), "Horodatage", 50),
                      "event": text_value(e.get("event"), "Événement", 300)}
                     for e in history if isinstance(e, dict)]
    return r


def validate_store(records):
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        raise ValueError(f"Le registre accepte au maximum {MAX_RECORDS} signalements.")
    result = [validate_record(r) for r in records]
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Deux signalements ont le même identifiant.")
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_STORE_BYTES:
        raise ValueError("Registre trop volumineux (32 Mo). Exporter et alléger les photos.")
    return result


def db_path():
    override = os.environ.get("ROUTEALERT_DATA_DIR")
    if override:
        root = Path(override).expanduser().resolve()
    elif os.name == "nt":
        root = Path.home() / "Documents" / "Codex" / "data" / "routealert"
    else:
        root = Path.home() / ".local" / "share" / "routealert"
    app_root = Path(__file__).resolve().parent
    if root == app_root or app_root in root.parents:
        raise ValueError("ROUTEALERT_DATA_DIR doit être situé hors du dossier de l'application.")
    root.mkdir(parents=True, exist_ok=True)
    return root / "routealert.sqlite3"


def db_load():
    connection = sqlite3.connect(db_path(), timeout=10)
    try:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS registry (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, payload TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO registry VALUES (1,0,'[]')")
            revision, payload = connection.execute("SELECT revision,payload FROM registry WHERE id=1").fetchone()
        return json.loads(payload), revision
    finally:
        connection.close()


def db_save(records, revision):
    payload = json.dumps(records, ensure_ascii=False, allow_nan=False)
    connection = sqlite3.connect(db_path(), timeout=10)
    try:
        with connection:
            cursor = connection.execute("UPDATE registry SET revision=revision+1,payload=? WHERE id=1 AND revision=?", (payload, revision))
            if cursor.rowcount != 1:
                raise ValueError("Le registre a changé dans une autre session. Actualiser la page avant de réessayer.")
    finally:
        connection.close()


def measurements(r):
    area = r["length_cm"] * r["width_cm"] / 10_000
    volume = area * r["depth_cm"] / 100
    return area, volume, volume * 1000


def score_parts(r, weights=(40, 35, 25)):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Choisir trois poids positifs ou nuls, avec un total supérieur à zéro.")
    factors = np.array([min(r["depth_cm"] / 10, 1),
                        min(measurements(r)[0] / 1, 1),
                        EXPOSURES.index(r["exposure"]) / 2])
    return factors * weights / weights.sum() * 100


def priority_score(r, weights=(40, 35, 25)):
    return float(score_parts(r, weights).sum())


def distance_m(a, b):
    lat1, lat2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dlat = lat2 - lat1
    dlon = math.radians(b["longitude"] - a["longitude"])
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(min(1, hav)))


def transition(record, status, planned, repaired, team, budget, spent, notes, after_photo=""):
    if status not in STATUSES:
        raise ValueError("Statut inconnu.")
    current = record["status"]
    allowed = {"Signalé": {"Signalé", "Vérifié"},
               "Vérifié": {"Vérifié", "Intervention prévue", "Réparé"},
               "Intervention prévue": {"Vérifié", "Intervention prévue", "Réparé"},
               "Réparé": {"Réparé", "Vérifié"}}
    if status not in allowed[current]:
        raise ValueError("Vérifier d'abord le signalement avant de planifier ou de clôturer une réparation.")
    result = copy.deepcopy(record)
    result.update(status=status, planned=planned if status == "Intervention prévue" else record.get("planned") if status == "Réparé" else None,
                  repaired=repaired if status == "Réparé" else None, team=team,
                  budget=budget, spent=spent, work_notes=notes)
    if after_photo:
        result["after_photo"] = after_photo
    result["history"].append({"at": stamp(), "event": f"Suivi : {current} → {status}. {notes[:150]}"})
    return validate_record(result)


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def simulate_volume(length, width, depth, errors, count=10000, seed=42):
    means = [number(length, "Longueur", 0, 2000, True),
             number(width, "Largeur", 0, 2000, True),
             number(depth, "Profondeur", 0, 200, True)]
    if len(errors) != 3:
        raise ValueError("Trois incertitudes sont attendues.")
    if not isinstance(count, int) or not 1000 <= count <= 50000:
        raise ValueError("Nombre de tirages attendu : 1 000 à 50 000.")
    rng = np.random.default_rng(seed)
    dimensions = []
    for mode, error in zip(means, errors):
        relative = number(error, "Incertitude (%)", 0, 90) / 100
        dimensions.append(np.full(count, mode) if relative == 0 else
                          rng.triangular(mode * (1 - relative), mode, mode * (1 + relative), count))
    return dimensions[0] * dimensions[1] * dimensions[2] / 1000  # cm³ → litres


def synthetic_photo(repaired=False):
    image = Image.new("RGB", (760, 440), "#444B4D")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(7)
    for _ in range(2800):
        x, y = rng.integers(0, 760), rng.integers(0, 440)
        shade = int(rng.integers(55, 100))
        draw.ellipse((int(x), int(y), int(x + 2), int(y + 2)), fill=(shade, shade, shade))
    draw.line((20, 60, 740, 100), fill="#F2DB83", width=12)
    if repaired:
        draw.rounded_rectangle((210, 160, 560, 340), radius=35, fill="#2C3436", outline="#697272", width=4)
    else:
        draw.polygon([(220, 210), (310, 160), (480, 175), (565, 250),
                      (500, 340), (300, 315), (200, 260)], fill="#1D2425", outline="#A48C71", width=8)
        draw.ellipse((250, 205, 510, 305), fill="#655848")
    draw.rectangle((0, 395, 760, 440), fill="#202729")
    draw.text((22, 411), "ILLUSTRATION FICTIVE - AUCUNE PHOTO DE ROUTE REELLE", fill="#F9DD73")
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def demo_records():
    examples = [
        ("Voie Alpha", "Secteur A (fictif)", 3.872, 11.512, 80, 50, 8, "Élevée", "Forte", "Vérifié", 14),
        ("Voie Alpha", "Secteur A (fictif)", 3.874, 11.515, 40, 30, 3, "Faible", "Faible", "Signalé", 5),
        ("Axe Delta", "Secteur B (fictif)", 3.867, 11.518, 120, 80, 12, "Élevée", "Forte", "Intervention prévue", 20),
        ("Rue Gamma", "Secteur C (fictif)", 3.877, 11.508, 60, 40, 5, "Modérée", "Moyenne", "Réparé", 18),
        ("Rue Gamma", "Secteur C (fictif)", 3.880, 11.516, 70, 45, 6, "Modérée", "Moyenne", "Signalé", 2),
        ("Axe Delta", "Secteur B (fictif)", 3.864, 11.511, 90, 60, 9, "Élevée", "Forte", "Intervention prévue", 12),
    ]
    result = []
    for i, (road, sector, lat, lon, length, width, depth, severity, exposure, status, days) in enumerate(examples):
        reported = today() - timedelta(days=days)
        repaired = today() - timedelta(days=3) if status == "Réparé" else None
        planned = (today() + timedelta(days=3 if i == 2 else -2)) if status == "Intervention prévue" else repaired
        record = dict(id=f"DEMO-{i + 1:03}", route=road, sector=sector, latitude=lat, longitude=lon,
                      reported=reported.isoformat(), length_cm=length, width_cm=width, depth_cm=depth,
                      severity=severity, exposure=exposure, status=status,
                      description="Exemple pédagogique entièrement fictif. Position illustrative, aucun constat réel.",
                      planned=planned.isoformat() if planned else None,
                      repaired=repaired.isoformat() if repaired else None,
                      team="Équipe démo" if status != "Signalé" else "",
                      budget=75000 if i in (2, 3, 5) else 0,
                      spent=68000 if repaired else 0,
                      work_notes="Réparation fictive documentée." if repaired else "",
                      photo=synthetic_photo(), after_photo=synthetic_photo(True) if repaired else "",
                      history=[{"at": reported.isoformat(), "event": "Exemple fictif chargé."}])
        result.append(validate_record(record))
    return result


def tabular(records, weights=(40, 35, 25)):
    return pd.DataFrame([{
        "Référence": r["id"], "Route": r["route"], "Secteur": r["sector"], "Statut": r["status"],
        "Signalement": r["reported"], "Gravité déclarée": r["severity"], "Exposition": r["exposure"],
        "Latitude": r["latitude"], "Longitude": r["longitude"], "Longueur (cm)": r["length_cm"],
        "Largeur (cm)": r["width_cm"], "Profondeur (cm)": r["depth_cm"],
        "Surface (m²)": measurements(r)[0], "Volume (L)": measurements(r)[2],
        "Score /100": round(priority_score(r, weights), 2), "Intervention prévue": r["planned"],
        "Réparation": r["repaired"], "Budget estimé (FCFA)": r["budget"], "Dépense réelle (FCFA)": r["spent"],
        "Description": r["description"], "Équipe": r["team"], "Compte rendu": r["work_notes"],
    } for r in records])


def literal_excel(frame):
    """XLSX : imposer le type texte même lorsque la chaîne commence par =."""
    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    sheet.title = "Signalements"
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        sheet.append([None if pd.isna(v) else v for v in row])
        for cell, value in zip(sheet[sheet.max_row], row):
            if isinstance(value, str):
                cell.data_type = "s"
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for col in sheet.columns:
        sheet.column_dimensions[col[0].column_letter].width = min(38, max(16, len(str(col[0].value)) + 2))
    buf = io.BytesIO()
    book.save(buf)
    return buf.getvalue()


def csv_bytes(frame):
    safe = frame.copy()
    for col in safe.columns:
        if safe[col].dtype == object or pd.api.types.is_string_dtype(safe[col]):
            safe[col] = safe[col].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
    return safe.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def import_table(raw, filename):
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Import limité à 2 Mo.")
    try:
        if filename.lower().endswith(".xlsx"):
            frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl", dtype=object)
        else:
            frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", sep=None, engine="python", dtype=object)
    except Exception:
        raise ValueError("Tableau illisible : utiliser le modèle CSV ou un classeur XLSX valide.") from None
    if not len(frame) or len(frame) > MAX_RECORDS:
        raise ValueError("Importer entre 1 et 500 lignes.")
    missing = set(CSV_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)))
    result = []
    for index, row in frame.fillna("").iterrows():
        mapping = dict(zip(CSV_COLUMNS, [row[c] for c in CSV_COLUMNS]))
        record = dict(route=mapping["route"], sector=mapping["secteur"], latitude=mapping["latitude"],
                      longitude=mapping["longitude"], reported=mapping["date_signalement"],
                      length_cm=mapping["longueur_cm"], width_cm=mapping["largeur_cm"],
                      depth_cm=mapping["profondeur_cm"], severity=mapping["gravite"],
                      exposure=mapping["exposition"], description=mapping["description"],
                      history=[{"at": stamp(), "event": "Signalement importé, à vérifier."}])
        try:
            result.append(validate_record(record))
        except ValueError as exc:
            raise ValueError(f"Ligne {index + 2} : {exc}") from None
    signatures = [(r["route"].casefold(), r["latitude"], r["longitude"], r["reported"]) for r in result]
    if len(set(signatures)) != len(signatures):
        raise ValueError("Lignes identiques détectées dans le fichier : retirer les doublons avant l'import.")
    return result


def import_backup(raw):
    if len(raw) > MAX_STORE_BYTES + 1024 * 1024:
        raise ValueError("Sauvegarde limitée à 33 Mo.")
    try:
        data = json.loads(raw)
    except Exception:
        raise ValueError("Sauvegarde JSON illisible.") from None
    if not isinstance(data, dict) or data.get("application") != "RouteAlert" or data.get("schema") != 1:
        raise ValueError("Ce fichier n'est pas une sauvegarde RouteAlert compatible.")
    records = validate_store(data.get("records"))
    for r in records:
        r["id"] = uid()
        r["history"].append({"at": stamp(), "event": "Copie restaurée depuis une sauvegarde."})
    return records


@st.cache_resource
def register_pdf_fonts():
    import reportlab
    fonts = Path(reportlab.__file__).resolve().parent / "fonts"
    for name, filename in (("RouteSans", "Vera.ttf"), ("RouteSans-Bold", "VeraBd.ttf"),
                           ("RouteSans-Italic", "VeraIt.ttf"), ("RouteSans-BoldItalic", "VeraBI.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("RouteSans", normal="RouteSans", bold="RouteSans-Bold",
                                  italic="RouteSans-Italic", boldItalic="RouteSans-BoldItalic")


def pdf_report(records, demo=False, detail=None):
    register_pdf_fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=(21 * cm, 29.7 * cm), rightMargin=1.4 * cm,
                            leftMargin=1.4 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="RouteAlert - Rapport des signalements", author="RouteAlert")
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "RouteSans-Bold" if "Heading" in style.name or style.name == "Title" else "RouteSans"
    styles.add(ParagraphStyle(name="CellRA", fontName="RouteSans", fontSize=8, leading=11))
    styles.add(ParagraphStyle(name="NoteRA", fontName="RouteSans", fontSize=9, leading=13, textColor=colors.HexColor("#45545A")))
    def para(value, style="CellRA"):
        return Paragraph(html.escape(str(value).replace("→", "->")).replace("\n", "<br/>"), styles[style])
    elements = [para("RouteAlert | Rapport des signalements", "Title"),
                para(f"Édité le {today().strftime('%d/%m/%Y')} - {'EXEMPLE FICTIF' if demo else 'Données du registre'}", "NoteRA"), Spacer(1, .4 * cm)]
    open_records = [r for r in records if r["status"] != "Réparé"]
    elements.append(para(f"{len(records)} signalement(s), {len(open_records)} ouvert(s). Volume géométrique ouvert : {sum(measurements(r)[2] for r in open_records):.2f} L.", "Normal"))
    elements.append(Spacer(1, .3 * cm))
    rows = [[para(c) for c in ("Référence / route", "Dimensions (cm)", "Statut / dates", "Volume / score")]]
    for r in sorted(records, key=lambda x: priority_score(x), reverse=True):
        rows.append([para(r["id"] + "\n" + r["route"] + "\n" + r["sector"]),
                     para(f"{r['length_cm']:g} x {r['width_cm']:g} x {r['depth_cm']:g}"),
                     para(f"{r['status']}\nSignalé : {r['reported']}\nRéparé : {r['repaired'] or '-'}"),
                     para(f"{measurements(r)[2]:.2f} L\n{priority_score(r):.1f} / 100")])
    table = Table(rows, colWidths=[6 * cm, 3.1 * cm, 5.3 * cm, 3.8 * cm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBCB53")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F6F4ED"), colors.white]),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    elements += [table, Spacer(1, .4 * cm), para("Hypothèses : volume = longueur x largeur x profondeur d'un parallélépipède équivalent. Score par défaut : profondeur 40 %, surface 35 %, exposition 25 %. Profondeur plafonnée à 10 cm, surface à 1 m² pour le score. Critères pédagogiques modifiables dans l'application ; aucun seuil réglementaire. La gravité est une appréciation déclarée distincte du score. Les besoins réels de réparation nécessitent un examen technique.", "NoteRA")]
    if detail:
        r = detail
        elements += [Spacer(1, .5 * cm), para(f"Fiche détaillée : {r['id']}", "Heading2"),
                     para(f"Coordonnées : {r['latitude']:.6f}, {r['longitude']:.6f}\nDescription : {r['description']}\nÉquipe : {r['team'] or '-'}\nBudget estimé : {r['budget']:,.0f} FCFA\nDépense réelle : {r['spent']:,.0f} FCFA\nCompte rendu : {r['work_notes'] or '-'}", "NoteRA")]
        for key, caption in (("photo", "État signalé"), ("after_photo", "Après intervention")):
            if r[key]:
                photo = io.BytesIO(base64.b64decode(r[key]))
                w, h = ImageReader(photo).getSize()
                scale = min(16 * cm / w, 8 * cm / h)
                elements += [Spacer(1, .3 * cm), KeepTogether([para(caption, "Heading3"), PDFImage(photo, width=w * scale, height=h * scale)])]
        elements += [para("Historique", "Heading3")]
        elements += [para(e["at"] + " - " + e["event"], "NoteRA") for e in r["history"]]
    def footer(canvas, document):
        canvas.setFont("RouteSans", 8)
        canvas.setFillColor(colors.HexColor("#566167"))
        canvas.drawString(1.4 * cm, .8 * cm, "RouteAlert - Outil pédagogique de suivi")
        canvas.drawRightString(19.6 * cm, .8 * cm, f"Page {document.page}")
    doc.build(elements, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


CSS = """
<style>
.stApp {background:#F3F2EC;color:#232D2F;}
[data-testid="stAppViewContainer"] {background:radial-gradient(#D7D9D3 .6px,transparent .6px) 0 0/18px 18px,#F3F2EC;}
[data-testid="stHeader"] {background:rgba(243,242,236,.94);}
.block-container {max-width:1370px;padding-top:4.5rem;padding-bottom:3rem;}
h1,h2,h3,h4,p,label {overflow-wrap:anywhere;}
h1,h2,h3,h4 {color:#232D2F!important;letter-spacing:-.035em;}
[data-testid="stMain"] {color:#232D2F;}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] {color:#232D2F;}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {color:#607067;}
[data-testid="stSidebar"] {background:#242D30;border-right:1px solid #3A4547;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:#F3F0DF!important;}
[data-testid="stSidebarNav"] a {color:#F3F0DF!important;border-radius:10px;}
[data-testid="stSidebarNav"] a[aria-current="page"] {background:#E8C849!important;color:#242D30!important;}
[data-testid="stSidebarNav"] a[aria-current="page"] span {color:#242D30!important;}
[data-testid="stSidebarNav"] a[aria-current="page"] p {color:#242D30!important;}
[data-testid="stSidebar"] .stButton button {background:#E8C849;color:#20292C;border:0;font-weight:700;}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:18px;}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {background:#FFFEF9;border-radius:18px;}
[data-testid="stMain"] input,[data-testid="stMain"] textarea {background:#FFFEFA;color:#232D2F;}
[data-testid="stMain"] [data-baseweb="select"]>div {background:#FFFEFA;color:#232D2F;}
.stButton button,.stFormSubmitButton button,.stDownloadButton button {border-radius:11px;min-height:44px;}
[data-testid="stMain"] button[kind="primary"], [data-testid="stMain"] .stFormSubmitButton button {background:#E8C849!important;color:#242D30!important;border-color:#C6A737!important;font-weight:750;}
[data-testid="stMain"] button[kind="secondary"] {background:#FFFEF9;color:#263235;border-color:#D3D7CC;}
button p {color:inherit!important;}
.ra-brand {display:flex;gap:12px;align-items:center;margin:4px 0 20px;}
.ra-logo {display:grid;place-items:center;width:44px;height:44px;border-radius:12px;background:#E8C849;color:#242D30;font-size:25px;font-weight:900;transform:rotate(-5deg);}
.ra-brand strong {font-size:24px;color:#F6F2DF;letter-spacing:-1px;}
.ra-brand small {display:block;color:#B7C3BF;font-size:11px;letter-spacing:2px;}
.ra-hero {position:relative;overflow:hidden;isolation:isolate;border-radius:24px;background:#242D30;color:#F5F3E5;min-height:260px;display:grid;grid-template-columns:1.4fr 1fr;margin-bottom:24px;border:1px solid #435052;box-shadow:0 15px 38px #24303112;}
.ra-hero-copy {position:relative;z-index:2;padding:32px 38px;}
.ra-overline {color:#E8C849;font-size:11px;font-weight:800;letter-spacing:2px;}
.ra-hero h1 {font-size:clamp(29px,3.5vw,46px);line-height:1.2;color:#F6F3E5!important;margin:14px 0 12px!important;padding:0!important;max-width:610px;}
.ra-hero p {font-size:14px;color:#C7D0CA!important;max-width:490px;line-height:1.7;}
.ra-tag {display:inline-block;border:1px solid #64705F;border-radius:20px;color:#E9CE63;font-size:11px;padding:5px 11px;margin-top:10px;}
.ra-road-scene {position:relative;min-height:270px;background:radial-gradient(ellipse at center,#46524B 0,transparent 67%);}
.ra-road {position:absolute;left:28%;top:-35%;width:170px;height:170%;transform:rotate(34deg);border-left:3px solid #E8C849;border-right:3px solid #E8C849;background:#333E40;box-shadow:0 0 0 14px #ffffff04;}
.ra-road:before {content:'';position:absolute;left:50%;width:4px;height:100%;background:repeating-linear-gradient(to bottom,#F1EDD3 0 30px,transparent 30px 56px);}
.ra-pin {position:absolute;z-index:2;width:22px;height:22px;border:5px solid #F0C84E;background:#ED7C39;border-radius:50%;box-shadow:0 0 0 13px #EDC4481C;}
.ra-pin.one {left:40%;top:30%;}.ra-pin.two {left:62%;top:60%;background:#81B99C;border-color:#A1D6B7;}
.ra-scan {position:absolute;bottom:22px;right:22px;padding:9px 13px;border-radius:11px;border:1px solid #ffffff26;background:#222C2DE8;color:#D5DEC8;font-size:11px;}
.ra-kpis {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin-bottom:26px;}
.ra-kpi {position:relative;min-width:0;border-radius:17px;background:#FFFEF9;border:1px solid #DADDD2;padding:19px 20px;overflow:hidden;}
.ra-kpi:before {content:'';position:absolute;left:0;top:22px;height:35px;width:4px;background:var(--accent,#E8C849);border-radius:0 4px 4px 0;}
.ra-kpi small {font-size:12px;color:#64716C;display:block;line-height:1.5;}
.ra-kpi strong {display:block;font-size:clamp(20px,2.2vw,31px);line-height:1.4;letter-spacing:-1px;color:#263235;overflow-wrap:anywhere;}
.ra-kpi span {font-size:11px;color:#7A837A;}
.ra-title {font-size:12px;color:#7A846F;letter-spacing:1.8px;font-weight:800;margin:10px 0 2px;}
.ra-mini {border-radius:13px;border:1px dashed #BFC4B7;padding:16px;color:#556359;background:#EEEFE5;margin:8px 0 15px;font-size:13px;line-height:1.7;}
.ra-badge {display:inline-block;background:#F7E9BB;color:#705A17;padding:5px 9px;border-radius:7px;font-size:11px;font-weight:700;}
.ra-card-title {font-size:20px;font-weight:750;letter-spacing:-.5px;margin:8px 0;}
.ra-progress {height:7px;background:#E8EBDF;border-radius:7px;overflow:hidden;margin:10px 0;}
.ra-progress div {height:100%;background:#E4BD33;border-radius:7px;}
.ra-empty {padding:40px 25px;border:1px dashed #B7C0AF;border-radius:20px;text-align:center;background:#FAFAF0;}
.ra-empty strong {font-size:22px;color:#344039;}
.ra-empty p {font-size:14px;color:#67735F;}
@media(max-width:900px){.ra-hero-copy{padding:28px;}.ra-hero{grid-template-columns:1.5fr .8fr;}.ra-kpis{gap:10px;}.ra-kpi{padding:16px 13px;}.ra-kpi strong{font-size:26px;}}
@media(max-width:640px){.block-container{padding:4rem 1rem 3rem;}.ra-hero{grid-template-columns:1fr;min-height:0;}.ra-hero-copy{padding:25px 23px;}.ra-hero h1{font-size:32px;}.ra-road-scene{display:none;}.ra-kpis{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}.ra-kpi{padding:14px;}.ra-kpi strong{font-size:25px;}.ra-kpi small{font-size:11px;}.ra-hero p{font-size:13px;}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important;transition:none!important;}}
</style>
"""


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=1):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def heading(kicker, title, subtitle=""):
    st.html(f'<div class="ra-title">{esc(kicker)}</div>')
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def kpis(items):
    st.html('<div class="ra-kpis">' + "".join(
        f'<div class="ra-kpi" style="--accent:{accent}"><small>{esc(label)}</small><strong>{esc(value)}</strong><span>{esc(note)}</span></div>'
        for label, value, note, accent in items) + '</div>')


def plot(figure, label, height=310, key=None):
    figure.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Segoe UI, Arial", color="#35413F", size=12),
                         height=height, margin=dict(l=16, r=16, t=35, b=24),
                         legend=dict(orientation="h", y=-.2), colorway=PALETTE)
    figure.update_xaxes(gridcolor="#E2E5DA", automargin=True)
    figure.update_yaxes(gridcolor="#E2E5DA", automargin=True)
    st.plotly_chart(figure, width="stretch", theme=None, config={"displaylogo": False, "responsive": True},
                    alt=label, key=key)


def current_records():
    return st.session_state["ra_demo_records"] if st.session_state["ra_mode"] == "Exemple fictif" else st.session_state["ra_real"]


def persist(records):
    records = validate_store(records)
    if st.session_state["ra_mode"] == "Exemple fictif":
        st.session_state["ra_demo_records"] = records
    else:
        db_save(records, st.session_state["ra_revision"])
    st.session_state["ra_notice"] = "Modifications enregistrées dans l'exemple fictif." if st.session_state["ra_mode"] == "Exemple fictif" else "Modifications enregistrées."


def put_record(record):
    records = copy.deepcopy(current_records())
    indices = [i for i, r in enumerate(records) if r["id"] == record["id"]]
    if indices:
        records[indices[0]] = record
    else:
        records.append(record)
    persist(records)


def open_record(record_id):
    st.session_state["ra_selected"] = record_id
    st.switch_page(st.session_state["ra_pages"]["registre"])


def require_records():
    records = current_records()
    if not records:
        st.html('<div class="ra-empty"><strong>Ton registre est prêt.</strong><p>Ajoute ton premier signalement ou charge l’exemple fictif depuis le menu.</p></div>')
        st.page_link(st.session_state["ra_pages"]["signaler"], label="Créer un signalement", icon=":material/add_location_alt:")
    return records


def picker(records, key):
    ids = [r["id"] for r in records]
    selected = st.session_state.get("ra_selected")
    lookup = {r["id"]: r for r in records}
    return lookup[st.selectbox("Signalement", ids, index=ids.index(selected) if selected in ids else 0,
                             format_func=lambda x: f"{x} · {lookup[x]['route']}", key=key)]


def record_card(r, index, key_prefix):
    with st.container(border=True):
        st.html(f'<span class="ra-badge">{esc(r["id"])}</span><div class="ra-card-title">{esc(r["route"])}</div>')
        st.caption(f"{r['sector']} · {r['status']}")
        if r["photo"]:
            st.image(base64.b64decode(r["photo"]), width="stretch", alt=f"État signalé sur {r['route']}")
        st.markdown(f"**{fmt(measurements(r)[2])} L** estimés · profondeur **{fmt(r['depth_cm'])} cm**")
        score = priority_score(r)
        st.html(f'<div class="ra-progress"><div style="width:{score:.2f}%"></div></div>')
        st.caption(f"Score pédagogique : {fmt(score)} / 100 · gravité déclarée : {r['severity'].lower()}")
        if st.button("Ouvrir la fiche", key=f"{key_prefix}_{index}", icon=":material/arrow_outward:", width="stretch"):
            open_record(r["id"])


def show_map(records, key, compact=False):
    if not records:
        return
    mode = "Fond de carte en ligne" if compact else st.segmented_control(
        "Affichage", ["Fond de carte en ligne", "Repères sans fond de carte"], default="Fond de carte en ligne", key=key + "_mode")
    if mode == "Repères sans fond de carte":
        fig = go.Figure()
        for status in STATUSES:
            rows = [r for r in records if r["status"] == status]
            if rows:
                fig.add_trace(go.Scatter(x=[r["longitude"] for r in rows], y=[r["latitude"] for r in rows], mode="markers",
                                         marker=dict(size=15, color=STATUS_COLORS[status], line=dict(color="white", width=2)),
                                         text=[r["id"] + " · " + r["route"] for r in rows], name=status,
                                         hovertemplate="%{text}<br>Longitude %{x:.5f}<br>Latitude %{y:.5f}<extra></extra>"))
        fig.update_layout(xaxis_title="Longitude (°)", yaxis_title="Latitude (°)")
        plot(fig, "Positions des signalements en latitude et longitude, sans fond géographique", 420, key)
        st.caption("Vue de coordonnées disponible sans chargement de tuiles. Elle ne représente pas les tracés des routes.")
        return
    data = []
    for r in records:
        hex_color = STATUS_COLORS[r["status"]].lstrip("#")
        data.append(dict(id=r["id"], position=[r["longitude"], r["latitude"]],
                         color=[int(hex_color[i:i+2], 16) for i in (0, 2, 4)], radius=8 + priority_score(r) / 8,
                         label=r["id"], status=r["status"], volume=f"{measurements(r)[2]:.1f} L"))
    layer = pdk.Layer("ScatterplotLayer", data=data, id="reports", get_position="position", get_fill_color="color",
                      get_radius="radius", radius_units=pdk.types.String("pixels"), pickable=True, stroked=True,
                      get_line_color=[255, 255, 240], line_width_min_pixels=2)
    # Adapte l'emprise aux points au lieu de masquer les signalements éloignés.
    latitudes = [r["latitude"] for r in records]
    longitudes = [r["longitude"] for r in records]
    span = max(max(latitudes) - min(latitudes), max(longitudes) - min(longitudes), .008)
    zoom = max(1, min(16, math.log2(180 / span)))
    view = pdk.ViewState(latitude=float(np.mean(latitudes)), longitude=float(np.mean(longitudes)), zoom=zoom, pitch=0)
    deck = pdk.Deck(layers=[layer], initial_view_state=view, map_provider="carto", map_style="light",
                    tooltip={"text": "{label}\n{status}\nVolume estimé : {volume}",
                             "style": {"backgroundColor": "#242D30", "color": "#F3E9B5"}})
    event = st.pydeck_chart(deck, height=345 if compact else 480, on_select="rerun", selection_mode="single-object",
                            key=key, alt="Carte des signalements. Les couleurs indiquent les statuts.")
    objects = event.get("selection", {}).get("objects", {}).get("reports", [])
    if objects:
        selected = objects[0].get("id")
        found = next((r for r in records if r["id"] == selected), None)
        if found:
            st.caption(f"Sélection : {found['id']} · {found['route']}")
            if st.button("Ouvrir ce signalement", key=key + "_open", icon=":material/open_in_new:"):
                open_record(selected)
    st.caption("Tuiles © CARTO · données © OpenStreetMap. Internet nécessaire au fond de carte. Cliquer un repère pour le sélectionner.")


def dashboard():
    records = current_records()
    st.html('''<div class="ra-hero"><div class="ra-hero-copy"><div class="ra-overline">ROUTEALERT / OBSERVATOIRE ROUTIER</div>
    <h1>Voir la dégradation.<br>Organiser l’intervention.</h1><p>Une carte, des constats documentés et un suivi clair des réparations de la chaussée.</p>
    <div class="ra-tag">SIGNALER → VÉRIFIER → INTERVENIR</div></div><div class="ra-road-scene" aria-hidden="true"><div class="ra-road"></div><div class="ra-pin one"></div><div class="ra-pin two"></div><div class="ra-scan">CHAQUE CONSTAT COMPTE</div></div></div>''')
    active = [r for r in records if r["status"] != "Réparé"]
    overdue = [r for r in active if r["status"] == "Intervention prévue" and r["planned"] < today().isoformat()]
    fixed = len(records) - len(active)
    kpis([("Signalements", str(len(records)), "dans le registre", "#E8C849"),
          ("À traiter", str(len(active)), f"{len(overdue)} intervention(s) en retard", "#E17734"),
          ("Réparations terminées", str(fixed), f"{fmt(100 * fixed / len(records), 0) if records else '0'} % du registre", "#278367"),
          ("Volume ouvert", fmt(sum(measurements(r)[2] for r in active)) + " L", "approximation géométrique", "#4777A3")])
    with st.container(horizontal=True):
        st.page_link(st.session_state["ra_pages"]["signaler"], label="Nouveau signalement", icon=":material/add_circle:")
        st.page_link(st.session_state["ra_pages"]["carte"], label="Explorer la carte", icon=":material/map:")
    if not require_records():
        return
    left, right = st.columns([1.65, 1])
    with left:
        with st.container(border=True):
            st.subheader("Le réseau sous observation")
            show_map(records, "dashboard_map", compact=True)
    with right:
        with st.container(border=True):
            st.subheader("Du constat à la réparation")
            counts = [sum(r["status"] == s for r in records) for s in STATUSES]
            fig = go.Figure(go.Bar(y=STATUSES[::-1], x=counts[::-1], orientation="h",
                                   marker_color=PALETTE[::-1], text=counts[::-1], textposition="auto"))
            fig.update_layout(xaxis_title="Nombre de signalements", xaxis_dtick=1)
            plot(fig, "Répartition des signalements par statut", 260)
            completed = [r for r in records if r["repaired"]]
            if completed:
                mean_days = np.mean([(date.fromisoformat(r["repaired"]) - date.fromisoformat(r["reported"])).days for r in completed])
                st.caption(f"Délai moyen observé : {fmt(mean_days)} jours sur {len(completed)} réparation(s).")
            else:
                st.caption("Le délai moyen apparaîtra après une première réparation enregistrée.")
    st.subheader("Signalements ouverts à examiner")
    st.caption("Classement pédagogique par défaut. Vérifier les constats et le contexte avant de programmer les travaux.")
    top = sorted(active, key=priority_score, reverse=True)[:3]
    if top:
        for i, col in enumerate(st.columns(len(top))):
            with col:
                record_card(top[i], i, "dash_card")
    else:
        st.success("Tous les signalements du registre sont marqués réparés.")


def map_page():
    heading("LOCALISER", "Carte des signalements", "Les positions fictives servent uniquement à la démonstration.")
    records = require_records()
    if not records:
        return
    c1, c2 = st.columns(2)
    with c1:
        status = st.multiselect("Statuts affichés", STATUSES, default=STATUSES)
    with c2:
        sector = st.selectbox("Secteur", ["Tous"] + sorted({r["sector"] for r in records}))
    filtered = [r for r in records if r["status"] in status and (sector == "Tous" or r["sector"] == sector)]
    if not filtered:
        st.info("Aucun signalement ne correspond aux filtres.")
        return
    show_map(filtered, "full_map")
    st.html('<div style="display:flex;flex-wrap:wrap;gap:15px;margin-bottom:12px">' + ''.join(
        f'<span style="font-size:13px;color:#36433D"><span style="display:inline-block;width:9px;height:9px;background:{STATUS_COLORS[s]};border-radius:50%;margin-right:6px"></span>{s}</span>' for s in STATUSES) + '</div>')
    r = picker(filtered, "map_record")
    if st.button("Consulter la fiche sélectionnée", icon=":material/description:"):
        open_record(r["id"])


def signal_form(existing=None):
    r = existing or dict(route="", sector="", latitude=3.87, longitude=11.51, length_cm=80., width_cm=50., depth_cm=8.,
                          severity="Modérée", exposure="Moyenne", reported=today().isoformat(), description="", photo="")
    form_key = "edit_" + r["id"] if existing else "new_signal"
    with st.form(form_key):
        st.markdown("**1 / Localisation et date**")
        c1, c2 = st.columns(2)
        with c1:
            road = st.text_input("Route / rue *", r["route"], max_chars=100)
            lat = st.number_input("Latitude (°)", -85., 85., float(r["latitude"]), format="%.6f", step=.0001)
        with c2:
            sector = st.text_input("Secteur / quartier *", r["sector"], max_chars=100)
            lon = st.number_input("Longitude (°)", -180., 180., float(r["longitude"]), format="%.6f", step=.0001)
        reported = st.date_input("Date du signalement", date.fromisoformat(r["reported"]), max_value=today())
        st.caption("Coordonnées WGS84 en degrés décimaux. Vérifier leur emplacement ; les valeurs initiales sont indicatives.")
        st.markdown("**2 / Dimensions approximatives**")
        a, b, c = st.columns(3)
        with a:
            length = st.number_input("Longueur (cm)", .1, 2000., float(r["length_cm"]), step=1.)
        with b:
            width = st.number_input("Largeur (cm)", .1, 2000., float(r["width_cm"]), step=1.)
        with c:
            depth = st.number_input("Profondeur (cm)", .1, 200., float(r["depth_cm"]), step=.5)
        a, b = st.columns(2)
        with a:
            severity = st.selectbox("Gravité déclarée", SEVERITIES, index=SEVERITIES.index(r["severity"]))
        with b:
            exposure = st.selectbox("Exposition à la circulation (appréciation)", EXPOSURES, index=EXPOSURES.index(r["exposure"]))
        st.markdown("**3 / Documentation**")
        description = st.text_area("Description du constat", r["description"], max_chars=2000, height=100)
        photo = st.file_uploader("Photo du nid-de-poule (facultative, 5 Mo maximum)", type=["jpg", "jpeg", "png", "webp"], max_upload_size=5)
        remove = st.checkbox("Retirer la photo actuelle") if existing and r["photo"] else False
        submitted = st.form_submit_button("Enregistrer les modifications" if existing else "Enregistrer le signalement", icon=":material/save:", type="primary", width="stretch")
    if submitted:
        try:
            new = copy.deepcopy(r)
            new.update(route=road, sector=sector, latitude=lat, longitude=lon, reported=reported.isoformat(),
                       length_cm=length, width_cm=width, depth_cm=depth, severity=severity, exposure=exposure, description=description)
            new["photo"] = normalize_photo(photo.getvalue()) if photo else "" if remove else r.get("photo", "")
            new.setdefault("history", []).append({"at": stamp(), "event": "Constat modifié." if existing else "Signalement créé, à vérifier."})
            new = validate_record(new)
            put_record(new)
            st.session_state["ra_selected"] = new["id"]
            st.rerun()
        except (ValueError, sqlite3.Error, OSError) as exc:
            st.error(str(exc))


def new_page():
    heading("COLLECTER", "Nouveau signalement", "Un constat documenté pourra ensuite être vérifié, planifié et suivi.")
    with st.container(border=True):
        signal_form()
    st.caption("Les dimensions sont saisies manuellement : aucune mesure automatique n'est déduite d'une photographie.")


def register_page():
    heading("DOCUMENTER", "Registre et fiches", "Retrouver le constat, corriger ses informations et consulter son historique.")
    records = require_records()
    if not records:
        return
    query = st.text_input("Rechercher une route, un secteur ou une référence", placeholder="Exemple : Voie Alpha")
    filtered = [r for r in records if query.casefold() in (r["id"] + " " + r["route"] + " " + r["sector"]).casefold()]
    if not filtered:
        st.info("Aucun résultat.")
        return
    r = picker(filtered, "register_pick")
    st.session_state["ra_selected"] = r["id"]
    st.caption(f"{len(filtered)} résultat(s) · référence {r['id']}")
    kpis([("Surface", fmt(measurements(r)[0], 2) + " m²", "rectangle équivalent", "#E8C849"),
          ("Volume", fmt(measurements(r)[2], 2) + " L", "approximation géométrique", "#4777A3"),
          ("Score", fmt(priority_score(r)) + " /100", "critères par défaut", "#E17734"),
          ("État", r["status"], "suivi de l'intervention", "#278367")])
    c1, c2 = st.columns(2)
    with c1:
        if r["photo"]:
            st.image(base64.b64decode(r["photo"]), width="stretch", alt="État lors du signalement", caption="État signalé")
        else:
            st.info("Aucune photo jointe au signalement.")
    with c2:
        if r["after_photo"]:
            st.image(base64.b64decode(r["after_photo"]), width="stretch", alt="État après intervention", caption="Après intervention (consulter la date dans l'historique)")
        else:
            st.markdown(f"**{r['route']}** · {r['sector']}")
            st.markdown(r["description"] or "Aucune description.")
            st.caption(f"Latitude : {r['latitude']:.6f} · Longitude : {r['longitude']:.6f}")
    nearby = [x for x in records if x["id"] != r["id"] and distance_m(r, x) < 30]
    if nearby:
        st.info("Signalement(s) à moins de 30 m, doublons possibles à examiner : " + ", ".join(x["id"] for x in nearby))
    with st.expander("Modifier les informations du constat"):
        signal_form(r)
    with st.expander("Historique des enregistrements", expanded=True):
        for event in reversed(r["history"][-15:]):
            st.caption(event["at"])
            st.text(event["event"])
        if len(r["history"]) > 15:
            st.caption("Les 15 derniers événements sont affichés. L'export JSON contient l'historique complet.")
    st.download_button("Télécharger cette fiche PDF", pdf_report([r], st.session_state["ra_mode"] == "Exemple fictif", r),
                       file_name=r["id"] + ".pdf", mime="application/pdf", icon=":material/download:")
    with st.expander("Supprimer ce signalement"):
        st.caption("La suppression retire également ses photos et son historique du registre actif.")
        with st.form("delete_" + r["id"]):
            confirm = st.text_input("Recopier la référence pour confirmer", placeholder=r["id"])
            delete = st.form_submit_button("Supprimer le signalement")
        if delete:
            if confirm.strip() != r["id"]:
                st.error("La référence saisie ne correspond pas.")
            else:
                try:
                    persist([x for x in records if x["id"] != r["id"]])
                    st.session_state.pop("ra_selected", None)
                    st.rerun()
                except (ValueError, sqlite3.Error, OSError) as exc:
                    st.error(str(exc))


def interventions_page():
    heading("ORGANISER", "Priorités et interventions", "Un score explicable, puis une décision documentée après vérification.")
    records = require_records()
    if not records:
        return
    with st.expander("Comprendre et ajuster le score", expanded=False):
        st.markdown("Trois composantes : profondeur / 10 cm, surface / 1 m², exposition faible = 0, moyenne = 0,5, forte = 1. Les deux premières sont plafonnées à 1.")
        a, b, c = st.columns(3)
        with a:
            w1 = st.number_input("Poids profondeur", 0, 100, 40)
        with b:
            w2 = st.number_input("Poids surface", 0, 100, 35)
        with c:
            w3 = st.number_input("Poids exposition", 0, 100, 25)
        st.caption("Les poids sont normalisés par leur somme. Les seuils sont des choix pédagogiques. La gravité déclarée reste une observation distincte. Les rapports utilisent les poids par défaut 40/35/25.")
    weights = (w1, w2, w3)
    if sum(weights) == 0:
        st.error("Le total des poids doit être supérieur à zéro.")
        return
    active = [r for r in records if r["status"] != "Réparé"]
    if active:
        frame = tabular(active, weights).sort_values("Score /100", ascending=False)
        st.dataframe(frame[["Référence", "Route", "Statut", "Score /100", "Volume (L)", "Intervention prévue"]], hide_index=True,
                      width="stretch", alt="Classement des signalements ouverts selon les poids choisis")
    else:
        st.success("Aucun signalement ouvert.")
    st.subheader("Mettre à jour une intervention")
    r = picker(records, "work_pick")
    if r["status"] == "Intervention prévue" and r["planned"] < today().isoformat():
        st.info(f"Date prévue dépassée : {r['planned']}. Actualiser le suivi de cette intervention.")
    st.caption("Un signalement doit passer par Vérifié avant une planification ou une réparation. Réouvrir un dossier réparé se fait en le repassant à Vérifié.")
    with st.form("work_" + r["id"]):
        status = st.selectbox("Statut de l'intervention", STATUSES, index=STATUSES.index(r["status"]))
        a, b = st.columns(2)
        with a:
            planned = st.date_input("Date prévue (utilisée pour une intervention prévue)", date.fromisoformat(r["planned"]) if r["planned"] else max(today(), date.fromisoformat(r["reported"])))
            team = st.text_input("Équipe / responsable", r["team"], max_chars=100)
            budget = st.number_input("Budget estimé (FCFA)", 0., 1e12, float(r["budget"]), step=1000.)
        with b:
            repaired = st.date_input("Date réelle de réparation (utilisée si Réparé)", date.fromisoformat(r["repaired"]) if r["repaired"] else today(), max_value=today())
            spent = st.number_input("Dépense réelle cumulée (FCFA)", 0., 1e12, float(r["spent"]), step=1000.)
            after = st.file_uploader("Photo après intervention (facultative)", type=["jpg", "jpeg", "png", "webp"], max_upload_size=5)
        notes = st.text_area("Compte rendu de l'intervention", r["work_notes"], max_chars=2000)
        save = st.form_submit_button("Enregistrer le suivi", type="primary", icon=":material/task_alt:", width="stretch")
    if save:
        try:
            result = transition(r, status, planned.isoformat(), repaired.isoformat(), team, budget, spent, notes,
                                normalize_photo(after.getvalue()) if after else "")
            put_record(result)
            st.rerun()
        except (ValueError, sqlite3.Error, OSError) as exc:
            st.error(str(exc))
    st.subheader("Évaluation financière")
    st.caption("Montants saisis manuellement, tous dossiers du registre. Un budget nul signifie qu'aucun montant estimé n'a été saisi.")
    sectors = sorted({r["sector"] for r in records})
    fig = go.Figure()
    for field, label, color in (("budget", "Budget estimé", "#BF941D"), ("spent", "Dépense réelle", "#278367")):
        fig.add_bar(y=sectors, x=[sum(r[field] for r in records if r["sector"] == s) for s in sectors],
                    name=label, orientation="h", marker_color=color)
    fig.update_layout(barmode="group", xaxis_title="Montant (FCFA)")
    plot(fig, "Comparaison des budgets estimés et des dépenses réelles par secteur", max(320, len(sectors) * 40))


def dimensions_figure(r):
    length, width, depth = r["length_cm"], r["width_cm"], r["depth_cm"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Vue de dessus", "Coupe simplifiée"), horizontal_spacing=.18)
    for col, x, y in ((1, length, width), (2, length, depth)):
        fig.add_trace(go.Scatter(x=[0, x, x, 0, 0], y=[0, 0, y, y, 0], mode="lines", fill="toself",
                                 fillcolor="#E8C84966", line=dict(color="#786927", width=2), showlegend=False,
                                 hovertemplate="%{x:.1f} cm · %{y:.1f} cm<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="Longueur (cm)", row=1, col=col)
        fig.update_yaxes(title_text="Largeur (cm)" if col == 1 else "Profondeur (cm)", row=1, col=col)
    return fig


def analysis_page():
    heading("COMPRENDRE", "Calculs et incertitudes", "Relier les mesures de terrain à une estimation géométrique et à ses limites.")
    records = require_records()
    if not records:
        return
    r = picker(records, "analysis_pick")
    area, volume, litres = measurements(r)
    kpis([("Longueur", fmt(r["length_cm"]) + " cm", "valeur saisie", "#E8C849"),
          ("Largeur", fmt(r["width_cm"]) + " cm", "valeur saisie", "#E17734"),
          ("Surface", fmt(area, 2) + " m²", "L × l", "#4777A3"),
          ("Volume", fmt(litres, 2) + " L", "L × l × p", "#278367")])
    with st.container(border=True):
        plot(dimensions_figure(r), "Schéma de la surface rectangulaire et de la coupe du volume équivalent", 310)
    st.markdown(f"**Surface :** {r['length_cm']:g} × {r['width_cm']:g} / 10 000 = **{fmt(area, 3)} m²**.\n\n**Volume :** {fmt(area, 3)} × {r['depth_cm']:g} / 100 = **{fmt(volume, 4)} m³**, soit **{fmt(litres, 2)} litres**.")
    st.html('<div class="ra-mini">Le modèle utilise un volume rectangulaire équivalent. Un nid-de-poule réel est irrégulier. Le décapage, la préparation et le compactage ne sont pas modélisés : ce volume ne constitue pas un devis de réparation.</div>')
    st.subheader("Que change une mesure approximative ?")
    st.caption("Chaque dimension suit une loi triangulaire symétrique : minimum = mesure × (1 − erreur %), mode = mesure, maximum = mesure × (1 + erreur %). Dimensions supposées indépendantes.")
    with st.form("simulation_form"):
        a, b, c = st.columns(3)
        with a:
            e1 = st.number_input("Erreur longueur (± %)", 0., 90., 10., step=1.)
        with b:
            e2 = st.number_input("Erreur largeur (± %)", 0., 90., 10., step=1.)
        with c:
            e3 = st.number_input("Erreur profondeur (± %)", 0., 90., 20., step=1.)
        a, b = st.columns(2)
        with a:
            threshold = st.number_input("Seuil de volume à dépasser (L)", 0., 1e9, float(litres), step=1.)
        with b:
            count = st.selectbox("Nombre de simulations", [1000, 10000, 50000], index=1)
        seed = st.number_input("Graine aléatoire (résultats reproductibles)", 0, 2_000_000_000, 42)
        run = st.form_submit_button("Simuler les volumes", type="primary", icon=":material/science:", width="stretch")
    signature = (r["id"], r["length_cm"], r["width_cm"], r["depth_cm"])
    if run:
        try:
            simulated = simulate_volume(*signature[1:], (e1, e2, e3), count, seed)
            st.session_state["ra_simulation"] = dict(signature=signature, values=simulated, errors=(e1, e2, e3), threshold=threshold, count=count, seed=seed)
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("ra_simulation")
    if result and result["signature"] == signature:
        values = result["values"]
        quantiles = np.quantile(values, [.025, .5, .975])
        probability = float(np.mean(values > result["threshold"]))
        kpis([("Volume moyen simulé", fmt(values.mean(), 2) + " L", "moyenne des tirages", "#4777A3"),
              ("Quantile 2,5 %", fmt(quantiles[0], 2) + " L", "borne basse simulée", "#E8C849"),
              ("Quantile 97,5 %", fmt(quantiles[2], 2) + " L", "borne haute simulée", "#E17734"),
              ("P(V > seuil)", fmt(100 * probability, 1) + " %", f"seuil {fmt(result['threshold'])} L", "#278367")])
        st.caption(f"Résultats simulés avec {result['count']:,} tirages, graine {result['seed']}, erreurs ±{result['errors']} %. Hypothèses du dernier calcul affiché ; resoumettre le formulaire après un changement.")
        fig = go.Figure(go.Histogram(x=values, nbinsx=45, marker_color="#BEA147", name="Volumes simulés"))
        fig.add_vline(x=result["threshold"], line_color="#D46E34", line_dash="dash")
        fig.update_layout(xaxis_title="Volume simulé (L)", yaxis_title="Nombre de tirages", showlegend=False)
        plot(fig, "Distribution simulée du volume. La ligne pointillée représente le seuil choisi.")
        se = math.sqrt(probability * (1 - probability) / len(values))
        st.caption(f"Quantile médian : {fmt(quantiles[1], 2)} L. Écart-type simulé : {fmt(values.std(ddof=1), 2)} L. Erreur-type approximative de la fréquence : {fmt(100 * se, 2)} point(s) de pourcentage (approximation peu informative aux fréquences 0 ou 100 %).")
        st.markdown("Cette probabilité concerne **le dépassement d'un volume sous les hypothèses choisies**. Elle ne mesure pas la probabilité d'un accident ni un risque structurel.")
        st.download_button("Exporter les volumes simulés (CSV)", csv_bytes(pd.DataFrame({"Volume simulé (L)": values})), "RouteAlert_simulations.csv", "text/csv")
    elif result:
        st.info("Les dimensions ou le signalement ont changé. Relancer la simulation pour afficher un résultat adapté.")
    with st.expander("Statistiques descriptives et sensibilité"):
        volumes = np.array([measurements(x)[2] for x in records])
        st.markdown(f"Sur **{len(records)} signalement(s)** : volume moyen **{fmt(volumes.mean(), 2)} L**, médiane **{fmt(np.median(volumes), 2)} L**.")
        if len(volumes) >= 2:
            st.caption(f"Écart-type d'échantillon : {fmt(volumes.std(ddof=1), 2)} L. Ces statistiques décrivent le registre, pas l'ensemble du réseau routier.")
        st.markdown(f"**Sensibilité :** augmenter uniquement la longueur, la largeur ou la profondeur de 10 % augmente le volume de 10 %, soit **+{fmt(litres * .1, 2)} L** pour cette fiche. Augmenter les trois de 10 % donne un facteur 1,1³ = 1,331, soit **+33,1 %**.")


def sample_csv():
    return (";".join(CSV_COLUMNS) + "\n" +
            f"Voie Test;Secteur fictif;3.87;11.51;{today().isoformat()};80;50;8;Élevée;Forte;Essai entièrement fictif\n").encode("utf-8-sig")


def exports_page():
    heading("TRANSMETTRE", "Rapports et échanges", "Les exports portent sur le registre actif indiqué dans le menu.")
    records = current_records()
    if records:
        frame = tabular(records)
        st.dataframe(frame.drop(columns=["Description", "Compte rendu"]), hide_index=True, width="stretch", alt="Tableau complet des signalements du registre actif")
        a, b, c = st.columns(3)
        with a:
            st.download_button("Rapport PDF", pdf_report(records, st.session_state["ra_mode"] == "Exemple fictif"), "RouteAlert_rapport.pdf", "application/pdf", icon=":material/picture_as_pdf:", width="stretch")
        with b:
            st.download_button("Tableau CSV", csv_bytes(frame), "RouteAlert_registre.csv", "text/csv", icon=":material/table:", width="stretch")
        with c:
            st.download_button("Tableau Excel", literal_excel(frame), "RouteAlert_registre.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:", width="stretch")
    backup = dict(application="RouteAlert", schema=1, version=VERSION, exported=stamp(), fictitious=st.session_state["ra_mode"] == "Exemple fictif", records=records)
    st.download_button("Sauvegarde complète avec photos (JSON)", json.dumps(backup, ensure_ascii=False, indent=2).encode("utf-8"), "RouteAlert_sauvegarde.json", "application/json", icon=":material/backup:")
    st.caption("PDF : synthèse et score par défaut. CSV/Excel : consultation du registre. JSON : restauration complète avec photos et historique.")
    st.subheader("Importer des signalements")
    st.download_button("Télécharger le modèle CSV à remplir", sample_csv(), "RouteAlert_modele_import.csv", "text/csv")
    st.caption("CSV UTF-8 ou XLSX, 2 Mo et 500 lignes maximum. Colonnes du modèle obligatoires. Les nouvelles lignes sont ajoutées au statut Signalé ; aucun dossier existant n'est écrasé. Les tableaux de consultation exportés ont un format différent du modèle d'import.")
    with st.form("table_import"):
        upload = st.file_uploader("Fichier CSV / Excel", type=["csv", "xlsx"], max_upload_size=2)
        submit = st.form_submit_button("Vérifier et ajouter les lignes", type="primary")
    if submit:
        try:
            if not upload:
                raise ValueError("Choisir un fichier avant de lancer l'import.")
            imported = import_table(upload.getvalue(), upload.name)
            current_signatures = {(r["route"].casefold(), r["latitude"], r["longitude"], r["reported"]) for r in records}
            if any((r["route"].casefold(), r["latitude"], r["longitude"], r["reported"]) in current_signatures for r in imported):
                raise ValueError("Une ligne correspond à un signalement déjà présent (route, coordonnées et date identiques). Aucun ajout effectué.")
            persist(records + imported)
            st.rerun()
        except (ValueError, sqlite3.Error, OSError) as exc:
            st.error(str(exc))
    with st.expander("Restaurer une sauvegarde JSON"):
        st.caption("La restauration ajoute des copies avec de nouveaux identifiants. Elle peut créer des doublons si les mêmes dossiers sont déjà présents.")
        with st.form("backup_import"):
            upload_json = st.file_uploader("Sauvegarde RouteAlert", type=["json"], max_upload_size=33)
            confirm = st.checkbox("Ajouter une copie des dossiers au registre actif")
            restore = st.form_submit_button("Restaurer les copies")
        if restore:
            try:
                if not confirm or not upload_json:
                    raise ValueError("Choisir une sauvegarde et cocher la confirmation d'ajout.")
                if st.session_state["ra_mode"] == "Mes données":
                    metadata = json.loads(upload_json.getvalue())
                    if isinstance(metadata, dict) and metadata.get("fictitious"):
                        raise ValueError("Cette sauvegarde est fictive. La restaurer dans l'espace Exemple fictif.")
                imported = import_backup(upload_json.getvalue())
                persist(records + imported)
                st.rerun()
            except (ValueError, sqlite3.Error, OSError) as exc:
                st.error(str(exc))


def guide_page():
    heading("PRENDRE EN MAIN", "Comprendre et tester", "Un parcours de démonstration que tu peux expliquer et vérifier toi-même.")
    st.markdown("**RouteAlert permet de localiser un nid-de-poule, de documenter son état et de suivre sa réparation.** La photo accompagne les dimensions saisies manuellement.")
    st.markdown("1. Dans le menu, cliquer **Charger l'exemple fictif**.\n2. Explorer le **Tableau de bord**, puis la **Carte**.\n3. Ouvrir **Registre et fiches** et choisir **DEMO-001**.\n4. Dans **Priorités et interventions**, suivre les dossiers.\n5. Tester les volumes dans **Calculs et incertitudes**.\n6. Télécharger le rapport dans **Rapports et échanges**.")
    cases = pd.DataFrame([
        ["Tableau de bord", "Charger l'exemple fictif", "6 signalements, 5 ouverts, 1 réparé, volume ouvert 218,30 L"],
        ["Fiche DEMO-001", "80 × 50 × 8 cm", "Surface 0,40 m² ; volume 0,032 m³ = 32 L"],
        ["Score DEMO-001", "Poids 40 / 35 / 25, exposition forte", "32 + 14 + 25 = 71 / 100"],
        ["Priorités", "Poids par défaut", "DEMO-003 : 98,6 ; DEMO-006 : 79,9 ; DEMO-001 : 71"],
        ["Intervention", "DEMO-002 : tenter Réparé", "Enregistrement refusé tant que le dossier n'est pas Vérifié"],
        ["Intervention", "DEMO-002 : Vérifié, puis Réparé aujourd'hui", "Statut Réparé et délai 5 jours ; le dossier quitte les ouverts"],
        ["Simulation DEMO-001", "Erreurs 0 / 0 / 0 %, seuil 32 L", "Tous les volumes = 32 L ; P(V > 32) = 0 %"],
        ["Import", "Télécharger puis importer le modèle", "Un nouveau signalement, volume 32 L ; un second import identique est refusé"],
        ["Exports", "PDF, CSV, Excel et JSON", "Fichiers lisibles ; sauvegarde JSON contenant les photos"],
    ], columns=["Outil", "Test fictif", "Résultat attendu"])
    st.dataframe(cases, hide_index=True, width="stretch", alt="Exemples de test et résultats attendus pour chaque outil")
    with st.expander("Formules et limites", expanded=True):
        st.markdown("**Conversions :** 100 cm = 1 m ; 1 m³ = 1 000 L.\n\n**Score :** 100 × [40 min(p/10, 1) + 35 min(A/1, 1) + 25 E] / 100, avec p en cm, A en m² et E = 0, 0,5 ou 1.\n\n**Priorité :** un outil de classement pédagogique ; il ne remplace pas la vérification du contexte routier.\n\n**Simulation :** incertitude hypothétique sur les mesures. Aucune prédiction d'accident.\n\n**Statistiques :** elles décrivent les signalements présents. Une absence de signalement ne prouve pas le bon état d'une route.")
    st.caption("Charger de nouveau l'exemple réinitialise ses données. Mes données est un registre distinct, sauvegardé sur la machine du serveur. Sur un hébergement distant, consulter les indications de persistance dans README.")


def load_demo():
    st.session_state["ra_demo_records"] = demo_records()
    st.session_state["ra_mode"] = "Exemple fictif"
    st.session_state["_ra_mode"] = "Exemple fictif"
    for key in ("ra_selected", "ra_simulation", "register_pick", "analysis_pick", "work_pick", "map_record"):
        st.session_state.pop(key, None)


def change_workspace():
    st.session_state["ra_mode"] = st.session_state["_ra_mode"]
    st.session_state.pop("ra_selected", None)
    st.session_state.pop("ra_simulation", None)


def main():
    st.set_page_config(page_title="RouteAlert · Observatoire routier", page_icon=":material/add_road:", layout="wide")
    st.html(CSS)
    st.session_state.setdefault("ra_mode", "Mes données")
    st.session_state.setdefault("ra_demo_records", [])
    try:
        st.session_state["ra_real"], st.session_state["ra_revision"] = db_load()
    except (ValueError, sqlite3.Error, OSError) as exc:
        st.error(f"Impossible d'ouvrir le registre : {exc}")
        st.stop()
    page_defs = [
        ("accueil", dashboard, "Tableau de bord", "dashboard"),
        ("carte", map_page, "Carte", "map"),
        ("signaler", new_page, "Nouveau signalement", "add_location_alt"),
        ("registre", register_page, "Registre et fiches", "photo_library"),
        ("interventions", interventions_page, "Priorités et interventions", "engineering"),
        ("analyses", analysis_page, "Calculs et incertitudes", "query_stats"),
        ("rapports", exports_page, "Rapports et échanges", "download"),
        ("guide", guide_page, "Comprendre et tester", "school"),
    ]
    pages = {slug: st.Page(function, title=title, icon=f":material/{icon}:", url_path=slug, default=slug == "accueil")
             for slug, function, title, icon in page_defs}
    st.session_state["ra_pages"] = pages
    navigation = st.navigation(list(pages.values()))
    with st.sidebar:
        st.html('<div class="ra-brand"><div class="ra-logo">R</div><div><strong>RouteAlert</strong><small>OBSERVER · AGIR</small></div></div>')
        st.session_state["_ra_mode"] = st.session_state["ra_mode"]
        st.selectbox("Espace de travail", ["Mes données", "Exemple fictif"], key="_ra_mode", on_change=change_workspace)
        st.button("Charger l'exemple fictif", icon=":material/science:", width="stretch", on_click=load_demo, key="load_demo")
        st.caption("RouteAlert 1.0 · Génie civil")
        st.caption("Registre local · photos et constats documentés")
    mode_label = "EXEMPLE FICTIF · tes données réelles restent dans Mes données" if st.session_state["ra_mode"] == "Exemple fictif" else "MES DONNÉES · registre enregistré sur le serveur"
    st.html(f'<div style="display:inline-block;background:#F7E9BB;color:#705A17;border-radius:7px;padding:7px 11px;margin-bottom:12px;font-size:12px;font-weight:700">{mode_label}</div>')
    if notice := st.session_state.pop("ra_notice", None):
        st.success(notice)
    navigation.run()


if __name__ == "__main__":
    main()
