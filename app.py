import json
import re
import time
import uuid
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from copy import deepcopy
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Automatisiertes Gemini Research Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* Sidebar Hintergrundfarbe */
[data-testid="stSidebar"] {
    background-color: rgb(72, 99, 117);
}

/* Sidebar Schriftfarbe komplett weiß */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar Inputs: Kontrast (sonst weiß auf weiß) */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    color: #111827 !important;
    background-color: #f0f2f6 !important;
    border-radius: 5px;
}

/* Optional: Radio Buttons & Labels */
[data-testid="stSidebar"] .stRadio label {
    color: white !important;
}

/* Neutrale Criterion Card - minimalistisches Design */
.criterion-card {
    background: #f5f6f8;
    border: 1px solid #d8dce4;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}

.criterion-card:hover {
    border-color: #6a8ba8;
    box-shadow: 0 2px 6px rgba(106, 139, 168, 0.08);
}

.criterion-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.4rem;
}

.criterion-description {
    color: #5a6470;
    font-size: 0.92rem;
    margin-bottom: 0.8rem;
    line-height: 1.4;
}

.criterion-anchors {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #e2e6ed;
}

.anchor-box {
    padding: 0.7rem;
    border-radius: 6px;
    font-size: 0.85rem;
    line-height: 1.4;
    background: #ffffff;
    border: 1px solid #e2e6ed;
    color: #4a5568;
}

.tag {
    display: inline-block;
    background: #e8ecf4;
    color: #4a5568;
    border-radius: 4px;
    padding: 0.25rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

.step-header {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #1f2937;
}

.step-sub {
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Score Pill */
.score-pill {
    display: inline-block;
    background: #6a8ba8;
    color: white;
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Edit Form Container */
.edit-form-container {
    background: #f9fafb;
    border: 2px solid #6a8ba8;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEFAULT SEED DATA
# ─────────────────────────────────────────────
DEFAULT_Unternehmen = "\n".join([
    "N26", "Klarna", "Revolut", "Trade Republic", "Monzo",
    "nubank", "Sparkassen-Finanzgruppe", "Deutsche Bank",
    "Commerzbank", "ING Deutschland",
])

DEFAULT_Kriterien = [
    {
        "id": str(uuid.uuid4()),
        "category": "Geschäftsmodell",
        "name": "Wertschöpfungstiefe",
        "description": "Bewerte, in welchem Umfang die Wertschöpfung intern erfolgt vs. über Drittpartner.",
        "scale": 4,
        "anchor_low":  "Plattform-Fokus (hohe Fremdfertigung) → >80 % der Wertschöpfung über Drittanbieter",
        "anchor_high": "Eigenfertigungs-Fokus (volle Integration) → 0–10 % Fremdabwicklung, überwiegend eigene Bilanz/Infrastruktur",
        "examples": [],
    },
    {
        "id": str(uuid.uuid4()),
        "category": "Geschäftsmodell",
        "name": "Erlösbasis",
        "description": "Bewerte Struktur und Diversifikation der Ertragsquellen.",
        "scale": 4,
        "anchor_low":  "Klassisch → primär Zinsen, Provisionen, transaktionsbasierte Gebühren",
        "anchor_high": "Alternativ/erweitert → signifikante Erlöse aus Services, Abonnements, Plattform- oder SaaS-Modellen",
        "examples": [],
    },
    {
        "id": str(uuid.uuid4()),
        "category": "Markt- und Kundenzugang",
        "name": "Zielgruppen-Fokus",
        "description": "Bewerte die strategische Breite des Angebots.",
        "scale": 4,
        "anchor_low":  "Starke Segment-Spezialisierung → klar definierte Zielgruppe oder Use Cases",
        "anchor_high": "Vollumfängliches Finanzportfolio → breites Angebot für mehrere Zielgruppen/Lebenssituationen",
        "examples": [],
    },
    {
        "id": str(uuid.uuid4()),
        "category": "Markt- und Kundenzugang",
        "name": "Beziehungs-Hoheit",
        "description": "Bewerte die Rolle im direkten Kundenkontakt.",
        "scale": 4,
        "anchor_low":  "Abwicklung ohne Kundenschnittstelle → B2B-/White-Label-Rolle, kaum direkte Kundeninteraktion",
        "anchor_high": "Zentrale und erste Schnittstelle für jeglichen Finanzbedarf → täglicher Touchpoint, hohe Nutzungstiefe",
        "examples": [],
    },
    {
        "id": str(uuid.uuid4()),
        "category": "Operating Model",
        "name": "Innovations-Modus",
        "description": "Bewerte Organisations- und Entwicklungslogik.",
        "scale": 4,
        "anchor_low":  "Starr & manuell → Silo-Strukturen, lange Entscheidungswege, hoher manueller Anteil",
        "anchor_high": "Agil & produktgetrieben → cross-funktionale Teams, schnelle Iterationen, hohe Automatisierung",
        "examples": [],
    },
    {
        "id": str(uuid.uuid4()),
        "category": "Operating Model",
        "name": "Daten- & Technologie-Fundament",
        "description": "Bewerte den Reifegrad von Technologie, Daten & Analytics.",
        "scale": 4,
        "anchor_low":  "Wenig weit entwickelt → geringe Datenintegration, limitierte Analytics/AI-Nutzung",
        "anchor_high": "Hoch entwickelt → moderne API-Architektur, starke Analytics, systematischer AI-Einsatz",
        "examples": [],
    },
]

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "Kriterien" not in st.session_state:
    st.session_state.Kriterien = deepcopy(DEFAULT_Kriterien)
if "Unternehmen_text" not in st.session_state:
    st.session_state.Unternehmen_text = DEFAULT_Unternehmen
if "results" not in st.session_state:
    st.session_state.results = []
if "adding_criterion" not in st.session_state:
    st.session_state.adding_criterion = False
if "editing_id" not in st.session_state:
    st.session_state.editing_id = None
if "criteria_group_names" not in st.session_state:
    st.session_state.criteria_group_names = None

# Navigation State
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

PAGES = [
    "Überblick",
    "Unternehmen",
    "Kriterien",
    "Kalibrierungsbeispiele",
    "Analyse durchführen",
]

# ─────────────────────────────────────────────
# SIDEBAR – KONFIGURATION & STATUS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Automatisiertes Gemini Research Tool")

    # API Key input field for users
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Gib deinen API Key ein...",
        help="Erhalte deinen Key von https://aistudio.google.com/app/apikey"
    )
    
    # Fallback: Load from Streamlit secrets if available
    if not api_key:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    # Check if PAGES and session_state are initialized to prevent errors
    if 'page_index' in st.session_state and 'PAGES' in globals():
        page = PAGES[st.session_state.page_index]
        st.markdown("---") # Visual separator
        st.markdown(f"**Schritt {st.session_state.page_index + 1}/{len(PAGES)}**")
        st.markdown(f"*{page}*")
    else:
        # Fallback if the rest of your app logic hasn't loaded yet
        st.info("Initialisiere Tool...")

# ═══════════════════════════════════════════════════════
# HILFSFUNKTIONEN (RESEARCH & ANALYSE)
# ═══════════════════════════════════════════════════════

def build_prompt(company_name: str, Kriterien: list) -> str:
    """Erstellt den Analyse-Prompt basierend auf den konfigurierten Kriterien."""

    Kriterien_block = ""
    for i, c in enumerate(Kriterien, 1):
        Kriterien_block += f"""
{i}. {c['category'].upper()} – {c['name']}

{c['description']}

Skalenanker:
1 = {c['anchor_low']}
{c['scale']} = {c['anchor_high']}
"""
        if c.get("examples"):
            Kriterien_block += "\nKalibrierungsbeispiele:\n"
            for ex in c["examples"]:
                Kriterien_block += f"  - {ex['company']}: Score {ex['score']} — {ex['reason']}\n"
        Kriterien_block += "\n---\n"

    json_example_items = ""
    for c in Kriterien:
        json_example_items += f"""    {{
      "kategorie": "{c['category']}",
      "kriterium": "{c['name']}",
      "score": "1-{c['scale']}",
      "begruendung": "..."
    }},
"""

    json_example_items_clean = json_example_items.rstrip(",\n")

    prompt = f"""<rolle>
Du bist ein unabhängiger, erfahrener Finanz- und Strategieberater.
Du arbeitest faktenbasiert, kritisch, vergleichend und nachvollziehbar.
</rolle>

<kontext>
Du bewertest Finanz- und FinTech-Unternehmen anhand öffentlich zugänglicher Informationen für das Jahr 2025.
</kontext>

<aufgabe>
Analysiere und bewerte das folgende Unternehmen: {company_name}

Führe eine gezielte Web-Recherche durch. Nutze ausschließlich überprüfbare Quellen.
Wichtig: Schreibe die Begründungen in klaren, faktischen Sätzen. Vermeide vage Formulierungen, damit die Quellen eindeutig zugeordnet werden können.
</aufgabe>

<bewertungssystem>
Nutze die definierte Skala pro Kriterium. Bewerte relativ zum Marktumfeld.
</bewertungssystem>

<kriterien>
{Kriterien_block}
</kriterien>

<ausgabeformat>
Gib die Antwort AUSSCHLIESSLICH als valides JSON zurück.

{{
  "unternehmen": "{company_name}",
  "bewertungen": [
{json_example_items_clean}
  ],
  "hinweise_zur_datenlage": "Hinweise zu Datenlücken oder Vergleichbarkeit."
}}
</ausgabeformat>
"""
    return prompt


def extract_json(text: str):
    """Extrahiert JSON-Inhalte aus dem KI-Antworttext."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


def get_granular_sources(text_to_check: str, metadata) -> list:
    """
    Identifiziert spezifische URLs aus den Grounding-Metadaten, 
    die direkt mit dem übergebenen Textsegment verknüpft sind.
    """
    if not metadata or not hasattr(metadata, 'grounding_supports'):
        return []

    found_urls = []
    for support in metadata.grounding_supports:
        support_text = support.segment.text
        # Prüfung auf Überschneidung zwischen Begründungstext und Quell-Segment
        if support_text in text_to_check or text_to_check in support_text:
            for index in support.grounding_chunk_indices:
                if index < len(metadata.grounding_chunks):
                    chunk = metadata.grounding_chunks[index]
                    if chunk.web:
                        found_urls.append(chunk.web.uri)
    
    return sorted(list(set(found_urls)))


def parse_response(data: dict, company: str, Kriterien: list, metadata=None) -> dict:
    """Wandelt die JSON-Antwort und Metadaten in ein flaches Dictionary für den Export um."""
    row = {"Unternehmen": company, "Status": "OK"}
    bewertungen = data.get("bewertungen", [])

    lookup = {}
    for b in bewertungen:
        key = (b.get("kategorie", "").strip(), b.get("kriterium", "").strip())
        lookup[key] = b

    for c in Kriterien:
        col_base = f"{c['category']} - {c['name']}"
        b = lookup.get((c["category"], c["name"]), {})
        
        begruendung = b.get("begruendung", "")
        # Granulares Mapping der Quellen pro Kriterium
        quellen_liste = get_granular_sources(begruendung, metadata)
        
        row[f"{col_base} | Score"] = b.get("score", "")
        row[f"{col_base} | Begründung"] = begruendung
        row[f"{col_base} | Quellen"] = "\n".join(quellen_liste)

    row["Hinweise Datenlage"] = data.get("hinweise_zur_datenlage", "")
    return row


def results_to_df(results: list, Kriterien: list) -> pd.DataFrame:
    """Konvertiert die Ergebnisliste in ein Pandas DataFrame."""
    return pd.DataFrame(results)


def to_excel(df: pd.DataFrame) -> bytes:
    """Erzeugt einen Excel-Datenstrom aus dem DataFrame."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Benchmark")
    return buf.getvalue()


def analyze_criteria_correlation(df: pd.DataFrame, Kriterien: list) -> dict:
    """
    Analysiert die Korrelation zwischen Kriterien und gruppiert sie in 2 Cluster.
    
    Returns:
        dict mit:
        - 'correlation_matrix': DataFrame mit Korrelationsmatrix
        - 'group1': Liste der Kriterien in Gruppe 1
        - 'group2': Liste der Kriterien in Gruppe 2
        - 'group1_names': Liste der Kriterien-Namen in Gruppe 1
        - 'group2_names': Liste der Kriterien-Namen in Gruppe 2
        - 'score_df': DataFrame mit nur den Score-Spalten
    """
    # Extrahiere nur Score-Spalten
    score_cols = [col for col in df.columns if col.endswith(" | Score")]
    
    if len(score_cols) < 2:
        return None
    
    # Erstelle DataFrame mit nur Scores
    score_df = df[score_cols].copy()
    
    # Konvertiere zu numerisch (handle Fehler/Missing)
    for col in score_cols:
        score_df[col] = pd.to_numeric(score_df[col], errors='coerce')
    
    # Entferne Zeilen mit zu vielen fehlenden Werten
    score_df = score_df.dropna(axis=0, thresh=len(score_cols) * 0.5)
    
    if len(score_df) < 2:
        return None
    
    # Berechne Korrelationsmatrix
    corr_matrix = score_df.corr()
    
    # Konvertiere Korrelation zu Distanz (1 - correlation)
    # Für hierarchisches Clustering brauchen wir Distanzen
    distance_matrix = 1 - corr_matrix.abs()
    
    # Konvertiere zu condensed distance matrix (für linkage)
    # Diagonal entfernen und obere Dreiecksmatrix nehmen
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
    condensed_distances = distance_matrix.values[mask]
    
    # Hierarchisches Clustering mit Ward-Linkage
    # (alternativ: 'average', 'complete', 'single')
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Teile in 2 Cluster
    clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
    
    # Mappe Cluster-Zuordnung zu Kriterien-Namen
    criterion_names = [col.replace(" | Score", "") for col in score_cols]
    
    group1_indices = [i for i, c in enumerate(clusters) if c == 1]
    group2_indices = [i for i, c in enumerate(clusters) if c == 2]
    
    group1_names = [criterion_names[i] for i in group1_indices]
    group2_names = [criterion_names[i] for i in group2_indices]
    
    # Falls eine Gruppe leer ist (sollte nicht passieren), verteile gleichmäßig
    if not group1_names or not group2_names:
        mid = len(criterion_names) // 2
        group1_names = criterion_names[:mid]
        group2_names = criterion_names[mid:]
    
    return {
        'correlation_matrix': corr_matrix,
        'group1': group1_names,
        'group2': group2_names,
        'group1_indices': group1_indices,
        'group2_indices': group2_indices,
        'score_df': score_df,
        'criterion_names': criterion_names
    }


def create_pca_plot(analysis_result: dict, df: pd.DataFrame, group_names: dict) -> None:
    """
    Erstellt eine PCA-Visualisierung der Unternehmen entlang der beiden Gruppen.
    
    Args:
        analysis_result: Ergebnis von analyze_criteria_correlation
        df: DataFrame mit allen Ergebnissen (inkl. Unternehmen-Namen)
        group_names: Dict mit generierten Gruppennamen
    """
    score_df = analysis_result['score_df'].copy()
    
    # Hole Unternehmen-Namen (falls verfügbar)
    company_col = None
    if 'Unternehmen' in df.columns:
        # Filtere auf die gleichen Zeilen wie score_df
        valid_indices = score_df.index
        companies = df.loc[valid_indices, 'Unternehmen'].values
    else:
        companies = [f"Unternehmen {i+1}" for i in range(len(score_df))]
    
    # Berechne Durchschnittsscores pro Gruppe
    group1_cols = [col for i, col in enumerate(analysis_result['score_df'].columns) 
                   if i in analysis_result['group1_indices']]
    group2_cols = [col for i, col in enumerate(analysis_result['score_df'].columns) 
                   if i in analysis_result['group2_indices']]
    
    # Berechne Gruppendurchschnitte
    group1_scores = score_df[group1_cols].mean(axis=1) if group1_cols else pd.Series([0] * len(score_df))
    group2_scores = score_df[group2_cols].mean(axis=1) if group2_cols else pd.Series([0] * len(score_df))
    
    # Erstelle Plot-Daten
    plot_df = pd.DataFrame({
        'Unternehmen': companies,
        group_names.get('group1_name', 'Gruppe 1'): group1_scores.values,
        group_names.get('group2_name', 'Gruppe 2'): group2_scores.values
    })
    
    # Erstelle interaktiven Scatter-Plot
    fig = px.scatter(
        plot_df,
        x=group_names.get('group1_name', 'Gruppe 1'),
        y=group_names.get('group2_name', 'Gruppe 2'),
        text='Unternehmen',
        title='Unternehmens-Positionierung entlang der übergeordneten Kriterien',
        labels={
            group_names.get('group1_name', 'Gruppe 1'): f"{group_names.get('group1_name', 'Gruppe 1')} (Ø Score)",
            group_names.get('group2_name', 'Gruppe 2'): f"{group_names.get('group2_name', 'Gruppe 2')} (Ø Score)"
        },
        hover_data=['Unternehmen']
    )
    
    # Verbessere Darstellung
    fig.update_traces(
        textposition="top center",
        marker=dict(size=12, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        textfont=dict(size=10)
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Zusätzlich: PCA auf allen Kriterien für alternative Sicht
    st.markdown("---")
    st.markdown("### PCA-Analyse (Alternative Sicht)")
    st.markdown("*Principal Component Analysis auf allen Kriterien - zeigt die Hauptvarianz-Dimensionen*")
    
    # PCA mit 2 Komponenten
    pca = PCA(n_components=2)
    pca_scores = pca.fit_transform(score_df.fillna(score_df.mean()))
    
    # Erklärte Varianz
    explained_var = pca.explained_variance_ratio_
    
    pca_plot_df = pd.DataFrame({
        'Unternehmen': companies,
        'PC1': pca_scores[:, 0],
        'PC2': pca_scores[:, 1]
    })
    
    fig_pca = px.scatter(
        pca_plot_df,
        x='PC1',
        y='PC2',
        text='Unternehmen',
        title=f'PCA-Visualisierung (PC1: {explained_var[0]:.1%} Varianz, PC2: {explained_var[1]:.1%} Varianz)',
        labels={
            'PC1': f'PC1 ({explained_var[0]:.1%} Varianz)',
            'PC2': f'PC2 ({explained_var[1]:.1%} Varianz)'
        },
        hover_data=['Unternehmen']
    )
    
    fig_pca.update_traces(
        textposition="top center",
        marker=dict(size=12, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        textfont=dict(size=10)
    )
    
    fig_pca.update_layout(
        height=600,
        showlegend=False,
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16)
    )
    
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Zeige PCA-Ladungen (welche Kriterien tragen zu welchem PC bei)
    st.markdown("**PCA-Ladungen (Beitrag der Kriterien zu PC1 und PC2):**")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=analysis_result['criterion_names']
    )
    loadings_df = loadings_df.round(3)
    st.dataframe(loadings_df, use_container_width=True)


def generate_group_names(api_key: str, group1_criteria: list, group2_criteria: list, Kriterien: list) -> dict:
    """
    Generiert mit Gemini aussagekräftige Namen für die beiden Kriterien-Gruppen.
    
    Args:
        api_key: Gemini API Key
        group1_criteria: Liste der Kriterien-Namen in Gruppe 1
        group2_criteria: Liste der Kriterien-Namen in Gruppe 2
        Kriterien: Vollständige Liste aller Kriterien-Objekte mit Beschreibungen
    
    Returns:
        dict mit 'group1_name' und 'group2_name', oder None bei Fehler
    """
    # Erstelle Lookup für Kriterien-Beschreibungen
    criteria_lookup = {}
    for crit in Kriterien:
        full_name = f"{crit['category']} - {crit['name']}"
        criteria_lookup[full_name] = {
            'category': crit['category'],
            'name': crit['name'],
            'description': crit.get('description', '')
        }
    
    # Baue Beschreibungen für Gruppe 1
    group1_details = []
    for name in group1_criteria:
        crit_info = criteria_lookup.get(name, {})
        group1_details.append(f"- {name}: {crit_info.get('description', 'Keine Beschreibung')}")
    
    # Baue Beschreibungen für Gruppe 2
    group2_details = []
    for name in group2_criteria:
        crit_info = criteria_lookup.get(name, {})
        group2_details.append(f"- {name}: {crit_info.get('description', 'Keine Beschreibung')}")
    
    prompt = f"""<rolle>
Du bist ein Experte für Geschäftsanalyse und Strategie. Du hilfst dabei, komplexe Kriterien-Sets zu strukturieren und zu benennen.
</rolle>

<aufgabe>
Analysiere die folgenden zwei Gruppen von Bewertungskriterien für Finanz- und FinTech-Unternehmen. 
Diese Kriterien wurden durch Korrelationsanalyse automatisch gruppiert, weil sie ähnliche Scoring-Muster zeigen.

Gib für jede Gruppe einen prägnanten, aussagekräftigen Namen (2-4 Wörter), der das gemeinsame Thema oder die übergeordnete Dimension beschreibt, die diese Kriterien zusammenfassen.

**Gruppe 1:**
{chr(10).join(group1_details)}

**Gruppe 2:**
{chr(10).join(group2_details)}

</aufgabe>

<anforderungen>
- Die Namen sollten auf Deutsch sein
- Jeder Name sollte 2-4 Wörter lang sein
- Die Namen sollten klar unterscheidbar sein
- Verwende Fachbegriffe aus dem Finanz-/Strategiebereich, wenn passend
- Beispiele: "Operative Exzellenz", "Marktpositionierung", "Innovationsfähigkeit", "Kundenbeziehung"
</anforderungen>

<ausgabeformat>
Gib die Antwort AUSSCHLIESSLICH als valides JSON zurück:

{{
  "gruppe1_name": "Name für Gruppe 1",
  "gruppe2_name": "Name für Gruppe 2",
  "begruendung_gruppe1": "Kurze Begründung, warum dieser Name passt (1-2 Sätze)",
  "begruendung_gruppe2": "Kurze Begründung, warum dieser Name passt (1-2 Sätze)"
}}
</ausgabeformat>
"""
    
    try:
        from google import genai as genai_client
        from google.genai import types
        
        client = genai_client.Client(api_key=api_key)
        # Kein Google Search für diese Aufgabe nötig
        config = types.GenerateContentConfig()
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config,
        )
        
        # Extrahiere JSON aus der Antwort
        data = extract_json(response.text)
        
        if data and 'gruppe1_name' in data and 'gruppe2_name' in data:
            return {
                'group1_name': data.get('gruppe1_name', 'Gruppe 1'),
                'group2_name': data.get('gruppe2_name', 'Gruppe 2'),
                'group1_reason': data.get('begruendung_gruppe1', ''),
                'group2_reason': data.get('begruendung_gruppe2', '')
            }
        else:
            return None
            
    except Exception as e:
        # Bei Fehler gebe None zurück (UI zeigt dann Standard-Namen)
        return None


def run_analysis(api_key: str, Unternehmen: list, Kriterien: list):
    """Führt die vollständige Benchmark-Analyse mit Fortschrittsanzeige aus."""
    try:
        from google import genai as genai_client
        from google.genai import types
    except ImportError:
        st.error("Das Paket google-genai ist nicht installiert.")
        return

    client = genai_client.Client(api_key=api_key)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.empty()
    log_lines = []

    for idx, company in enumerate(Unternehmen):
        pct = (idx + 1) / len(Unternehmen)
        status_text.markdown(f"**Verarbeite {idx+1} von {len(Unternehmen)}: {company}**")

        try:
            prompt = build_prompt(company, Kriterien)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config,
            )

            metadata = response.candidates[0].grounding_metadata
            data = extract_json(response.text)

            if data and "bewertungen" in data:
                # Übergabe der Metadaten für das präzise Source-Mapping
                row = parse_response(data, company, Kriterien, metadata)
                results.append(row)
                log_lines.append(f"Erfolg: {company}")
            else:
                results.append({"Unternehmen": company, "Status": "Fehler beim Auslesen der Daten"})
                log_lines.append(f"Fehler: {company} (JSON konnte nicht gelesen werden)")

        except Exception as e:
            results.append({"Unternehmen": company, "Status": f"Systemfehler: {str(e)}"})
            log_lines.append(f"Fehler: {company} ({str(e)})")

        log_area.code("\n".join(log_lines[-10:]))
        progress_bar.progress(pct)
        time.sleep(2) # Kurze Pause zur Stabilisierung

    status_text.markdown("**Analyse vollständig abgeschlossen**")
    st.session_state.results = results
    st.rerun()


def render_navigation_bottom():
    """Zeigt Navigations-Buttons am Ende der Seite an."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col1:
        if st.session_state.page_index > 0:
            if st.button("Zurück", use_container_width=True):
                st.session_state.page_index -= 1
                st.rerun()

    with col3:
        if st.session_state.page_index < len(PAGES) - 1:
            if st.button("Weiter", use_container_width=True):
                st.session_state.page_index += 1
                st.rerun()

# ═══════════════════════════════════════════════════════
# PAGE 0 – ÜBERBLICK
# ═══════════════════════════════════════════════════════
page = PAGES[st.session_state.page_index]

if page == "Überblick":
    st.markdown("## Willkommen zum automatisierten Gemini Research Tool")

    st.markdown("""
    Diese Anwendung ermöglicht:

    1. Definition von Unternehmen
    2. Konfiguration individueller Bewertungskriterien
    3. Kalibrierung mittels Referenzbeispielen
    4. KI-gestützte Analyse mit Web-Recherche
    5. Export der Ergebnisse als CSV oder Excel

    Vorgehen:

    - Schritt 1: Unternehmen definieren
    - Schritt 2: Kriterien prüfen oder anpassen
    - Schritt 3: Optional Kalibrierungsbeispiele hinzufügen
    - Schritt 4: Analyse starten
    """)

    if st.button("Starten", use_container_width=True, type="primary"):
        st.session_state.page_index = 1
        st.rerun()

# ═══════════════════════════════════════════════════════
# PAGE 1 – UNTERNEHMEN
# ═══════════════════════════════════════════════════════
elif page == "Unternehmen":
    st.markdown('<div class="step-header">Unternehmen definieren</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Ein Unternehmensname pro Zeile. Das Tool recherchiert jedes nacheinander.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        raw = st.text_area(
            "Unternehmen",
            value=st.session_state.Unternehmen_text,
            height=420,
            label_visibility="collapsed",
            placeholder="N26\nKlarna\nRevolut\n...",
        )
        st.session_state.Unternehmen_text = raw

    with col_right:
        Unternehmen = [c.strip() for c in raw.splitlines() if c.strip()]
        st.markdown(f"**{len(Unternehmen)} Unternehmen geladen**")
        st.markdown("---")
        for c in Unternehmen[:20]:
            st.markdown(f"• {c}")
        if len(Unternehmen) > 20:
            st.markdown(f"*...und {len(Unternehmen)-20} weitere*")
        st.markdown("---")
        if st.button("Auf Standard zurücksetzen"):
            st.session_state.Unternehmen_text = DEFAULT_Unternehmen
            st.rerun()

    render_navigation_bottom()

# ═══════════════════════════════════════════════════════
# PAGE 2 – KRITERIEN
# ═══════════════════════════════════════════════════════
elif page == "Kriterien":
    st.markdown('<div class="step-header">Bewertungskriterien definieren</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Definiere, was bewertet wird und wie. Jedes Kriterium verwendet eine Likert-Skala von 1 (niedrig) bis N (hoch).</div>', unsafe_allow_html=True)

    to_delete = None

    for idx, crit in enumerate(st.session_state.Kriterien):
        # Zeige das Kriterium Card
        with st.container():
            st.markdown(f"""
            <div class="criterion-card">
                <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="tag">{crit['category']}</span>
                    <span class="tag">Skala: 1–{crit['scale']}</span>
                </div>
                <div class="criterion-title">{crit['name']}</div>
                <div class="criterion-description">{crit['description']}</div>
                <div class="criterion-anchors">
                    <div class="anchor-box">
                        <strong>Wert 1 (niedrig)</strong><br/>
                        {crit['anchor_low']}
                    </div>
                    <div class="anchor-box">
                        <strong>Wert {crit['scale']} (hoch)</strong><br/>
                        {crit['anchor_high']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 5])
            with btn_col1:
                if st.button("Bearbeiten", key=f"edit_{crit['id']}", use_container_width=True):
                    st.session_state.editing_id = crit["id"]
                    st.rerun()
            with btn_col2:
                if st.button("Löschen", key=f"del_{crit['id']}", use_container_width=True):
                    to_delete = crit["id"]
            with btn_col3:
                n_ex = len(crit.get("examples", []))
                ex_text = "Beispiel" if n_ex == 1 else "Beispiele"
                st.markdown(f"<span style='font-size:0.8rem;color:#888'>{n_ex} {ex_text}</span>", unsafe_allow_html=True)

        # Zeige Edit-Form direkt darunter, wenn dieses Kriterium bearbeitet wird
        if st.session_state.editing_id == crit["id"]:
            st.markdown('<div class="edit-form-container">', unsafe_allow_html=True)
            st.subheader("Kriterium bearbeiten")

            with st.form(f"criterion_form_{crit['id']}"):
                f_category = st.text_input("Kategorie", value=crit.get("category", ""))
                f_name     = st.text_input("Kriterium-Name", value=crit.get("name", ""))
                f_desc     = st.text_area("Beschreibung (wird dem Modell angezeigt)",
                                          value=crit.get("description", ""), height=90)
                f_scale    = st.selectbox("Skala", [3, 4, 5],
                                          index=[3,4,5].index(crit.get("scale", 4)),
                                          key=f"scale_{crit['id']}")
                f_low      = st.text_area(f"Anker für 1 (niedrigster Wert)",
                                          value=crit.get("anchor_low", ""), height=70)
                f_high     = st.text_area(f"Anker für {f_scale} (höchster Wert)",
                                          value=crit.get("anchor_high", ""), height=70)

                save_col, cancel_col = st.columns([1, 1])
                with save_col:
                    submitted = st.form_submit_button("Speichern")
                with cancel_col:
                    cancelled = st.form_submit_button("Abbrechen")

            if submitted:
                new_crit = {
                    "id": crit["id"],
                    "category": f_category,
                    "name": f_name,
                    "description": f_desc,
                    "scale": f_scale,
                    "anchor_low": f_low,
                    "anchor_high": f_high,
                    "examples": crit.get("examples", []),
                }
                st.session_state.Kriterien[idx] = new_crit
                st.session_state.editing_id = None
                st.rerun()

            if cancelled:
                st.session_state.editing_id = None
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

    if to_delete:
        st.session_state.Kriterien = [c for c in st.session_state.Kriterien if c["id"] != to_delete]
        st.rerun()

    # Buttons für Hinzufügen und Reset (nur wenn nicht editiert wird)
    if st.session_state.editing_id is None:
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Kriterium hinzufügen", use_container_width=True):
                st.session_state.adding_criterion = True
                st.rerun()
        with col2:
            if st.button("Alle auf Standard zurücksetzen", use_container_width=True):
                st.session_state.Kriterien = deepcopy(DEFAULT_Kriterien)
                st.rerun()

        # Add New Form (am Ende, wenn nichts editiert wird)
        if st.session_state.adding_criterion:
            st.markdown('<div class="edit-form-container">', unsafe_allow_html=True)
            st.subheader("Neues Kriterium")

            with st.form("criterion_form_new"):
                f_category = st.text_input("Kategorie", value="")
                f_name     = st.text_input("Kriterium-Name", value="")
                f_desc     = st.text_area("Beschreibung (wird dem Modell angezeigt)",
                                          value="", height=90)
                f_scale    = st.selectbox("Skala", [3, 4, 5], index=1)
                f_low      = st.text_area(f"Anker für 1 (niedrigster Wert)",
                                          value="", height=70)
                f_high     = st.text_area(f"Anker für {f_scale} (höchster Wert)",
                                          value="", height=70)

                save_col, cancel_col = st.columns([1, 1])
                with save_col:
                    submitted = st.form_submit_button("Speichern")
                with cancel_col:
                    cancelled = st.form_submit_button("Abbrechen")

            if submitted:
                new_crit = {
                    "id": str(uuid.uuid4()),
                    "category": f_category,
                    "name": f_name,
                    "description": f_desc,
                    "scale": f_scale,
                    "anchor_low": f_low,
                    "anchor_high": f_high,
                    "examples": [],
                }
                st.session_state.Kriterien.append(new_crit)
                st.session_state.adding_criterion = False
                st.rerun()

            if cancelled:
                st.session_state.adding_criterion = False
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    render_navigation_bottom()

# ═══════════════════════════════════════════════════════
# PAGE 3 – KALIBRIERUNGSBEISPIELE
# ═══════════════════════════════════════════════════════
elif page == "Kalibrierungsbeispiele":
    st.markdown('<div class="step-header">Kalibrierungsbeispiele</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Für jedes Kriterium kannst du bewertete Referenz-Unternehmen angeben. Das Modell nutzt diese als Ankerpunkte.</div>', unsafe_allow_html=True)

    for crit_idx, crit in enumerate(st.session_state.Kriterien):
        with st.expander(f"**{crit['category']} › {crit['name']}** (Skala 1–{crit['scale']})  —  {len(crit['examples'])} Beispiel(e)"):
            examples = crit["examples"]

            to_remove = None
            for ex_idx, ex in enumerate(examples):
                col_score, col_company, col_reason, col_del = st.columns([1, 2, 5, 1])
                with col_score:
                    st.markdown(f"<span class='score-pill'>{ex['score']}</span>", unsafe_allow_html=True)
                with col_company:
                    st.markdown(f"**{ex['company']}**")
                with col_reason:
                    st.markdown(f"<span style='color:#5a6470;font-size:0.88rem'>{ex['reason']}</span>", unsafe_allow_html=True)
                with col_del:
                    if st.button("Löschen", key=f"rm_ex_{crit['id']}_{ex_idx}", use_container_width=True):
                        to_remove = ex_idx

            if to_remove is not None:
                st.session_state.Kriterien[crit_idx]["examples"].pop(to_remove)
                st.rerun()

            st.markdown("**Beispiel hinzufügen**")
            with st.form(f"ex_form_{crit['id']}"):
                ex_col1, ex_col2, ex_col3 = st.columns([2, 1, 4])
                with ex_col1:
                    ex_company = st.text_input("Unternehmen", key=f"exc_{crit['id']}")
                with ex_col2:
                    ex_score   = st.selectbox("Wert", list(range(1, crit["scale"] + 1)), key=f"exs_{crit['id']}")
                with ex_col3:
                    ex_reason  = st.text_input("Begründung", key=f"exr_{crit['id']}")
                if st.form_submit_button("Hinzufügen"):
                    if ex_company.strip():
                        st.session_state.Kriterien[crit_idx]["examples"].append({
                            "company": ex_company.strip(),
                            "score":   ex_score,
                            "reason":  ex_reason.strip(),
                        })
                        st.rerun()

    render_navigation_bottom()

# ═══════════════════════════════════════════════════════
# PAGE 4 – ANALYSE DURCHFÜHREN
# ═══════════════════════════════════════════════════════
elif page == "Analyse durchführen":
    st.markdown('<div class="step-header">Analyse durchführen</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Überprüfe deine Konfiguration und starte die Benchmark-Analyse.</div>', unsafe_allow_html=True)

    Unternehmen = [c.strip() for c in st.session_state.Unternehmen_text.splitlines() if c.strip()]
    Kriterien  = st.session_state.Kriterien

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Unternehmen", len(Unternehmen))
    col_b.metric("Kriterien",  len(Kriterien))
    est_mins = max(1, round(len(Unternehmen) * 8 / 60))
    col_c.metric("Geschätzte Dauer", f"ca. {est_mins} Min.")

    if not api_key:
        st.warning("Bitte gib einen Gemini API Key ein.")
        st.stop()
    if not Unternehmen:
        st.warning("Keine Unternehmen definiert.")
        st.stop()
    if not Kriterien:
        st.warning("Keine Kriterien definiert.")
        st.stop()

    with st.expander("Prompt-Vorschau (erstes Unternehmen)"):
        st.code(build_prompt(Unternehmen[0], Kriterien), language="markdown")

    st.markdown("---")

    if st.button("Benchmark-Analyse starten", type="primary", use_container_width=True):
        run_analysis(api_key, Unternehmen, Kriterien)

    if st.session_state.results:
        st.markdown("---")
        st.markdown("### Ergebnisse")
        df = results_to_df(st.session_state.results, Kriterien)
        st.dataframe(df, use_container_width=True, height=400)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("CSV herunterladen", csv_bytes, "benchmark_results.csv", "text/csv", use_container_width=True)
        with dl_col2:
            excel_bytes = to_excel(df)
            st.download_button("Excel herunterladen", excel_bytes, "benchmark_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        # Kriterien-Korrelationsanalyse
        st.markdown("---")
        with st.expander("📊 Kriterien-Korrelationsanalyse", expanded=False):
            st.markdown("""
            **Gruppierung der Kriterien basierend auf Korrelation**
            
            Diese Analyse gruppiert die Kriterien in 2 Cluster, basierend darauf, 
            wie ähnlich die Scoring-Muster zwischen den Unternehmen sind.
            Stark korrelierte Kriterien werden in derselben Gruppe sein.
            
            **Übergeordnete Kriterien:** Mit Hilfe von KI werden für jede Gruppe 
            aussagekräftige Namen generiert, die das gemeinsame Thema beschreiben.
            """)
            
            analysis_result = analyze_criteria_correlation(df, Kriterien)
            
            if analysis_result is None:
                st.warning("Nicht genügend Daten für Korrelationsanalyse verfügbar. "
                          "Benötigt mindestens 2 Kriterien und 2 Unternehmen mit gültigen Scores.")
            else:
                # Generiere oder lade Gruppennamen
                # Erstelle einen Cache-Key basierend auf den aktuellen Gruppen
                group_key = f"{sorted(analysis_result['group1'])}|{sorted(analysis_result['group2'])}"
                
                # Prüfe ob wir bereits Namen für diese Gruppierung haben
                if (st.session_state.criteria_group_names is None or 
                    st.session_state.criteria_group_names.get('group_key') != group_key):
                    
                    # Generiere neue Namen mit Gemini
                    with st.spinner("Generiere aussagekräftige Namen für die Kriterien-Gruppen..."):
                        group_names = generate_group_names(
                            api_key, 
                            analysis_result['group1'], 
                            analysis_result['group2'],
                            Kriterien
                        )
                        
                        if group_names:
                            group_names['group_key'] = group_key
                            st.session_state.criteria_group_names = group_names
                        else:
                            # Fallback falls Gemini-Call fehlschlägt
                            st.session_state.criteria_group_names = {
                                'group_key': group_key,
                                'group1_name': 'Gruppe 1',
                                'group2_name': 'Gruppe 2',
                                'group1_reason': '',
                                'group2_reason': ''
                            }
                
                group_names = st.session_state.criteria_group_names
                
                # Button zum Neugenerieren der Namen
                if st.button("🔄 Gruppennamen neu generieren", use_container_width=False):
                    st.session_state.criteria_group_names = None
                    st.rerun()
                
                st.markdown("---")
                
                # Zeige die beiden Gruppen mit generierten Namen
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 🎯 {group_names.get('group1_name', 'Gruppe 1')}")
                    if group_names.get('group1_reason'):
                        st.caption(f"*{group_names['group1_reason']}*")
                    st.markdown(f"**{len(analysis_result['group1'])} Kriterien:**")
                    for name in analysis_result['group1']:
                        st.markdown(f"- {name}")
                
                with col2:
                    st.markdown(f"### 🎯 {group_names.get('group2_name', 'Gruppe 2')}")
                    if group_names.get('group2_reason'):
                        st.caption(f"*{group_names['group2_reason']}*")
                    st.markdown(f"**{len(analysis_result['group2'])} Kriterien:**")
                    for name in analysis_result['group2']:
                        st.markdown(f"- {name}")
                
                # Zeige Korrelationsmatrix als Heatmap
                st.markdown("---")
                st.markdown("### Korrelationsmatrix")
                st.markdown("*Werte nahe 1 = starke positive Korrelation, nahe -1 = starke negative Korrelation*")
                
                # Vereinfachte Darstellung der Korrelationsmatrix
                corr_df = analysis_result['correlation_matrix']
                # Runde auf 2 Dezimalstellen für bessere Lesbarkeit
                corr_display = corr_df.round(2)
                st.dataframe(corr_display, use_container_width=True)
                
                # Statistische Zusammenfassung
                st.markdown("---")
                st.markdown("### Statistische Zusammenfassung")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    # Durchschnittliche Korrelation innerhalb Gruppe 1
                    if len(analysis_result['group1_indices']) > 1:
                        group1_corr = corr_df.iloc[analysis_result['group1_indices'], analysis_result['group1_indices']]
                        avg_corr_1 = group1_corr.values[np.triu_indices_from(group1_corr.values, k=1)].mean()
                        st.metric("Ø Korrelation Gruppe 1", f"{avg_corr_1:.2f}")
                    else:
                        st.metric("Ø Korrelation Gruppe 1", "N/A")
                
                with summary_col2:
                    # Durchschnittliche Korrelation innerhalb Gruppe 2
                    if len(analysis_result['group2_indices']) > 1:
                        group2_corr = corr_df.iloc[analysis_result['group2_indices'], analysis_result['group2_indices']]
                        avg_corr_2 = group2_corr.values[np.triu_indices_from(group2_corr.values, k=1)].mean()
                        st.metric("Ø Korrelation Gruppe 2", f"{avg_corr_2:.2f}")
                    else:
                        st.metric("Ø Korrelation Gruppe 2", "N/A")
                
                with summary_col3:
                    # Durchschnittliche Korrelation zwischen den Gruppen
                    if len(analysis_result['group1_indices']) > 0 and len(analysis_result['group2_indices']) > 0:
                        between_corr = corr_df.iloc[analysis_result['group1_indices'], analysis_result['group2_indices']]
                        avg_between = between_corr.values.mean()
                        st.metric("Ø Korrelation zwischen Gruppen", f"{avg_between:.2f}")
                    else:
                        st.metric("Ø Korrelation zwischen Gruppen", "N/A")
                
                # PCA-Visualisierung
                st.markdown("---")
                st.markdown("### 📈 PCA-Visualisierung")
                st.markdown("""
                **Positionierung der Unternehmen entlang der übergeordneten Kriterien**
                
                Die Visualisierung zeigt, wie sich die Unternehmen in Bezug auf die beiden 
                Kriterien-Gruppen positionieren. Unternehmen mit ähnlichen Scores gruppieren sich zusammen.
                """)
                
                try:
                    create_pca_plot(analysis_result, df, group_names)
                except Exception as e:
                    st.error(f"Fehler bei der PCA-Visualisierung: {str(e)}")
                    st.info("Stelle sicher, dass genügend Datenpunkte vorhanden sind.")

    render_navigation_bottom()
