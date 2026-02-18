import streamlit as st
import json
import re
import time
import uuid
import pandas as pd
from io import BytesIO
from copy import deepcopy

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

/* Optional: Sidebar Inputs ebenfalls weiß */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    color: white !important;
}

/* Optional: Radio Buttons & Labels */
[data-testid="stSidebar"] .stRadio label {
    color: white !important;
}

/* Deine bestehenden UI Styles bleiben erhalten */
.step-header {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.step-sub {
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
}

.criterion-card {
    background: #1a1d27;
    border: 1px solid #2e3347;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.8rem;
}

.tag {
    display: inline-block;
    background: #2e3347;
    color: #aab;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.75rem;
    margin-right: 4px;
}

.score-pill {
    display: inline-block;
    background: #1d4ed8;
    color: white;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
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

# ─────────────────────────────────────────────
# SIDEBAR – API KEY & NAV
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("##Automatisiertes Gemini Research Tool")
    st.markdown("---")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Hier den Key eingeben",
        help="Frag Yella",
    )
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        "Seite auswählen",
        ["① Unternehmen", "② Kriterien", "③ Kalibrierungsbeispiele", "④ Analyse durchführen"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    n_Unternehmen = len([c for c in st.session_state.Unternehmen_text.splitlines() if c.strip()])
    n_Kriterien  = len(st.session_state.Kriterien)
    n_results   = len(st.session_state.results)
    st.markdown(f"**{n_Unternehmen}** Unternehmen")
    st.markdown(f"**{n_Kriterien}** Kriterien")
    st.markdown(f"**{n_results}** Ergebnisse sind ready:)")


# ═══════════════════════════════════════════════════════
# HELPER FUNCTIONS  (defined after pages so Streamlit doesn't complain)
# ═══════════════════════════════════════════════════════

def build_prompt(company_name: str, Kriterien: list) -> str:
    """Dynamically build the analysis prompt from Kriterien config."""

    # Build Kriterien block
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
                Kriterien_block += f"  • {ex['company']}: Score {ex['score']} — {ex['reason']}\n"
        Kriterien_block += "\n---\n"

    # Build JSON schema example
    json_example_items = ""
    for c in Kriterien:
        json_example_items += f"""    {{
      "kategorie": "{c['category']}",
      "kriterium": "{c['name']}",
      "score": "1-{c['scale']}",
      "begruendung": "...",
      "quellen": ["URL1", "URL2"]
    }},
"""

    prompt = f"""<rolle>
Du bist ein unabhängiger, erfahrener Finanz- und Strategieberater.
Du arbeitest faktenbasiert, kritisch, vergleichend und nachvollziehbar.
</rolle>

<kontext>
Du bewertest Finanz- und FinTech-Unternehmen anhand öffentlich zugänglicher Informationen
(z. B. Unternehmenswebsites, Geschäftsberichte, Pressemitteilungen, regulatorische Veröffentlichungen).
Das aktuelle Jahr ist 2025.
</kontext>

<aufgabe>
Analysiere und bewerte das folgende Unternehmen anhand der definierten Kriterien.

Unternehmen: {company_name}

Führe dafür eine gezielte Web-Recherche durch.
Nutze ausschließlich öffentlich zugängliche, überprüfbare Quellen.
Wenn Informationen fehlen, veraltet oder nicht eindeutig belegbar sind, weise explizit darauf hin. Keine Spekulation.
</aufgabe>

<bewertungssystem>
Verwende für JEDE Teilkategorie die angegebene Likert-Skala (je Kriterium 3, 4 oder 5 Punkte).
Bewerte stets relativ zu anderen Finanz- und FinTech-Unternehmen, nicht absolut oder idealtypisch.
Alle Bewertungen müssen logisch aus den gefundenen Fakten ableitbar sein.
</bewertungssystem>

<kriterien>
{Kriterien_block}
</kriterien>

<ausgabeformat>
Gib die Antwort AUSSCHLIESSLICH als valides JSON zurück – kein Text außerhalb des JSON.

{{
  "unternehmen": "{company_name}",
  "bewertungen": [
{json_example_items.rstrip(',\n')}
  ],
  "hinweise_zur_datenlage": "Optionale Hinweise zu Datenlücken oder eingeschränkter Vergleichbarkeit."
}}
</ausgabeformat>

<finale_anweisung>
Arbeite strukturiert, konsistent und faktenbasiert.
Priorisiere Nachvollziehbarkeit, Vergleichbarkeit und Transparenz.
Keine Inhalte außerhalb des JSON ausgeben.
</finale_anweisung>
"""
    return prompt


def extract_json(text: str):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


def parse_response(data: dict, company: str, Kriterien: list) -> dict:
    """Flatten the JSON response into a row dict."""
    row = {"Unternehmen": company, "Status": "OK"}
    bewertungen = data.get("bewertungen", [])

    # Map by (category, name) for robust lookup
    lookup = {}
    for b in bewertungen:
        key = (b.get("kategorie", "").strip(), b.get("kriterium", "").strip())
        lookup[key] = b

    for c in Kriterien:
        col_base = f"{c['category']} › {c['name']}"
        b = lookup.get((c["category"], c["name"]), {})
        row[f"{col_base} | Score"]       = b.get("score", "")
        row[f"{col_base} | Begründung"]  = b.get("begruendung", "")
        row[f"{col_base} | Quellen"]     = "; ".join(b.get("quellen", []))

    row["Hinweise Datenlage"] = data.get("hinweise_zur_datenlage", "")
    return row


def results_to_df(results: list, Kriterien: list) -> pd.DataFrame:
    return pd.DataFrame(results)


def to_excel(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Benchmark")
    return buf.getvalue()


def run_analysis(api_key: str, Unternehmen: list, Kriterien: list):
    """Run the full benchmark analysis with progress tracking."""
    try:
        from google import genai as genai_client
        from google.genai import types
    except ImportError:
        st.error("google-genai package not installed. Run: pip install google-genai")
        return

    client = genai_client.Client(api_key=api_key)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    results = []
    progress_bar = st.progress(0)
    status_text  = st.empty()
    log_area     = st.empty()
    log_lines    = []

    for idx, company in enumerate(Unternehmen):
        pct = idx / len(Unternehmen)
        progress_bar.progress(pct)
        status_text.markdown(f"**Processing {idx+1}/{len(Unternehmen)}: {company}**")

        try:
            prompt   = build_prompt(company, Kriterien)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config,
            )

            # Collect grounding sources
            sources = []
            try:
                metadata = response.candidates[0].grounding_metadata
                if metadata and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if chunk.web:
                            sources.append(f"{chunk.web.title}: {chunk.web.uri}")
            except Exception:
                pass

            data = extract_json(response.text)
            if data and "bewertungen" in data:
                row = parse_response(data, company, Kriterien)
                row["_raw_json"]    = response.text[:2000]
                row["_all_sources"] = "\n".join(set(sources))
                results.append(row)
                log_lines.append(f"{company}")
            else:
                results.append({"Unternehmen": company, "Status": "Parse error – no JSON found"})
                log_lines.append(f"{company} — JSON parse failed")

        except Exception as e:
            results.append({"Unternehmen": company, "Status": f"Error: {str(e)}"})
            log_lines.append(f"{company} — {str(e)}")

        log_area.code("\n".join(log_lines[-15:]))
        time.sleep(3)  # rate limiting

    progress_bar.progress(1.0)
    status_text.markdown("**Run complete!**")
    st.session_state.results = results
    st.rerun()
    
# ═══════════════════════════════════════════════════════
# PAGE 1 – Unternehmen
# ═══════════════════════════════════════════════════════
if page == "① Unternehmen":
    st.markdown('<div class="step-header">① Company List</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">One company name per line. The tool will research each one in order.</div>', unsafe_allow_html=True)

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
        st.markdown(f"**{len(Unternehmen)} Unternehmen loaded**")
        st.markdown("---")
        for c in Unternehmen[:20]:
            st.markdown(f"• {c}")
        if len(Unternehmen) > 20:
            st.markdown(f"*…and {len(Unternehmen)-20} more*")
        st.markdown("---")
        if st.button("🔄 Reset to defaults"):
            st.session_state.Unternehmen_text = DEFAULT_Unternehmen
            st.rerun()

# ═══════════════════════════════════════════════════════
# PAGE 2 – Kriterien
# ═══════════════════════════════════════════════════════
elif page == "② Kriterien":
    st.markdown('<div class="step-header">② Scoring Kriterien</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Define what gets scored and how. Each criterion uses a Likert scale anchored at 1 (low) and N (high).</div>', unsafe_allow_html=True)

    # ── Existing Kriterien ──
    to_delete = None
    to_edit   = None

    for idx, crit in enumerate(st.session_state.Kriterien):
        with st.container():
            st.markdown(f"""
            <div class="criterion-card">
                <span class="tag">{crit['category']}</span>
                <span class="tag">{crit['scale']}-point</span>
                <strong style="font-size:1.05rem">&nbsp;{crit['name']}</strong>
                <p style="color:#aab;font-size:0.88rem;margin:0.4rem 0 0.2rem">{crit['description']}</p>
                <p style="color:#66f;font-size:0.82rem;margin:0">1 = {crit['anchor_low'][:80]}…</p>
                <p style="color:#6f6;font-size:0.82rem;margin:0">{crit['scale']} = {crit['anchor_high'][:80]}…</p>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 5])
            with btn_col1:
                if st.button("Edit", key=f"edit_{crit['id']}"):
                    st.session_state.editing_id = crit["id"]
                    st.rerun()
            with btn_col2:
                if st.button("Delete", key=f"del_{crit['id']}"):
                    to_delete = crit["id"]
            with btn_col3:
                n_ex = len(crit.get("examples", []))
                st.markdown(f"<span style='font-size:0.8rem;color:#888'>{n_ex} example(s)</span>", unsafe_allow_html=True)

    if to_delete:
        st.session_state.Kriterien = [c for c in st.session_state.Kriterien if c["id"] != to_delete]
        st.rerun()

    st.markdown("---")

    # ── Edit or Add Form ──
    editing = st.session_state.editing_id is not None
    adding  = st.session_state.adding_criterion

    if not editing and not adding:
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("➕ Add criterion"):
                st.session_state.adding_criterion = True
                st.rerun()
        with col2:
            if st.button("Reset all to defaults"):
                st.session_state.Kriterien = deepcopy(DEFAULT_Kriterien)
                st.rerun()

    else:
        # Populate existing values if editing
        existing = {}
        if editing:
            for c in st.session_state.Kriterien:
                if c["id"] == st.session_state.editing_id:
                    existing = c
                    break
            st.subheader("✏️ Edit Criterion")
        else:
            st.subheader("➕ New Criterion")

        with st.form("criterion_form"):
            f_category = st.text_input("Category",       value=existing.get("category", ""))
            f_name     = st.text_input("Criterion name", value=existing.get("name", ""))
            f_desc     = st.text_area("Description (shown to the model)",
                                      value=existing.get("description", ""), height=90)
            f_scale    = st.selectbox("Scale",  [3, 4, 5],
                                      index=[3,4,5].index(existing.get("scale", 4)))
            f_low      = st.text_area(f"Anchor for 1 (lowest)",
                                      value=existing.get("anchor_low",  ""), height=70)
            f_high     = st.text_area(f"Anchor for {f_scale} (highest)",
                                      value=existing.get("anchor_high", ""), height=70)

            save_col, cancel_col = st.columns([1, 1])
            with save_col:
                submitted = st.form_submit_button("Save")
            with cancel_col:
                cancelled = st.form_submit_button("✖ Cancel")

        if submitted:
            new_crit = {
                "id": existing.get("id") or str(uuid.uuid4()),
                "category":    f_category,
                "name":        f_name,
                "description": f_desc,
                "scale":       f_scale,
                "anchor_low":  f_low,
                "anchor_high": f_high,
                "examples":    existing.get("examples", []),
            }
            if editing:
                st.session_state.Kriterien = [
                    new_crit if c["id"] == existing["id"] else c
                    for c in st.session_state.Kriterien
                ]
            else:
                st.session_state.Kriterien.append(new_crit)
            st.session_state.editing_id      = None
            st.session_state.adding_criterion = False
            st.rerun()

        if cancelled:
            st.session_state.editing_id      = None
            st.session_state.adding_criterion = False
            st.rerun()

# ═══════════════════════════════════════════════════════
# PAGE 3 – Kalibrierungsbeispiele
# ═══════════════════════════════════════════════════════
elif page == "③ Kalibrierungsbeispiele":
    st.markdown('<div class="step-header">③ Kalibrierungsbeispiele</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">For each criterion, you can provide scored reference Unternehmen. The model will use these as calibration anchors.</div>', unsafe_allow_html=True)

    for crit_idx, crit in enumerate(st.session_state.Kriterien):
        with st.expander(f"**{crit['category']} › {crit['name']}** ({crit['scale']}-point scale)  —  {len(crit['examples'])} example(s)"):
            examples = crit["examples"]

            # Display + delete existing examples
            to_remove = None
            for ex_idx, ex in enumerate(examples):
                col_score, col_company, col_reason, col_del = st.columns([1, 2, 5, 1])
                with col_score:
                    st.markdown(f"<span class='score-pill'>{ex['score']}</span>", unsafe_allow_html=True)
                with col_company:
                    st.markdown(f"**{ex['company']}**")
                with col_reason:
                    st.markdown(f"<span style='color:#aab;font-size:0.88rem'>{ex['reason']}</span>", unsafe_allow_html=True)
                with col_del:
                    if st.button("✖", key=f"rm_ex_{crit['id']}_{ex_idx}"):
                        to_remove = ex_idx

            if to_remove is not None:
                st.session_state.Kriterien[crit_idx]["examples"].pop(to_remove)
                st.rerun()

            # Add new example
            st.markdown("**Add example**")
            with st.form(f"ex_form_{crit['id']}"):
                ex_col1, ex_col2, ex_col3 = st.columns([2, 1, 4])
                with ex_col1:
                    ex_company = st.text_input("Company", key=f"exc_{crit['id']}")
                with ex_col2:
                    ex_score   = st.selectbox("Score", list(range(1, crit["scale"] + 1)), key=f"exs_{crit['id']}")
                with ex_col3:
                    ex_reason  = st.text_input("Reasoning", key=f"exr_{crit['id']}")
                if st.form_submit_button("➕ Add"):
                    if ex_company.strip():
                        st.session_state.Kriterien[crit_idx]["examples"].append({
                            "company": ex_company.strip(),
                            "score":   ex_score,
                            "reason":  ex_reason.strip(),
                        })
                        st.rerun()

# ═══════════════════════════════════════════════════════
# PAGE 4 – Analyse durchführen
# ═══════════════════════════════════════════════════════
elif page == "④ Analyse durchführen":
    st.markdown('<div class="step-header">④ Analyse durchführen</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Review your configuration, then launch the benchmark run.</div>', unsafe_allow_html=True)

    Unternehmen = [c.strip() for c in st.session_state.Unternehmen_text.splitlines() if c.strip()]
    Kriterien  = st.session_state.Kriterien

    # ── Config summary ──
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Unternehmen", len(Unternehmen))
    col_b.metric("Kriterien",  len(Kriterien))
    est_mins = max(1, round(len(Unternehmen) * 8 / 60))
    col_c.metric("Est. runtime", f"~{est_mins} min")

    if not api_key:
        st.warning("Bitte Gemini API Key eingeben.")
        st.stop()
    if not Unternehmen:
        st.warning("Keine Unternehmen definiert.")
        st.stop()
    if not Kriterien:
        st.warning("Keine Kriterien definiert.")
        st.stop()

    # ── Prompt preview ──
    with st.expander("Preview prompt (first company)"):
        st.code(build_prompt(Unternehmen[0], Kriterien), language="markdown")

    st.markdown("---")

    # ── Run button ──
    if st.button("Start Benchmark Run", type="primary", use_container_width=True):
        run_analysis(api_key, Unternehmen, Kriterien)

    # ── Results ──
    if st.session_state.results:
        st.markdown("---")
        st.markdown("### Results")
        df = results_to_df(st.session_state.results, Kriterien)
        st.dataframe(df, use_container_width=True, height=400)

        # Download buttons
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ CSV herunterladen", csv_bytes, "benchmark_results.csv", "text/csv")
        with dl_col2:
            excel_bytes = to_excel(df)
            st.download_button("⬇️ Excel herunterladen", excel_bytes, "benchmark_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
