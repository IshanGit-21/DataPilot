"""Visual system for the Streamlit app: deep oxblood hero, warm paper body, sand accents,
liquid-glass controls and restrained motion. Kept as a Python string so app.py can
st.markdown() it without file-serving setup. Every animation is switched off under
prefers-reduced-motion."""

THEME_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500&family=Inter+Tight:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --wine-950: #231012;
  --wine-900: #301518;
  --wine-800: #3C1A1E;
  --wine-700: #5A2329;
  --wine-500: #8A3A40;
  --paper: #F8F5EF;
  --paper-2: #F1ECE2;
  --surface: #FCFAF6;
  --ink: #221719;
  --muted: #7A6E69;
  --line: #E4DDD0;
  --sand: #DCD1B9;
  --sand-deep: #C2B18C;
  --sand-ink: #4A3A1E;
  --accent: #7A2830;
  --accent-soft: #F1E3DF;
  --accent-2: #A98B52;
  --accent-2-soft: #F3EAD6;
  --ok: #4F7A5B;
  --danger: #B04A3F;
  --radius: 4px;
  --ease: cubic-bezier(.22,.9,.3,1);
  --glass-bg: linear-gradient(180deg, rgba(255,255,255,.72), rgba(255,255,255,.36));
  --glass-edge: inset 0 1px 0 rgba(255,255,255,.9), inset 0 -1px 0 rgba(60,26,30,.08);
}

html, body, [class*="css"] { font-family: 'Inter Tight', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp { background: var(--paper); color: var(--ink); overflow-x: hidden; }
[data-testid="stAppViewContainer"] { overflow-x: hidden; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stMainBlockContainer, .block-container { padding-top: 0 !important; max-width: 1120px; padding-bottom: 4rem; }

.mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.display { font-family: 'Inter Tight', sans-serif; letter-spacing: -0.01em; }
.serif { font-family: 'Fraunces', Georgia, serif; }

/* ------------------------------------------------------------ motion --- */
@keyframes dp-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes dp-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes dp-breathe { 0%,100% { transform: scaleY(.82); opacity: .55; } 50% { transform: scaleY(1.08); opacity: 1; } }
@keyframes dp-pulse { 0% { box-shadow: 0 0 0 0 rgba(169,139,82,.55); } 70% { box-shadow: 0 0 0 9px rgba(169,139,82,0); } 100% { box-shadow: 0 0 0 0 rgba(169,139,82,0); } }
@keyframes dp-draw { to { stroke-dashoffset: 0; } }
@keyframes dp-shimmer { from { background-position: -200% 0; } to { background-position: 200% 0; } }
@keyframes dp-spin { to { transform: rotate(360deg); } }
.element-container, [data-testid="stVerticalBlock"] > div { animation: dp-fade .5s var(--ease) both; }
.dp-rise { animation: dp-rise .7s var(--ease) both; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* -------------------------------------------------------------- hero ---- */
.dp-hero {
  position: relative; overflow: hidden; color: #F5EFE6;
  margin: -2rem calc(50% - 50vw) 0; width: 100vw; padding: 22px max(24px, calc(50vw - 560px)) 0;
  background: radial-gradient(1200px 500px at 85% -10%, #4A1E24 0%, transparent 60%), var(--wine-900);
}
.dp-hero-bars { position: absolute; inset: 0; pointer-events: none; opacity: .55; }
.dp-hero-bars svg { position: absolute; right: -2%; bottom: 0; height: 100%; width: 70%; }
.dp-hero-bars .bx { transform-box: fill-box; transform-origin: 50% 100%;
  animation: dp-breathe 7s ease-in-out infinite; animation-delay: calc(var(--i) * -.45s); }
.dp-hero-bars .bx rect { fill: rgba(176,120,112,.30); }
.dp-hero-bars .bx line { stroke: rgba(176,120,112,.35); stroke-width: 1.2; }
.dp-nav-bar { position: relative; display: flex; align-items: center; justify-content: space-between; z-index: 2; }
.dp-wordmark { font-family: 'Fraunces', Georgia, serif; font-size: 27px; font-weight: 400; letter-spacing: -.01em; color: #FBF6EC; display: flex; align-items: center; gap: 10px; }
.dp-wordmark .grid { display: grid; grid-template-columns: repeat(4, 5px); gap: 3px; }
.dp-wordmark .grid i { width: 5px; height: 5px; background: #FBF6EC; border-radius: 1px; display: block; opacity: .9; }
.dp-wordmark .grid i:nth-child(3n) { opacity: .45; }
.dp-glass-nav { display: flex; gap: 4px; padding: 5px; border-radius: 14px;
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12);
  backdrop-filter: blur(16px) saturate(160%); -webkit-backdrop-filter: blur(16px) saturate(160%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.18); }
.dp-glass-nav span { padding: 9px 16px; font-size: 14.5px; color: rgba(245,239,230,.78); border-radius: 10px; transition: background .3s var(--ease), color .3s; }
.dp-glass-nav span.done { color: var(--sand); }
.dp-glass-nav span.done::before { content: "✓ "; font-size: 12px; }
.dp-glass-nav span.active { background: #F8F3EA; color: var(--wine-900); font-weight: 500; box-shadow: 0 6px 16px -8px rgba(0,0,0,.6); }
.dp-glass-nav span { cursor: default; }
.dp-glass-nav span:hover { background: transparent; }
.dp-glass-nav span.active:hover { background: #F8F3EA; color: var(--wine-900); }
.dp-hero-grid { position: relative; z-index: 2; display: grid; grid-template-columns: 1.25fr .75fr; gap: 56px; align-items: end; padding: 74px 0 56px; }
.dp-h1 { font-weight: 500; letter-spacing: -.035em; font-size: clamp(40px, 5.6vw, 76px); line-height: .98; margin: 0 0 22px; color: #FBF6EC; animation: dp-rise .9s var(--ease) both; }
.dp-h1 em { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400; color: var(--sand); letter-spacing: -.02em; }
.dp-lede { font-size: 17px; line-height: 1.6; color: rgba(245,239,230,.72); max-width: 46ch; animation: dp-rise 1s .1s var(--ease) both; }
.dp-hero .dp-panel { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); color: #F5EFE6;
  backdrop-filter: blur(18px) saturate(150%); -webkit-backdrop-filter: blur(18px) saturate(150%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.16), 0 30px 60px -30px rgba(0,0,0,.6); border-radius: 14px; animation: dp-rise 1s .2s var(--ease) both; }
.dp-hero .dp-panel-head, .dp-hero .dp-row, .dp-hero .dp-foot { border-color: rgba(255,255,255,.10); }
.dp-hero .dp-row .k, .dp-hero .dp-foot, .dp-hero .dp-panel-head .tag { color: rgba(245,239,230,.58); }
.dp-hero .dp-row .v.hi { color: var(--sand); }
.dp-hero .dp-pill { background: rgba(220,209,185,.16); color: var(--sand); }
.dp-strip { position: relative; z-index: 2; display: flex; justify-content: space-between; align-items: center; gap: 24px;
  border-top: 1px solid rgba(255,255,255,.12); padding: 22px 0 26px; font-size: 14.5px; color: rgba(245,239,230,.68); }
.dp-strip .btns { display: flex; gap: 12px; }
.dp-btn { display: inline-block; padding: 13px 22px; font-size: 14.5px; font-weight: 500; border-radius: 4px; text-decoration: none;
  transition: transform .25s var(--ease), background .3s; }
.dp-btn.ghost { border: 1px solid rgba(255,255,255,.22); color: #F5EFE6; }
.dp-btn.fill { background: rgba(255,255,255,.12); color: #F5EFE6; }
.dp-btn:hover { transform: translateY(-2px); background: rgba(255,255,255,.2); }
@media (max-width: 860px) { .dp-hero-grid { grid-template-columns: 1fr; padding-top: 48px; } .dp-strip { flex-direction: column; align-items: flex-start; } .dp-glass-nav { display: none; } }

/* ----------------------------------------------------------- sections --- */
.dp-eyebrow { display: inline-flex; align-items: center; gap: 9px; font-size: 12.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink); margin-bottom: 14px; }
.dp-eyebrow::before { content: ""; width: 13px; height: 13px; border: 1.5px solid var(--sand-deep); border-radius: 3px; background: var(--paper-2); }
.dp-fact-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin: 56px 0 8px; }
.dp-fact { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px 22px 24px;
  transition: transform .35s var(--ease), box-shadow .35s var(--ease); animation: dp-rise .8s var(--ease) both; }
.dp-fact:nth-child(2) { animation-delay: .08s; } .dp-fact:nth-child(3) { animation-delay: .16s; }
.dp-fact:hover { transform: translateY(-4px); box-shadow: 0 18px 40px -24px rgba(60,26,30,.45); }
.dp-fact .num { font-family: 'Fraunces', Georgia, serif; font-size: 34px; color: var(--accent); display: block; margin-bottom: 8px; }
.dp-fact .label { font-size: 14.5px; color: var(--muted); line-height: 1.5; }
@media (max-width: 860px) { .dp-fact-row { grid-template-columns: 1fr; } }

.dp-section-head { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 6px; margin: 14px 0 18px; }
.dp-section-head h2 { font-family: 'Inter Tight', sans-serif; font-size: 30px; font-weight: 500; letter-spacing: -.02em; margin: 0; color: var(--ink); }
.dp-section-head .hint { font-size: 13.5px; color: var(--muted); }
hr { border: none; border-top: 1px solid var(--line); margin: 2.2rem 0 1.2rem; }

/* ------------------------------------------------------------- panels --- */
.dp-panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.dp-panel-head { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; }
.dp-panel-head .fname { font-size: 13.5px; font-weight: 500; }
.dp-panel-head .tag { font-family: 'IBM Plex Mono'; font-size: 11.5px; color: var(--muted); }
.dp-row { display: flex; justify-content: space-between; padding: 9px 16px; border-bottom: 1px solid var(--line); font-size: 14px; }
.dp-row:last-child { border-bottom: none; }
.dp-row .k { color: var(--muted); }
.dp-row .v { font-family: 'IBM Plex Mono'; font-size: 13px; }
.dp-row .v.hi { color: var(--accent); font-weight: 500; }
.dp-pill { display: inline-block; font-size: 11px; font-family: 'IBM Plex Mono'; padding: 3px 9px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); }
.dp-pill.warn { background: var(--accent-2-soft); color: var(--accent-2); }
.dp-foot { padding: 10px 16px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); }

.dp-pipe-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 20px; height: 100%;
  transition: transform .35s var(--ease), box-shadow .35s var(--ease), border-color .3s; }
.dp-pipe-card:hover { transform: translateY(-3px); box-shadow: 0 18px 38px -26px rgba(60,26,30,.5); border-color: var(--sand-deep); }
.dp-pipe-card .pipe-name { font-family: 'Fraunces', Georgia, serif; font-size: 20px; font-weight: 400; margin-bottom: 4px; }
.dp-pipe-card .pipe-desc { font-size: 13px; color: var(--muted); line-height: 1.5; margin-bottom: 14px; min-height: 38px; }
.dp-pipe-card .pipe-subhead { font-size: 11.5px; color: var(--muted); margin: 12px 0 6px; }
.dp-pipe-card ul.pipe-steps { margin: 0 0 10px; padding-left: 18px; }
.dp-pipe-card ul.pipe-steps li { font-family: 'IBM Plex Mono'; font-size: 12px; color: var(--ink); margin-bottom: 5px; line-height: 1.4; }
.dp-decision { border-top: 1px solid var(--line); padding-top: 10px; margin-top: 10px; }
.dp-decision .row { font-size: 12px; margin-bottom: 6px; line-height: 1.45; }
.dp-decision .row b { color: var(--muted); font-weight: 500; }

.dp-stage { border-top: 2px solid var(--line); padding-top: 12px; }
.dp-stage.done { border-top-color: var(--accent); }
.dp-stage.active { border-top-color: var(--accent-2); }
.dp-stage .n { font-family: 'IBM Plex Mono'; font-size: 11.5px; color: var(--muted); display: block; margin-bottom: 8px; }
.dp-stage.done .n, .dp-stage.active .n { color: var(--ink); }
.dp-stage .name { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
.dp-stage .desc { font-size: 12.5px; color: var(--muted); line-height: 1.45; min-height: 48px; }
.dp-stage .status { font-family: 'IBM Plex Mono'; font-size: 10.5px; color: var(--muted); }
.dp-stage.done .status { color: var(--accent); }
.dp-stage.active .status { color: var(--accent-2); }

/* -------------------------------------------------- live status panel --- */
.dp-live { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; margin: 10px 0 18px;
  box-shadow: 0 20px 44px -34px rgba(60,26,30,.5); animation: dp-rise .5s var(--ease) both; }
.dp-live .title { display: flex; align-items: center; gap: 10px; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
.dp-live .title .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent-2); animation: dp-pulse 1.6s infinite; }
.dp-live.finished .title .dot { background: var(--ok); animation: none; }
.dp-step { display: grid; grid-template-columns: 22px 1fr auto; gap: 12px; align-items: center; padding: 8px 0; border-top: 1px dashed var(--line); font-size: 14.5px; }
.dp-step:first-of-type { border-top: none; }
.dp-step .mark { width: 20px; height: 20px; border-radius: 50%; border: 1.5px solid var(--line); display: grid; place-items: center; }
.dp-step.pending { color: #B3A9A2; }
.dp-step.active .mark { border-color: var(--accent-2); border-top-color: transparent; animation: dp-spin .9s linear infinite; }
.dp-step.done .mark { border-color: var(--ok); background: var(--ok); }
.dp-step.done .mark svg { width: 12px; height: 12px; }
.dp-step.done .mark path { stroke: #fff; stroke-width: 2.4; fill: none; stroke-dasharray: 20; stroke-dashoffset: 20; animation: dp-draw .45s .05s var(--ease) forwards; }
.dp-step .detail { font-family: 'IBM Plex Mono'; font-size: 12.5px; color: var(--muted); text-align: right; }
.dp-step.done .detail { animation: dp-fade .6s both; }
.dp-bar { height: 6px; border-radius: 999px; background: var(--paper-2); overflow: hidden; margin: 10px 0 8px; }
.dp-bar > i { display: block; height: 100%; border-radius: 999px; transition: width .5s var(--ease);
  background: linear-gradient(90deg, var(--wine-700), var(--accent-2), var(--wine-700)); background-size: 200% 100%; animation: dp-shimmer 2.2s linear infinite; }
.dp-live.finished .dp-bar > i { animation: none; background: var(--ok); }
.dp-bar-meta { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono'; font-size: 12.5px; color: var(--muted); }
.dp-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.dp-chip { font-size: 12.5px; padding: 6px 12px; border-radius: 999px; background: var(--paper-2); border: 1px solid var(--line);
  animation: dp-rise .5s var(--ease) both; animation-delay: calc(var(--i) * .07s); }
.dp-chip b { font-weight: 500; color: var(--accent); }

.dp-live-pill { position: fixed; top: 16px; right: 20px; z-index: 9999; display: flex; align-items: center; gap: 10px;
  padding: 9px 16px 9px 13px; border-radius: 999px; font-size: 13px; color: #2A1B1D;
  background: var(--glass-bg); border: 1px solid rgba(255,255,255,.6);
  backdrop-filter: blur(16px) saturate(170%); -webkit-backdrop-filter: blur(16px) saturate(170%);
  box-shadow: var(--glass-edge), 0 14px 30px -16px rgba(30,10,12,.55); animation: dp-rise .5s var(--ease) both; max-width: 46vw; }
.dp-live-pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); flex: none; }
.dp-live-pill.busy .dot { background: var(--accent-2); animation: dp-pulse 1.4s infinite; }
.dp-live-pill .txt { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ----------------------------------------------- liquid glass controls --- */
.stTabs [data-baseweb="tab-list"] { position: relative; gap: 2px; padding: 4px; border-radius: 999px; width: fit-content; max-width: 100%;
  background: rgba(60,26,30,.06); border: 1px solid rgba(60,26,30,.09);
  backdrop-filter: blur(14px) saturate(150%); -webkit-backdrop-filter: blur(14px) saturate(150%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.7), 0 10px 26px -18px rgba(60,26,30,.5); margin-bottom: 14px; }
.stTabs [data-baseweb="tab"] { height: 36px; padding: 0 18px; border-radius: 999px; background: transparent; color: var(--muted);
  font-weight: 500; font-size: 14px; position: relative; z-index: 1; transition: color .3s var(--ease); }
.stTabs [data-baseweb="tab"]:hover { color: var(--ink); }
.stTabs [aria-selected="true"] { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { top: 0 !important; bottom: 0 !important; height: auto !important; border-radius: 999px !important;
  background: var(--glass-bg) !important; border: 1px solid rgba(255,255,255,.75);
  backdrop-filter: blur(12px) saturate(190%); -webkit-backdrop-filter: blur(12px) saturate(190%);
  box-shadow: var(--glass-edge), 0 6px 16px -6px rgba(60,26,30,.4);
  transition: left .5s cubic-bezier(.3,1.25,.4,1), width .5s cubic-bezier(.3,1.25,.4,1) !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 6px; animation: dp-rise .5s var(--ease) both; }

/* ----------------------------------------------------- run bar + mode --- */
.dp-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; box-shadow: 0 20px 44px -36px rgba(60,26,30,.5); }
.dp-sub { font-size: 13px; color: var(--muted); margin: 2px 0 10px; }
.dp-note { font-size: 12.5px; color: var(--accent-2); margin: 2px 0 6px; }
.dp-kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
.dp-kpi .l { font-size: 11px; color: var(--muted); letter-spacing: .04em; text-transform: uppercase; }
.dp-kpi .v { font-family: 'IBM Plex Mono', monospace; font-size: 20px; color: var(--ink); }
.dp-best { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; margin: 6px 0 14px; }
.dp-best .name { font-family: 'Fraunces', Georgia, serif; font-size: 30px; color: var(--ink); }
.dp-best .meta { font-size: 13.5px; color: var(--muted); }
[data-testid="stRadio"] div[role="radiogroup"] { gap: 4px; padding: 4px; width: fit-content; border-radius: 999px;
  background: rgba(60,26,30,.06); border: 1px solid rgba(60,26,30,.09); backdrop-filter: blur(12px); }
[data-testid="stRadio"] label { padding: 4px 14px; border-radius: 999px; margin: 0 !important; transition: background .3s var(--ease), box-shadow .3s; }
[data-testid="stRadio"] label:has(input:checked) { background: var(--glass-bg); box-shadow: var(--glass-edge), 0 6px 14px -8px rgba(60,26,30,.4); }
[data-testid="stRadio"] label > div:first-child { display: none; }
[data-testid="stTextArea"] textarea { font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; line-height: 1.55; border-radius: 10px; background: #FFFDF9; }
.stButton > button[kind="primary"] { background: var(--wine-800); }

/* ------------------------------------------------------------ widgets --- */
[data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; border: 1.5px dashed var(--sand-deep) !important; border-radius: 10px !important;
  transition: border-color .3s, background .3s, transform .3s var(--ease); }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; background: #FFFDF9 !important; transform: translateY(-1px); }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 500; padding: .55rem 1.2rem; border: 1px solid transparent;
  transition: transform .25s var(--ease), box-shadow .3s var(--ease), background .3s, opacity .2s; }
.stButton > button { background: var(--wine-800); color: #F8F2E8; box-shadow: inset 0 1px 0 rgba(255,255,255,.14), 0 10px 22px -14px rgba(48,21,24,.8); }
.stButton > button:hover { background: var(--wine-700); color: #fff; transform: translateY(-2px); box-shadow: inset 0 1px 0 rgba(255,255,255,.2), 0 16px 28px -14px rgba(48,21,24,.85); border-color: transparent; }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(.98); }
.stDownloadButton > button { background: var(--sand); color: var(--sand-ink); box-shadow: inset 0 1px 0 rgba(255,255,255,.5); }
.stDownloadButton > button:hover { background: #E4DAC4; color: var(--sand-ink); transform: translateY(-2px); border-color: transparent; box-shadow: 0 14px 26px -16px rgba(74,58,30,.7); }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stNumberInput"] input { border-radius: 10px !important; border-color: var(--line) !important; background: var(--surface) !important; transition: border-color .25s, box-shadow .25s; }
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"] > div:focus-within { border-color: var(--wine-500) !important; box-shadow: 0 0 0 3px rgba(122,40,48,.12) !important; }
[data-baseweb="tag"] { background: var(--accent-soft) !important; color: var(--accent) !important; border-radius: 999px !important; }
[data-testid="stSegmentedControl"] button, [data-testid="stPills"] button { border-radius: 999px; backdrop-filter: blur(10px); }
[data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--line); }
[data-testid="stDataFrame"] { font-family: 'IBM Plex Mono'; font-size: 12.5px; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; }

.dp-footer { position: relative; border-top: none; padding-top: 16px; margin-top: 40px; display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); }
.dp-footer::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 1px;
  background: linear-gradient(90deg, var(--line) 0%, var(--line) 35%, var(--sand-deep) 50%, var(--line) 65%, var(--line) 100%);
  background-size: 200% 100%; animation: dp-sweep 9s linear infinite; }
@keyframes dp-sweep { from { background-position: 200% 0; } to { background-position: -200% 0; } }
.dp-footer span:first-child { color: var(--accent); letter-spacing: .04em; }

/* extra motion: quiet, short, and off for reduced-motion users */
.dp-kpi { animation: dp-rise .6s var(--ease) both; transition: transform .3s var(--ease), box-shadow .3s var(--ease); }
[data-testid="stHorizontalBlock"] > div:nth-child(2) .dp-kpi { animation-delay: .06s; }
[data-testid="stHorizontalBlock"] > div:nth-child(3) .dp-kpi { animation-delay: .12s; }
[data-testid="stHorizontalBlock"] > div:nth-child(4) .dp-kpi { animation-delay: .18s; }
[data-testid="stHorizontalBlock"] > div:nth-child(5) .dp-kpi { animation-delay: .24s; }
.dp-kpi:hover { transform: translateY(-3px); box-shadow: 0 16px 30px -22px rgba(60,26,30,.5); }
@keyframes dp-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.dp-grow { transform-origin: left center; animation: dp-grow .9s var(--ease) both; }
.dp-best { animation: dp-rise .7s var(--ease) both; }

/* ------------------------------------------------ live run + summary --- */
@property --dp-s { syntax: '<integer>'; inherits: false; initial-value: 0; }
@property --dp-m { syntax: '<integer>'; inherits: false; initial-value: 0; }
@keyframes dp-sec { from { --dp-s: 0; } to { --dp-s: 60; } }
@keyframes dp-min { from { --dp-m: 0; } to { --dp-m: 600; } }
.dp-clock { font-family: 'IBM Plex Mono'; font-size: 13px; color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
.dp-clock i { font-style: normal; }
.dp-clock .s { animation: dp-sec 60s steps(60, end) infinite; animation-delay: calc(var(--e) * -1s); counter-reset: dps var(--dp-s); }
.dp-clock .s::before { content: counter(dps, decimal-leading-zero); }
.dp-clock .m { animation: dp-min 36000s steps(600, end) infinite; animation-delay: calc(var(--e) * -1s); counter-reset: dpm var(--dp-m); }
.dp-clock .m::before { content: counter(dpm); }

.dp-run { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 18px 22px 22px; margin: 12px 0 22px;
  box-shadow: 0 24px 50px -38px rgba(60,26,30,.55); }
.dp-run-head { display: flex; align-items: center; gap: 10px; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }
.dp-run-head .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent-2); animation: dp-pulse 1.6s infinite; }
.dp-run-body { display: grid; grid-template-columns: 240px 1fr; gap: 28px; }
@media (max-width: 760px) { .dp-run-body { grid-template-columns: 1fr; } }

.dp-nodes { position: relative; display: flex; flex-direction: column; gap: 4px; }
.dp-node { position: relative; display: grid; grid-template-columns: 22px 1fr; gap: 12px; padding: 9px 0 13px; font-size: 14.5px; transition: color .4s var(--ease); }
.dp-node::before { content: ""; position: absolute; left: 10px; top: 31px; bottom: -6px; width: 2px; background: var(--line); border-radius: 2px; }
.dp-node:last-child::before { display: none; }
.dp-node.done::before { background: var(--ok); }
.dp-node.active::before { background: linear-gradient(180deg, var(--accent-2), var(--line)); }
.dp-node.active::after { content: ""; position: absolute; left: 7px; top: 31px; width: 8px; height: 8px; border-radius: 50%; background: var(--accent-2);
  box-shadow: 0 0 10px 2px rgba(169,139,82,.6); animation: dp-travel 1.8s var(--ease) infinite; }
@keyframes dp-travel { 0% { transform: translateY(0); opacity: 0; } 20% { opacity: 1; } 100% { transform: translateY(34px); opacity: 0; } }
.dp-node .mark { width: 22px; height: 22px; border-radius: 50%; border: 1.5px solid var(--line); display: grid; place-items: center; background: var(--surface); z-index: 1; }
.dp-node.pending { color: #B3A9A2; }
.dp-node.active .mark { border-color: var(--accent-2); animation: dp-pulse 1.8s infinite; }
.dp-node.active .mark::after { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--accent-2); }
.dp-node.done .mark { border-color: var(--ok); background: var(--ok); }
.dp-node.done .mark svg { width: 12px; height: 12px; }
.dp-node.done .mark path { stroke: #fff; stroke-width: 2.4; fill: none; }
.dp-node .n { display: block; font-weight: 500; }
.dp-node .d { display: block; font-family: 'IBM Plex Mono'; font-size: 11.5px; color: var(--muted); margin-top: 2px; }

.dp-stage-card { background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; min-height: 230px; }
.dp-stage-card .why { font-size: 14px; color: var(--ink); line-height: 1.5; margin-bottom: 16px; max-width: 60ch; }

.dp-cols { display: flex; flex-wrap: wrap; gap: 8px; }
.dp-colchip { position: relative; overflow: hidden; font-family: 'IBM Plex Mono'; font-size: 12px; padding: 5px 10px; border-radius: 999px;
  border: 1px solid var(--line); background: var(--surface); color: var(--ink); }
.dp-colchip::after { content: ""; position: absolute; inset: 0; background: linear-gradient(100deg, transparent 30%, rgba(169,139,82,.38) 50%, transparent 70%);
  transform: translateX(-120%); animation: dp-scan 2.4s var(--ease) infinite; animation-delay: calc(var(--i) * .12s); }
@keyframes dp-scan { 0% { transform: translateX(-120%); } 55%, 100% { transform: translateX(120%); } }
.dp-colchip.number { border-color: #B9C9BE; } .dp-colchip.category { border-color: #D9C7A2; } .dp-colchip.text { border-color: #C8B7D0; }
.dp-colchip.skip { border-style: dashed; color: var(--muted); } .dp-colchip.target { background: var(--accent-soft); border-color: var(--accent); }
.dp-colchip.more { color: var(--muted); }
.dp-legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; font-size: 11.5px; color: var(--muted); }
.dp-legend span::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; border: 1.5px solid var(--line); vertical-align: -1px; }
.dp-legend .number::before { border-color: #8FAE9A; } .dp-legend .category::before { border-color: #C9AE72; } .dp-legend .text::before { border-color: #A98DB5; }
.dp-legend .skip::before { border-style: dashed; } .dp-legend .target::before { border-color: var(--accent); background: var(--accent-soft); }

.dp-preps { display: flex; flex-direction: column; gap: 12px; }
.dp-prep { display: grid; grid-template-columns: 170px 1fr 110px; gap: 14px; align-items: center; font-size: 13.5px; }
.dp-prep .bar { height: 9px; border-radius: 6px; background: var(--line); overflow: hidden; }
.dp-prep .bar i { display: block; height: 100%; background: var(--accent); border-radius: 6px; transform-origin: left; animation: dp-grow .9s var(--ease) both; }
.dp-prep .bar i.busy { width: 40%; background: linear-gradient(90deg, transparent, var(--accent-2), transparent); animation: dp-slide 1.3s ease-in-out infinite; }
@keyframes dp-slide { from { transform: translateX(-100%); } to { transform: translateX(260%); } }
.dp-prep .v { font-family: 'IBM Plex Mono'; font-size: 12px; color: var(--muted); text-align: right; }
.dp-note-line { margin-top: 14px; font-size: 12.5px; color: var(--muted); }

.dp-models { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.dp-model { font-size: 12.5px; padding: 5px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); }
.dp-model b { font-family: 'IBM Plex Mono'; font-weight: 400; font-size: 10.5px; margin-left: 7px; }
.dp-model.full { color: var(--ink); border-color: var(--ok); }
.dp-model.full::before { content: "\\2713 "; color: var(--ok); }
.dp-model.new { animation: dp-pop .5s var(--ease) both; }
@keyframes dp-pop { from { opacity: 0; transform: translateY(8px) scale(.94); } to { opacity: 1; transform: none; } }
.dp-bar { height: 8px; border-radius: 6px; background: var(--line); overflow: hidden; }
.dp-bar i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); border-radius: 6px; transition: width .6s var(--ease); }
.dp-bar-meta { display: flex; justify-content: space-between; margin-top: 8px; font-family: 'IBM Plex Mono'; font-size: 11.5px; color: var(--muted); }

.dp-tracks { display: flex; flex-direction: column; gap: 16px; max-width: 460px; }
.dp-track { display: grid; grid-template-columns: 130px 1fr; gap: 14px; align-items: center; font-size: 13px; color: var(--muted); }
.dp-track div { position: relative; height: 4px; border-radius: 4px; background: var(--line); }
.dp-track i { position: absolute; top: -6px; left: 0; width: 16px; height: 16px; border-radius: 50%; background: var(--surface); border: 2px solid var(--accent-2);
  animation: dp-knob var(--d) ease-in-out infinite alternate; }
@keyframes dp-knob { from { left: 4%; } to { left: calc(96% - 16px); } }

.dp-final { display: flex; align-items: center; gap: 18px; }
.dp-final .ring { width: 44px; height: 44px; border-radius: 50%; border: 3px solid var(--line); border-top-color: var(--accent); animation: dp-spin 1.1s linear infinite; }
.dp-final .n { font-size: 16px; font-weight: 500; } .dp-final .s { font-size: 12.5px; color: var(--muted); margin-top: 3px; }

.dp-sum { display: flex; flex-direction: column; gap: 10px; margin-top: 6px; }
.dp-sum-card { display: grid; grid-template-columns: 34px 1fr; gap: 14px; padding: 14px 18px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  animation: dp-rise .5s var(--ease) both; animation-delay: calc(var(--i) * .05s); }
.dp-sum-card .num { width: 28px; height: 28px; border-radius: 50%; background: var(--accent-soft); color: var(--accent); display: grid; place-items: center; font-family: 'IBM Plex Mono'; font-size: 12.5px; }
.dp-sum-card .t { font-weight: 500; font-size: 15px; margin-bottom: 3px; } .dp-sum-card .b { font-size: 13.5px; color: #4A3F3B; line-height: 1.55; }
.dp-sum-time { padding: 14px 18px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
.dp-sum-time .t { font-weight: 500; font-size: 15px; margin-bottom: 10px; }
.dp-sum-time .bar { display: flex; height: 10px; border-radius: 6px; overflow: hidden; gap: 2px; }
.dp-sum-time .bar i { display: block; transform-origin: left; animation: dp-grow .9s var(--ease) both; animation-delay: calc(var(--i) * .1s); }
.dp-sum-time .bar i:nth-child(1) { background: #8A3A40; } .dp-sum-time .bar i:nth-child(2) { background: #A98B52; }
.dp-sum-time .bar i:nth-child(3) { background: #6E8B74; } .dp-sum-time .bar i:nth-child(4) { background: #5A2329; }
.dp-sum-time .legend { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px; font-size: 12.5px; color: var(--muted); }
.dp-sum-time .legend b { font-weight: 500; color: var(--ink); margin-right: 5px; }
@media (prefers-reduced-motion: reduce) {
  .dp-colchip::after, .dp-node.active::after, .dp-prep .bar i.busy, .dp-track i, .dp-final .ring, .dp-model.new, .dp-sum-card { animation: none !important; }
}
</style>
"""
