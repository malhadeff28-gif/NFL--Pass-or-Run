"""NFL pre-snap RUN/PASS dashboard.

Start with::

    streamlit run app.py

Page flow, top to bottom:
    1. Enter the game situation
    2. Predict the play
    3. See the result
    4. Understand why the model called it that way
    5. Explore how the call changes with down and distance

Model logic lives in :mod:`src.predict`; this file is presentation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app.py` from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.charts import crossing_summary, factor_chart, sensitivity_chart  # noqa: E402
from src.config import MODEL_PATH  # noqa: E402
from src.predict import PlayCallPredictor  # noqa: E402
from src.theme import (  # noqa: E402
    build_css,
    get_theme,
    notice,
    prediction_card,
    section_header,
    situation_strip,
    stat_tile,
)

st.set_page_config(
    page_title="NFL Play-Call Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FORMATIONS = ["Unknown", "SHOTGUN", "SINGLEBACK", "I_FORM", "EMPTY",
              "PISTOL", "JUMBO", "WILDCAT"]
PERSONNEL = ["Unknown", "11", "12", "21", "10", "13", "22", "20", "01", "02"]
PERSONNEL_HELP = ("First digit = running backs, second = tight ends. "
                  "11 personnel means 1 RB, 1 TE, 3 WR.")
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> PlayCallPredictor:
    return PlayCallPredictor()


def init_state(predictor: PlayCallPredictor) -> None:
    teams = predictor.teams or ["KC"]
    defaults = {
        "theme_name": "night",          # night mode is the default
        "posteam": "SEA" if "SEA" in teams else teams[0],
        "defteam": "SF" if "SF" in teams else teams[min(1, len(teams) - 1)],
        "is_home": "Home",
        "season_type": "Regular season",
        "week": 10,
        "down": 3,
        "ydstogo": 8,
        "yardline_100": 45,
        "goal_to_go": False,
        "qtr": 4,
        "clock_min": 2,
        "clock_sec": 15,
        "score_off": 17,
        "score_def": 21,
        "pos_to": 3,
        "def_to": 3,
        "shotgun": True,
        "no_huddle": False,
        "formation": "Unknown",
        "personnel": "Unknown",
        "box_known": False,
        "defenders_in_box": 6,
        "motion_choice": "Unknown",
        "qb_out": 0,
        "rb_out": 0,
        "wr_out": 0,
        "te_out": 0,
        "result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def current_theme():
    return get_theme(st.session_state.get("theme_name", "night"))


# --------------------------------------------------------------------------
# Situation assembly
# --------------------------------------------------------------------------
def collect_kwargs() -> dict:
    """Turn widget state into predictor keyword arguments."""
    s = st.session_state
    formation = None if s["formation"] == "Unknown" else s["formation"]
    personnel = None if s["personnel"] == "Unknown" else s["personnel"]
    motion = {"Unknown": None, "Yes": True, "No": False}[s["motion_choice"]]
    return dict(
        posteam=s["posteam"],
        defteam=s["defteam"],
        down=s["down"],
        ydstogo=s["ydstogo"],
        yardline_100=s["yardline_100"],
        qtr=s["qtr"],
        quarter_seconds_remaining=s["clock_min"] * 60 + s["clock_sec"],
        score_differential=s["score_off"] - s["score_def"],
        week=s["week"],
        season_type="REG" if s["season_type"] == "Regular season" else "POST",
        posteam_timeouts_remaining=s["pos_to"],
        defteam_timeouts_remaining=s["def_to"],
        shotgun=s["shotgun"],
        no_huddle=s["no_huddle"],
        goal_to_go=s["goal_to_go"],
        is_home=s["is_home"] == "Home",
        offense_formation=formation,
        personnel_group=personnel,
        defenders_in_box=s["defenders_in_box"] if s["box_known"] else None,
        motion=motion,
        inj_qb_out=s["qb_out"],
        inj_rb_out=s["rb_out"],
        inj_wr_out=s["wr_out"],
        inj_te_out=s["te_out"],
    )


def situation_sentence() -> str:
    """Plain-English restatement of what is about to be predicted."""
    s = st.session_state
    diff = s["score_off"] - s["score_def"]
    if diff > 0:
        score = f"{s['posteam']} up {diff}"
    elif diff < 0:
        score = f"{s['posteam']} down {abs(diff)}"
    else:
        score = "tied"
    down_txt = ORDINAL.get(s["down"], str(s["down"]))
    dist = "goal" if s["goal_to_go"] else f"{s['ydstogo']}"
    qtr_txt = "OT" if s["qtr"] == 5 else f"Q{s['qtr']}"
    return (
        f"{s['posteam']} vs {s['defteam']}  •  {down_txt} & {dist} "
        f"from {s['yardline_100']} yards out  •  {qtr_txt} "
        f"{s['clock_min']}:{s['clock_sec']:02d}  •  {score}"
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar(predictor: PlayCallPredictor, t) -> None:
    bundle = predictor.bundle
    metrics = bundle.get("metrics", {})
    final = metrics.get("final", {})
    baseline = metrics.get("baseline_majority", {}).get("accuracy")
    split = bundle.get("split", {})
    acc = final.get("accuracy", float("nan"))

    with st.sidebar:
        st.html(section_header("Model card", "Performance",
                               "Measured on a held-out season the model "
                               "never saw during training."))
        c1, c2 = st.columns(2)
        with c1:
            st.html(stat_tile("Accuracy", f"{acc:.1%}",
                              (f"+{acc - baseline:.1%} vs baseline"
                               if baseline else ""), t.accent, t))
        with c2:
            st.html(stat_tile("ROC AUC", f"{final.get('roc_auc', float('nan')):.3f}",
                              "ranking quality", None, t))
        c3, c4 = st.columns(2)
        with c3:
            st.html(stat_tile("Brier", f"{final.get('brier', float('nan')):.3f}",
                              "lower is better", None, t))
        with c4:
            st.html(stat_tile("Baseline", f"{baseline:.1%}" if baseline else "-",
                              "always guess PASS", None, t))

        st.html(section_header("Model card", "Setup"))
        model_type = str(bundle.get("model_type", "n/a")).replace("_", " ").title()
        calibration = str(bundle.get("calibration", "n/a"))
        calibration = ("Well calibrated as-is" if calibration.startswith("none")
                       else calibration.title())
        train = split.get("train", ["?"])
        rows = [
            ("Algorithm", model_type),
            ("Calibration", calibration),
            ("Train seasons", f"{train[0]}-{train[-1]}"),
            ("Validation", str(split.get("validation", ["?"])[0])),
            ("Test season", str(split.get("test", ["?"])[0])),
            ("Last trained", str(bundle.get("trained_at", "n/a"))[:10]),
        ]
        st.html(
            '<div class="nfl-card" style="padding:12px 14px;">' + "".join(
                f'<div style="display:flex;justify-content:space-between;gap:10px;'
                f'padding:5px 0;border-bottom:1px solid {t.border};">'
                f'<span style="font-size:0.76rem;color:{t.text_muted};">{k}</span>'
                f'<span style="font-size:0.76rem;font-weight:650;'
                f'color:{t.text_primary};text-align:right;">{v}</span></div>'
                for k, v in rows
            ) + "</div>"
        )

        with st.expander("Optional features"):
            from src import vision

            ok, reason = vision.is_available()
            st.html(notice(
                "Photo analysis is on. Upload a pre-snap frame to fill the form."
                if ok else
                "Photo analysis is off. Set ANTHROPIC_API_KEY to enable reading "
                "a situation from a photo. Everything else works without it.",
                t, icon="i", tone="info" if ok else "muted",
            ))

        st.caption("A statistical model of historical tendencies. "
                   "Not a guarantee, and not for betting.")


# --------------------------------------------------------------------------
# Photo panel
# --------------------------------------------------------------------------
def render_photo_panel(t) -> None:
    from src import vision

    available, reason = vision.is_available()
    with st.expander("Read the situation from a photo (optional)", expanded=False):
        if not available:
            st.html(notice(reason, t, icon="i", tone="muted"))
            return

        st.caption("The scorebug gives down, distance, clock and score. "
                   "The field gives formation, personnel and the box count. "
                   "Anything not clearly visible is left as you set it.")
        upload = st.file_uploader("Pre-snap photo",
                                  type=["png", "jpg", "jpeg", "webp", "gif"],
                                  label_visibility="collapsed")
        if upload is None:
            return

        left, right = st.columns([2, 3])
        with left:
            st.image(upload, width="stretch")
        with right:
            if st.button("Read this photo", width="stretch"):
                with st.spinner("Reading the frame..."):
                    try:
                        st.session_state["photo_read"] = vision.analyze_image(
                            upload.getvalue(), upload.name)
                    except vision.VisionError as exc:
                        st.html(notice(str(exc), t, icon="!", tone="muted"))
                        return

            read = st.session_state.get("photo_read")
            if read is None:
                st.caption("Press **Read this photo** to extract the situation.")
                return

            st.html(notice(
                f"Reading confidence: {read.confidence}. "
                + (("Read: " + ", ".join(read.visible_fields))
                   if read.visible_fields else "Nothing could be read."),
                t, icon="i",
            ))
            if read.notes:
                st.html(notice(read.notes, t, icon="i", tone="muted"))
            if st.button("Apply to the form", type="primary", width="stretch"):
                if apply_photo_read(read):
                    st.rerun()


def apply_photo_read(read) -> bool:
    applied: list[str] = []

    def put(key, value, label):
        if value is not None:
            st.session_state[key] = value
            applied.append(label)

    put("down", read.down, "down")
    put("ydstogo", read.ydstogo, "distance")
    put("yardline_100", read.yardline_100, "field position")
    put("qtr", read.qtr, "quarter")
    put("shotgun", read.shotgun, "shotgun")
    put("no_huddle", read.no_huddle, "no huddle")
    put("score_off", read.score_offense, "offense score")
    put("score_def", read.score_defense, "defense score")

    if read.clock and ":" in str(read.clock):
        try:
            mm, ss = str(read.clock).split(":", 1)
            st.session_state["clock_min"] = max(0, min(15, int(mm)))
            st.session_state["clock_sec"] = max(0, min(59, int(float(ss))))
            applied.append("clock")
        except ValueError:
            pass
    if read.offense_formation in FORMATIONS:
        put("formation", read.offense_formation, "formation")
    if read.personnel_group in PERSONNEL:
        put("personnel", read.personnel_group, "personnel")
    if read.defenders_in_box is not None:
        st.session_state["defenders_in_box"] = read.defenders_in_box
        st.session_state["box_known"] = True
        applied.append("defenders in box")
    if read.motion is not None:
        st.session_state["motion_choice"] = "Yes" if read.motion else "No"
        applied.append("motion")

    predictor = load_predictor()
    for key, value, label in (("posteam", read.posteam, "offense"),
                              ("defteam", read.defteam, "defense")):
        if value and value in predictor.teams:
            st.session_state[key] = value
            applied.append(label)
    return bool(applied)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def render_inputs(predictor: PlayCallPredictor, t) -> None:
    teams = predictor.teams or ["KC"]

    st.html(section_header(
        "Step 1", "Enter the game situation",
        "Everything the model needs before the snap. Advanced pre-snap details "
        "are optional."))

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("**Matchup**")
        st.selectbox("Offense", teams, key="posteam")
        st.selectbox("Defense", teams, key="defteam")
        a, b = st.columns(2)
        a.selectbox("Side", ["Home", "Away"], key="is_home")
        b.selectbox("Season", ["Regular season", "Playoffs"], key="season_type")
        coach = predictor.coach_for(st.session_state["posteam"])
        st.caption(f"Head coach on record: {coach or 'unknown'}")

    with g2:
        st.markdown("**Down and field position**")
        a, b = st.columns(2)
        a.selectbox("Down", [1, 2, 3, 4], key="down")
        b.number_input("Yards to go", 1, 50, key="ydstogo")
        st.slider("Yards from opponent's end zone", 1, 99, key="yardline_100",
                  help="1 = about to score. 99 = backed up to your own goal line.")
        st.checkbox("Goal to go", key="goal_to_go")

    with g3:
        st.markdown("**Time and score**")
        a, b, c = st.columns([1.1, 1, 1])
        a.selectbox("Quarter", [1, 2, 3, 4, 5], key="qtr",
                    format_func=lambda q: "OT" if q == 5 else f"Q{q}")
        b.number_input("Min left", 0, 15, key="clock_min",
                       help="Minutes left in the quarter")
        c.number_input("Sec left", 0, 59, key="clock_sec",
                       help="Seconds left in the quarter")
        d, e = st.columns(2)
        d.number_input("Offense score", 0, 99, key="score_off")
        e.number_input("Defense score", 0, 99, key="score_def")
        f, g = st.columns(2)
        f.selectbox("Offense timeouts", [0, 1, 2, 3], key="pos_to")
        g.selectbox("Defense timeouts", [0, 1, 2, 3], key="def_to")

    with st.expander("Pre-snap look and injuries (optional)"):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.checkbox("Shotgun", key="shotgun")
            st.checkbox("No huddle", key="no_huddle")
            st.number_input("Week", 1, 22, key="week")
        with p2:
            st.selectbox("Formation", FORMATIONS, key="formation")
            st.selectbox("Personnel", PERSONNEL, key="personnel",
                         help=PERSONNEL_HELP)
        with p3:
            st.radio("Pre-snap motion", ["Unknown", "Yes", "No"],
                     horizontal=True, key="motion_choice")
            st.checkbox("I know the box count", key="box_known")
            st.slider("Defenders in the box", 3, 11, key="defenders_in_box",
                      disabled=not st.session_state["box_known"])
        with p4:
            st.markdown("**Offense ruled out**")
            q, r = st.columns(2)
            q.number_input("QB", 0, 3, key="qb_out")
            r.number_input("RB", 0, 5, key="rb_out")
            u, v = st.columns(2)
            u.number_input("WR", 0, 6, key="wr_out")
            v.number_input("TE", 0, 4, key="te_out")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def render_results(predictor: PlayCallPredictor, t) -> None:
    payload = st.session_state.get("result")
    if payload is None:
        st.html(notice(
            "Set the situation above, then press Predict the play call.",
            t, icon="i", tone="muted"))
        return

    result = payload["result"]
    kwargs = payload["kwargs"]

    st.html('<hr class="nfl-divider" />')
    st.html(section_header("Step 3", "The call",
                           payload["sentence"]))

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.html(prediction_card(result.prediction, result.pass_probability,
                                result.confidence, t))
        for text in result.warnings:
            st.html(notice(text, t, icon="i", tone="muted"))

    with right:
        st.html(section_header(
            "Step 4", "Why the model called it that way",
            "Bars to the right pushed the model toward PASS, bars to the left "
            "toward RUN. Longer means a stronger push."))
        st.plotly_chart(
            factor_chart(result.factors, t),
            width="stretch",
            config={"displayModeBar": False},
            key="factors",
        )

    if result.factors:
        st.html(
            '<div class="nfl-card">' + "".join(
                f'<div style="display:flex;gap:10px;padding:6px 0;'
                f'{"border-bottom:1px solid " + t.border if i < len(result.factors) else ""}">'
                f'<span style="flex:0 0 22px;height:22px;border-radius:6px;'
                f'background:{(t.pass_hue if f.contribution > 0 else t.run_hue)};'
                f'color:#fff;font-size:0.72rem;font-weight:800;display:flex;'
                f'align-items:center;justify-content:center;">{i}</span>'
                f'<span style="font-size:0.88rem;color:{t.text_secondary};'
                f'line-height:1.5;">{_esc(f.text)}</span></div>'
                for i, f in enumerate(result.factors, 1)
            ) + "</div>"
        )

    # --- Step 5: sensitivity -------------------------------------------
    st.html('<hr class="nfl-divider" />')
    st.html(section_header(
        "Step 5", "How down and distance affect pass probability",
        "Each line shows how the model's estimated pass probability changes as "
        "yards to go increase. Above the 50% line, the model predicts PASS."))

    with st.spinner("Sweeping every down and distance..."):
        grid = predictor.sensitivity_grid(kwargs)
    st.plotly_chart(
        sensitivity_chart(
            grid,
            current_down=kwargs["down"],
            current_distance=float(kwargs["ydstogo"]),
            t=t,
            current_probability=result.pass_probability,
        ),
        width="stretch",
        config={"displayModeBar": False},
        key="sensitivity",
    )
    summary = crossing_summary(grid, kwargs["down"])
    if summary:
        st.html(notice(summary, t, icon="i"))

    with st.expander("What the model actually saw"):
        s = result.situation
        st.markdown(
            f"- **Down and distance** {s['down']} & {s['ydstogo']:g}\n"
            f"- **Distance bucket** {s['distance_bucket']}\n"
            f"- **Field zone** {s['field_zone']} "
            f"({s['yardline_100']:g} yards out)\n"
            f"- **Score state** {s['score_state']} ({s['score_differential']:+g})\n"
            f"- **Game phase** {s['time_state']}\n"
            f"- **Seconds left in half / game** "
            f"{s['half_seconds_remaining']:.0f} / {s['game_seconds_remaining']:.0f}\n"
            f"- **Obvious passing situation** "
            f"{'yes' if s['is_obvious_pass'] else 'no'}"
        )
        st.markdown("**Historical tendencies applied** (each computed only from "
                    "plays that happened before this one):")
        rows = [(k, v) for k, v in s.items()
                if "pass_rate" in k and not k.endswith("_n")]
        for name, value in sorted(rows):
            n = s.get(f"{name}_n", 0)
            st.markdown(f"- `{name}` &mdash; **{value:.1%}** "
                        f"from {int(n):,} prior plays")


def _esc(value) -> str:
    from src.theme import esc

    return esc(value)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    if not MODEL_PATH.exists():
        st.error(
            "No trained model found. Run:\n\n"
            "```\npython -m src.download_data\npython -m src.clean_data\n"
            "python -m src.build_features\npython -m src.train\n```"
        )
        st.stop()

    predictor = load_predictor()
    init_state(predictor)
    t = current_theme()
    st.html(build_css(t))

    # --- Masthead + theme toggle ---------------------------------------
    head, toggle = st.columns([5, 1.15])
    with head:
        st.html(
            '<div style="margin-bottom:2px;">'
            f'<p class="nfl-eyebrow">Pre-snap analytics</p>'
            f'<h1 style="margin:0;font-size:2.3rem;">NFL Play-Call Predictor</h1>'
            f'<p class="nfl-sub" style="max-width:70ch;margin-top:4px;">'
            "Estimates whether the offense will run or pass on the next snap, "
            "using only information available before the ball is snapped."
            "</p></div>"
        )
    with toggle:
        st.write("")
        choice = st.segmented_control(
            "Theme", options=["Night", "Light"],
            default="Night" if t.is_dark else "Light",
            key="theme_toggle", label_visibility="collapsed",
        )
        wanted = "night" if (choice or "Night") == "Night" else "light"
        if wanted != st.session_state["theme_name"]:
            st.session_state["theme_name"] = wanted
            st.rerun()

    render_sidebar(predictor, t)
    st.html('<hr class="nfl-divider" style="margin-top:10px;" />')

    render_photo_panel(t)
    render_inputs(predictor, t)

    # --- Step 2: predict ------------------------------------------------
    st.html('<hr class="nfl-divider" />')
    st.html(section_header("Step 2", "Predict the play call"))
    st.html(situation_strip(situation_sentence(), t))

    if st.button("Predict the play call", type="primary", width="stretch"):
        kwargs = collect_kwargs()
        with st.spinner("Running the model..."):
            st.session_state["result"] = {
                "result": predictor.predict(**kwargs),
                "kwargs": kwargs,
                "sentence": situation_sentence(),
            }

    render_results(predictor, t)

    st.html('<hr class="nfl-divider" />')
    st.caption(
        "Probabilities are calibrated historical frequencies for situations like "
        "this one, not certainties about any single play. Not for gambling."
    )


if __name__ == "__main__":
    main()
