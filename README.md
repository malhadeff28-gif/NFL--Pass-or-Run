# 🏈 NFL Play-Call Predictor

Predicts whether an NFL offense will call a **RUN** or a **PASS** on any given
snap, and reports the probability of each outcome plus the factors that drove
that particular prediction.

Describe the situation two ways: **type it into the dashboard**, or **upload a
pre-snap photo** and let a vision model read the scorebug and the formation for
you (see [§8](#8-starting-the-dashboard)).

Real output — Kansas City on 3rd-and-8, trailing by 4, two minutes left:

```
Prediction: PASS
Pass likelihood: 95.9%
Run likelihood: 4.1%
Confidence: Very high

Main factors:
  1. Offense's pass rate in this down & distance (94%) strongly increases pass probability.
  2. 3rd-and-8 moderately increases pass probability.
  3. The offense is in shotgun, which moderately increases pass probability.
  4. This is an obvious passing situation, which moderately increases pass probability.
  5. Defenders in the box (unknown) moderately decreases pass probability.
```

And the mirror image — Baltimore at the goal line, up 10, clock to kill:

```
Prediction: RUN
Pass likelihood: 6.5%
Run likelihood: 93.5%
Confidence: Very high

Main factors:
  1. Defenders in the box (9) moderately decreases pass probability.
  2. Time left in game (5:00) moderately decreases pass probability.
  3. The offense is under center, which moderately decreases pass probability.
  4. Offense's pass rate in this down & distance (47%) slightly decreases pass probability.
  5. 1st-and-1 slightly decreases pass probability.
```

---

## Table of contents

1. [What the predictor does](#1-what-the-predictor-does)
2. [How RUN and PASS are defined](#2-how-run-and-pass-are-defined)
3. [Data source](#3-data-source)
4. [Installation](#4-installation)
5. [Downloading the data](#5-downloading-the-data)
6. [Training the model](#6-training-the-model)
7. [Running the evaluation](#7-running-the-evaluation)
8. [Starting the dashboard](#8-starting-the-dashboard)
9. [Making a CLI prediction](#9-making-a-cli-prediction)
10. [Model performance](#10-model-performance)
11. [Probability calibration](#11-probability-calibration)
12. [Known limitations](#12-known-limitations)
13. [Injury-data limitations](#13-injury-data-limitations)
14. [Coach-data limitations](#14-coach-data-limitations)
15. [Why you should not gamble on this](#15-why-you-should-not-gamble-on-this)

Also: [how data leakage is prevented](#how-data-leakage-is-prevented) ·
[photo-reading limitations](#photo-reading-limitations) ·
[project structure](#project-structure) · [features](#feature-list) ·
[tests](#tests)

---

## 1. What the predictor does

Given the situation **before the snap** — down, distance, field position, score,
clock, personnel, and who is playing — the model estimates the probability that
the offense's next play is a designed pass. You can describe that situation by
filling in the dashboard or by uploading a photo of the pre-snap frame.

It is a gradient-boosted tree classifier (XGBoost) trained on ten completed NFL
seasons of play-by-play data, and it returns four things:

1. A predicted play call: **RUN** or **PASS**
2. The estimated probability of a pass
3. The estimated probability of a run
4. A short explanation of the factors that most affected *this* prediction

The explanation is not a generic feature-importance chart. It uses exact tree
SHAP contributions for the single row being predicted, so it reflects why the
model reached its answer for that specific situation.

There are two interfaces: a Streamlit web app and a command-line tool.

---

## 2. How RUN and PASS are defined

The target is the **intended offensive play call**, not what the box score
eventually recorded. `PASS = 1`, `RUN = 0`.

### Counted as PASS

| Play | Why |
|---|---|
| Completed passes | Designed dropback |
| Incomplete passes | Designed dropback |
| Interceptions | Designed dropback |
| **Sacks** | The call was a pass; the protection broke down |
| **QB scrambles** | Began as a designed dropback, then the QB improvised |

Sacks and scrambles are the two cases that matter. Both look like non-passes in
a naive `play_type` filter, and both are genuinely *pass calls*. Scrambles are
especially deceptive: nflverse records them with `play_type == "run"`.

### Counted as RUN

Designed rushing attempts — handoffs, QB draws, designed QB keepers, end-arounds.

### Excluded entirely

Punts · field goals · extra points · kickoffs · kneel-downs · spikes ·
penalty-only plays with no snap (`play_type == "no_play"`) · timeouts and
end-of-period markers · two-point attempts · aborted plays · any play with no
offensive team, an invalid down, or an impossible distance.

Two-point attempts are excluded because they cannot be classified consistently
against the same down-and-distance framework as normal scrimmage plays.

The implementation keys off nflfastR's `pass` and `rush` indicators, which are
derived from the play description and already handle sacks and scrambles
correctly. `python -m src.clean_data` prints a full audit:

```
  Raw plays read:               484,254

  Excluded:
    - no offensive team on play                               25,904
    - missing play_type (timeouts, end-of-period markers)      8,569
    - punt/FG/XP/kickoff/kneel/spike/penalty-only            104,968
    - two-point attempt (cannot be classified consistently)     1,267
    - aborted play (no clear call intent)                      1,044
    TOTAL EXCLUDED                                           141,752

  Included (modelled) plays:    342,502
    - labeled PASS (1):         209,976  (61.3%)
    - labeled RUN  (0):         132,526  (38.7%)
```

---

## 3. Data source

All data comes from **[nflverse](https://github.com/nflverse)**, which is free
and requires no API key. The project reads it through
**[`nflreadpy`](https://github.com/nflverse/nflreadpy)**, and falls back to
direct Parquet downloads from the
[`nflverse-data`](https://github.com/nflverse/nflverse-data) GitHub releases if
`nflreadpy` is unavailable or a call fails.

> The deprecated `nfl_data_py` package is deliberately **not** used.

| Dataset | Seasons available | What it provides | Used? |
|---|---|---|---|
| `pbp` (play-by-play) | 2016–2025 | The core: situation, clock, score, shotgun, no-huddle | ✅ |
| `schedules` | all | Game context and **head coach names** (`home_coach` / `away_coach`) | ✅ |
| `participation` | 2016–2025 | Offensive formation, personnel groupings, defenders in the box | ✅ (99.9% join) |
| `ftn_charting` | 2022–2025 | Pre-snap **motion**, QB alignment, backfield count | ✅ (40.8% of plays) |
| `injuries` | 2016–2025 | Weekly injury and practice reports | ✅ (99.6% join) |
| `depth_charts` | 2016–2025 | Weekly depth charts | ❌ — see below |
| `weekly_rosters` | 2016–2025 | Weekly rosters | ❌ — disabled by default |

**Season selection is dynamic.** The pipeline does not assume the current
season is complete. It reads the nflverse schedule and looks for a Super Bowl
(`game_type == "SB"`) that has a recorded score; the newest season satisfying
that is the latest completed season. A calendar heuristic is used only if the
schedule cannot be read. Change the span in `config.yaml`:

```yaml
data:
  n_seasons: 10            # roughly the last 8-12 completed seasons
  seasons: null            # or pin an explicit list, e.g. [2018, 2019, 2020]
  latest_completed_season: null   # null = auto-detect
```

### Why depth charts are not used

nflverse changed the depth-chart schema for 2025. Seasons 2016–2024 provide
`season / week / club_code / depth_team`, but the 2025 file replaces these with
`dt / team / pos_grp / pos_rank` and **has no `week` column at all**. Depth
charts therefore cannot be joined per-week for the test season, which is
precisely the season the model is judged on. Rather than build a feature that is
present in training and absent in evaluation — or fabricate a week mapping —
the dataset is skipped and the limitation is reported.

Raw downloads are cached under `data/raw/` and are never re-fetched unless you
pass `--force`. Processed artifacts live separately under `data/processed/`, so
you can delete and rebuild them without re-downloading ~233 MB.

---

## 4. Installation

Requires **Python 3.10+** (developed and tested on 3.12).

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac or Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

Stack: pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Joblib · Streamlit.

---

## 5. Downloading the data

```bash
python -m src.download_data
```

This downloads roughly 233 MB across 46 files and takes a few minutes on a
first run. Subsequent runs are instant because everything is cached.

- `--force` re-downloads even when cached
- `--seasons 2020 2021 2022` overrides the configured season list

Failures are reported per season rather than silently swallowed, and a machine-
readable inventory of exactly what was and was not available is written to
`reports/data_inventory.json`.

Then build the labeled play set and the feature matrix:

```bash
python -m src.clean_data      # applies the RUN/PASS labels, prints the audit
python -m src.build_features  # engineers features, prints the leakage audit
```

---

## 6. Model training

```bash
python -m src.train
```

### Chronological splitting

Plays are **never** split randomly. A random split would put snaps from the same
drive on both sides of the divide and produce meaningless, inflated scores. The
split simulates predicting future football from past football:

| Split | Seasons | Purpose |
|---|---|---|
| Train | 2016–2023 | Fitting |
| Validation | 2024 | Hyper-parameter search, model selection, calibration |
| Test | 2025 | Held out; scored once at the very end |

The boundaries are derived from whatever data is actually present, so the split
shifts automatically as new seasons complete.

### Models trained

1. **Majority-class baseline** — always predicts PASS
2. **Logistic regression** — median imputation + standardisation + one-hot
3. **XGBoost** (`XGBClassifier`) — the intended final model

XGBoost is promoted to final model **only if it beats logistic regression on the
validation season**, never on the basis of test performance.

Hyper-parameters are tuned by randomised search over `n_estimators`,
`max_depth`, `learning_rate`, `min_child_weight`, `subsample`,
`colsample_bytree`, `reg_alpha`, `reg_lambda` and `gamma`, each candidate fit
with early stopping and scored on the validation season. `RandomizedSearchCV` is
deliberately avoided: k-fold cross-validation shuffles plays across time and
would reintroduce exactly the leakage the chronological split removes.

Preprocessing lives in a scikit-learn `ColumnTransformer` and is fitted on
training data only. The complete preprocessing + model (+ calibration) pipeline
is saved as one object to `models/play_call_model.joblib`.

---

## 7. Running the evaluation

```bash
python -m src.evaluate
```

Prints headline test-season metrics, a calibration table, and accuracy broken
down by **season, team, down, distance bucket, quarter, score state, field-
position zone, and prediction confidence**. Writes `reports/evaluation.json`
and these figures to `reports/figures/`:

`calibration.png` · `roc_curve.png` · `confusion_matrix.png` ·
`probability_distribution.png` · `accuracy_by_down.png` ·
`accuracy_by_distance.png` · `accuracy_by_score_state.png` ·
`feature_importance.png`

---

## 8. Starting the dashboard

```bash
streamlit run app.py
```

There are two ways to describe a play.

### a. Type the situation in

Every input the model uses is exposed: teams, down, distance, field position,
quarter, clock, score, timeouts, home/away, week, shotgun, no-huddle, formation,
personnel grouping, pre-snap motion, defenders in the box, and the offensive
injury report. Everything derived — clock splits, buckets, and all team / coach /
defensive tendencies — is computed automatically.

### b. Upload a pre-snap photo

Open **Read the situation from a photo**, drop in a frame, and press **Read this
photo**. Two things get extracted:

| Source | Fields |
|---|---|
| The broadcast scorebug | down, distance, quarter, clock, score, teams |
| The field itself | formation, shotgun vs under centre, RB/TE/WR counts, defenders in the box, motion |

Press **Apply to the form** and those fields populate the controls. **Nothing is
auto-predicted from the photo** — you see exactly which fields were read, the
model's confidence, and any caveats it flagged, and you can correct anything
before predicting. Fields that were not clearly visible are left exactly as you
had them rather than guessed.

This uses Claude's vision model and needs an Anthropic API key:

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...
# Mac / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key the photo panel explains that it is disabled and **everything else
works normally** — the key is only ever used for reading images, never for the
prediction itself.

> The photo path is a **data-entry shortcut, not a second model**. Whatever the
> photo produces flows into the exact same XGBoost pipeline a typed-in situation
> uses. Vision never predicts the play call.

### The result panel

- **Prediction card** — the call, both probabilities, a split bar, and a
  confidence band. PASS and RUN are separated by position, label and percentage,
  not by colour alone.
- **Why the model called it that way** — a SHAP-style horizontal contribution
  chart. Bars right pushed toward PASS, bars left toward RUN, length is the size
  of the push. Each bar is restated as a plain-English sentence underneath.
- **How down and distance affect pass probability** — pass probability swept
  across every down and distance with the rest of the game state held fixed,
  with the 50% decision line, a diamond at each down's flip point, and your own
  play called out.
- **Quiet notices** when an input falls outside the model's training range.
- **What the model actually saw** — every derived bucket and tendency value with
  the sample size behind it.

### Theme

A **Night / Light** toggle sits at the top right. Night is the default. The
toggle is stored in session state and restyles every surface, including both
charts — dark mode uses its own colour steps chosen for the navy surface, not an
inverted copy of the light ones.

Confidence is worded as a statement about the model, never a promise about the
play:

| Winning probability | Reads as |
|---|---|
| 50–57% | Very uncertain |
| 57–65% | Slight lean |
| 65–75% | Moderate confidence |
| above 75% | Strong model confidence |

---

## 9. Making a CLI prediction

```bash
python -m src.predict --posteam KC --defteam BUF --down 3 --ydstogo 8 \
    --yardline-100 45 --qtr 4 --clock 2:15 --score-diff -4 --shotgun
```

Useful flags: `--week`, `--season-type REG|POST`, `--posteam-timeouts`,
`--defteam-timeouts`, `--no-huddle`, `--goal-to-go`, `--away`, `--formation`,
`--personnel`, `--defenders-in-box`, `--motion`, `--qb-out`, `--coach`.
Run `python -m src.predict --help` for the full list.

The photo path works headlessly too — any flag you also pass wins over what the
photo showed:

```bash
python -m src.predict --image play.jpg --posteam KC --defteam BUF
```

Programmatically:

```python
from src.predict import PlayCallPredictor

predictor = PlayCallPredictor()
result = predictor.predict(
    posteam="KC", defteam="BUF", down=3, ydstogo=8,
    yardline_100=45, qtr=4, quarter_seconds_remaining=135,
    score_differential=-4, shotgun=True,
)
print(result.prediction)         # "PASS"
print(result.pass_probability)   # 0.959...
print(result.confidence)         # "Very high"
print(result.format())           # the full text block shown at the top

```

---

## 10. Model performance

All figures below are on the **2025 test season (34,415 plays)**, which the model
never saw during training, tuning, or calibration.

| Model | Accuracy | ROC AUC | Log loss | Brier |
|---|---|---|---|---|
| Majority-class baseline (always PASS) | 0.6041 | 0.500 | 0.6716 | 0.2393 |
| Logistic regression | 0.7510 | 0.816 | 0.5140 | 0.1707 |
| **XGBoost (final)** | **0.7614** | **0.838** | **0.4804** | **0.1600** |

XGBoost beats the majority-class baseline by **+15.7 accuracy points** and beats
logistic regression on every metric, so it was selected as the final model
(decided on the validation season, not the test season).

**Per-class performance**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| PASS | 0.804 | 0.800 | 0.802 |
| RUN | 0.697 | 0.702 | 0.700 |

Balanced accuracy is 0.751, so the model is not simply riding the 60/40 class
imbalance — it identifies runs nearly as well as passes.

### By season

The generalisation gap is visible and worth being honest about: seasons the
model trained on score noticeably higher than seasons it did not.

| Season | Split | Accuracy | AUC |
|---|---|---|---|
| 2016–2023 | train (in-sample) | 0.793 – 0.836 | 0.883 – 0.922 |
| 2024 | validation | 0.785 | 0.863 |
| **2025** | **test** | **0.761** | **0.838** |

About 4–7 accuracy points of the in-sample number is optimism. Only the 2025 row
is an honest estimate of performance on a future season.

### By situation

| Slice | Best | Worst |
|---|---|---|
| Down | 3rd down: 87.5% (AUC 0.906) | 1st down: 70.3% (AUC 0.771) |
| Distance | 11+ yards: 85.5% | 2–3 yards: 73.6% |
| Quarter | Q4: 81.0% | Q1: 72.8% |
| Score state | trailing 9+: 79.7% | tied: 74.0% |
| Field zone | backed up: 77.8% | goal line: 71.4% |

This matches football intuition. Third-and-long is nearly deterministic; first-
and-10 in a tied first quarter is where coordinators have — and use — the most
freedom. Goal-line is the hardest zone because run/pass is genuinely balanced
and the field is compressed.

Team predictability spans **70.6% (LA Rams) to 81.3% (Atlanta)** — a 10-point
spread in how legible different offenses are.

### Most important features (by gain)

| # | Feature | Share |
|---|---|---|
| 1 | Offensive formation | 18.9% |
| 2 | Shotgun alignment | 18.2% |
| 3 | Head coach | 11.0% |
| 4 | Offensive team | 4.6% |
| 5 | Obvious passing situation | 3.6% |
| 6 | Distance bucket | 3.3% |
| 7 | Defensive team | 3.3% |
| 8 | QB alignment | 2.8% |
| 9 | Defenders in the box | 2.7% |
| 10 | Game phase | 2.5% |

The pre-snap *look* (formation, shotgun, QB alignment, box count) dominates,
which is exactly what you would expect and also a caution: if you do not know
the formation, the model is working with substantially less information.

---

## 11. Probability calibration

Calibration asks whether the probabilities mean anything: when the model says
70%, does a pass actually follow about 70% of the time?

**Expected calibration error on the test season: 0.0116** — predictions are off
by about 1.2 percentage points on average.

| Predicted band | Plays | Predicted | Observed | Gap |
|---|---|---|---|---|
| 0–10% | 846 | 6.3% | 7.9% | 1.7% |
| 10–20% | 2,319 | 15.5% | 18.2% | 2.7% |
| 20–30% | 3,562 | 25.3% | 28.0% | 2.7% |
| 30–40% | 3,914 | 34.8% | 34.5% | 0.4% |
| 40–50% | 3,069 | 44.9% | 42.8% | 2.1% |
| 50–60% | 2,945 | 55.0% | 54.9% | 0.1% |
| 60–70% | 2,933 | 64.9% | 63.0% | 1.9% |
| 70–80% | 2,932 | 75.1% | 74.8% | 0.3% |
| 80–90% | 3,563 | 85.3% | 83.7% | 1.5% |
| 90–100% | 8,332 | 95.7% | 96.0% | 0.3% |

The consistent small over-prediction of runs in the 10–30% bands is the only
systematic bias, and it is minor.

### Accuracy by confidence

| Confidence | Coverage | Accuracy |
|---|---|---|
| 50–60% ("coin flip") | 17.5% | 56.1% |
| 60–70% | 19.9% | 64.5% |
| 70–80% | 18.9% | 73.3% |
| 80–90% | 17.1% | 83.0% |
| 90–100% | 26.7% | 95.7% |
| **≥ 70% (cumulative)** | **62.6%** | **85.5%** |
| **≥ 90% (cumulative)** | **26.7%** | **95.7%** |

Accuracy tracks stated confidence closely at every level, which is the practical
test of calibration. Note also that roughly **one play in six is a genuine coin
flip** — the model knows when it does not know.

### Was calibration applied?

**No — and that is the reported result, not an omission.** Isotonic calibration
was fitted on the first 70% of the validation season and judged on the remaining
10,437 plays:

```
uncalibrated log loss = 0.45865
isotonic calibrated   = 0.45967   -> rejected
```

XGBoost's raw probabilities were already better calibrated than the isotonic
remapping, so the uncalibrated model ships.

Importantly, **that decision was made without looking at the test season.**
Choosing whether to calibrate based on test performance would be test-set
peeking and would quietly inflate the headline metrics. Set
`model.calibration` to `sigmoid` or `none` in `config.yaml` to change the
method; the same holdout procedure re-runs either way.

---

## How data leakage is prevented

This is the part of the project that matters most. A model that sees
`yards_gained` or `complete_pass` will score above 99% and be completely
worthless, because those columns *are* the answer.

**Three independent defences:**

**1. A denylist enforced in code.** `src/leakage.py` maintains exact column
names and substring patterns covering outcome families — yards gained, EPA/WPA,
completions, touchdowns, turnovers, sacks, first downs, penalties, the players
who touched the ball, drive and series results, and any `next_*` look-ahead
column. `assert_no_leakage()` raises and **fails the build** if a banned name
reaches the feature list. It runs when features are built *and* again at the top
of training, so a stale matrix on disk cannot slip through.

The full, audited feature list is printed before every training run.

**2. Label sources are dropped after labelling.** `pass`, `rush`, `qb_scramble`,
`play_type`, `desc` and friends are used to build the target and then removed
from the frame entirely.

**3. Tendency features are exclusive by construction.** Every historical rate
uses a strictly-earlier window:

```python
prior_sum = groupby(key).cumsum() - y   # sum over strictly earlier plays
prior_cnt = groupby(key).cumcount()     # count of strictly earlier plays
rate = (prior_sum + k * league_prior) / (prior_cnt + k)
```

Subtracting `y` and using a zero-based `cumcount` is what makes the window
exclusive — a play can never contribute to its own tendency feature. Even the
league-average smoothing prior is an expanding, strictly-prior quantity, because
using the whole-dataset mean would leak a sliver of each play's own outcome back
into its own feature.

This is verified by a **perturbation test** (`tests/test_tendencies.py`): flip a
single play's label, recompute, and assert that play's own features are
bit-identical while later plays do change. That test caught a real leak during
development.

Four flavours of history are combined, all leakage-safe: expanding career-to-
date, rolling recent form (last 200 plays), previous season, and previous game.
Small samples are shrunk toward the league average with an empirical-Bayes
update (`k = 50`), so a coach with three prior third-and-shorts does not get a
100% run rate.

### What is deliberately excluded even though it was requested

`number_of_pass_rushers` is available in the participation data, but the number
of rushers is only observable **after** the snap. It is excluded by default and
recorded in the "unavailable features" report. Set
`features.allow_post_snap_pass_rushers: true` in `config.yaml` to include it,
understanding that it is leakage.

The same reasoning excludes FTN's `is_play_action`, `is_screen_pass`, `is_rpo`,
`n_blitzers` and `n_pass_rushers` — several of which encode the play call
outright.

---

## Photo-reading limitations

The photo panel is a convenience for filling in the form. It is the least
reliable part of this project and should be treated that way.

- **It is a reader, not a predictor.** The vision model is explicitly instructed
  not to guess the play call, and its output is never used as a feature. It
  fills form fields; the XGBoost model does the predicting.
- **Nothing is inferred.** Fields that are not clearly visible come back as
  "unknown" and are left at whatever you had set. A tight sideline shot with no
  scorebug will populate almost nothing — that is correct behaviour, not a
  failure.
- **Always check what it read.** The panel lists exactly which fields came from
  the image and reports its own confidence. Personnel counts and box counts are
  the easiest things to get wrong, especially on an angled or zoomed shot.
- **Camera angle beats everything.** An end-zone or tight shot cannot show the
  full formation. A blurry or small scorebug is often misread by a digit.
- **Yard line is the weakest field.** Converting a broadcast "KC 45" graphic into
  yards-from-the-opponent's-end-zone requires knowing which team has the ball
  and which direction they are going. Verify it.
- **A still frame cannot show motion.** Motion is a movement, so unless a player
  is visibly mid-stride in a way the model can identify, this stays unknown.
- **It costs money and needs a network call.** Each read is one Anthropic API
  request. The rest of the project is free and offline after the initial data
  download.
- **Images are sent to the Anthropic API.** Don't upload anything you would not
  want to leave your machine.

If any of this bothers you, ignore the panel entirely — typing the situation in
is the primary path, and it is exact.

---

## Feature list

**121 pre-snap features** (109 numeric, 12 categorical).

**Situation** — down, yards to go, goal-to-go, `yardline_100`, quarter, seconds
remaining in the quarter / half / game, score differential from the offense's
perspective, leading/tied/trailing, offensive and defensive timeouts, home/away,
regular season vs playoffs, week, two-minute flag, obvious-passing-situation
flag, short-yardage flag, plus derived ratios (comeback urgency, timeout
advantage, yards-to-go vs field position).

**Buckets** (kept *alongside* the raw numerics, never instead of them):

- Distance: `1`, `2-3`, `4-6`, `7-10`, `11+`
- Field position: `backed_up`, `own_territory`, `midfield`, `opp_territory`, `red_zone`, `goal_line`
- Score state: `trail_9plus`, `trail_1_8`, `tied`, `lead_1_8`, `lead_9plus`
- Time state: `early_game`, `second_half`, `fourth_quarter`, `final_five_min`, `final_two_min`

**Formation and personnel** — shotgun, no-huddle, offensive formation
(SHOTGUN / SINGLEBACK / I_FORM / EMPTY / PISTOL / JUMBO / WILDCAT), personnel
grouping parsed from strings like `"1 RB, 1 TE, 3 WR"` into RB/TE/WR/OL/QB
counts, defensive personnel (DL/LB/DB counts), defenders in the box, QB
alignment (under centre / shotgun / pistol), pre-snap motion, starting hash, and
explicit missing-value indicators for each.

**Tendencies** (23 specs → 46 columns, each paired with its sample size) — the
offense's prior pass rate overall and by down, distance, score situation, field
zone, red zone, short yardage and two-minute; recent form; previous season;
previous game. The head coach's equivalents. The opposing defense's rate of
facing runs and passes overall, by down, by distance and by score state.

**Injuries** — players ruled out and not practising, in total and by position
group (QB / RB / WR / TE / OL), plus a report-availability indicator.

---

## Project structure

```
nfl-play-call-predictor/
├── app.py                      # Streamlit UI
├── config.yaml                 # seasons, smoothing, split, model settings
├── requirements.txt
├── pytest.ini
├── src/
│   ├── config.py               # paths, config loading, season detection
│   ├── nflverse_sources.py     # dataset catalogue + nflreadpy/HTTP access
│   ├── download_data.py        # caching downloader + data inventory
│   ├── clean_data.py           # RUN/PASS labelling, filtering, cleaning
│   ├── tendencies.py           # leakage-safe historical rates
│   ├── build_features.py       # feature engineering + leakage audit
│   ├── leakage.py              # the denylist and its enforcement
│   ├── preprocessing.py        # ColumnTransformer (separate module so it pickles)
│   ├── train.py                # split, 3 models, tuning, calibration, save
│   ├── evaluate.py             # slice metrics + figures
│   ├── explain.py              # per-prediction SHAP factors
│   ├── theme.py                # design tokens, CSS, reusable HTML components
│   ├── charts.py               # Plotly factor + sensitivity charts
│   ├── vision.py               # optional: photo -> pre-snap situation
│   └── predict.py              # PlayCallPredictor + CLI
├── tests/
├── data/raw/                   # cached nflverse downloads
├── data/processed/             # labeled plays, model matrix, tendency store
├── models/play_call_model.joblib
└── reports/                    # metrics, evaluation, figures
```

Full pipeline from scratch:

```bash
pip install -r requirements.txt
python -m src.download_data
python -m src.clean_data
python -m src.build_features
python -m src.train
python -m src.evaluate
streamlit run app.py
```

---

## Tests

```bash
python -m pytest
```

**272 tests**, covering:

| File | What it checks |
|---|---|
| `test_leakage.py` | ~50 known post-play columns are each rejected; clean pre-snap columns are not falsely flagged; the **actual persisted feature list** passes the audit |
| `test_labeling.py` | Every label rule and exclusion — sacks and scrambles count as PASS, punts/kneels/spikes/2-pt/penalty-only are dropped, duplicates removed, invalid downs and distances rejected, franchise aliases don't chain |
| `test_features.py` | Bucket boundaries, that the vectorised and scalar bucket paths agree on 500 random situations, personnel-string parsing, clock derivation including overtime |
| `test_tendencies.py` | Tendency exclusivity by **perturbation**, smoothing behaviour, and that serving-time lookups actually hit training history |
| `test_predict.py` | Saved-model loading, a complete sample prediction, output format, and football sanity checks |
| `test_charts.py` | Theme-token integrity, that action green is never a data colour, factor-bar direction and ordering, and that every chart's text uses theme ink rather than default black |
| `test_vision.py` | Photo-reading logic without touching the API: derived fields, clamping of impossible values, formation/shotgun reconciliation, credential detection, image downscaling |
| `test_app.py` | Runs the real Streamlit dashboard headlessly via `AppTest`, clicks Predict in **both themes**, and asserts no raw markup ever reaches the page as text |

The football sanity checks are worth calling out — they assert the model
reproduces things that must be true: 3rd-and-12 leans more pass than 3rd-and-1,
trailing late raises pass probability, leading late raises run probability, and
shotgun raises pass probability. A model that passes the leakage audit but fails
these is broken in a way metrics alone would not reveal.

---

## 12. Known limitations

- **A pre-snap ceiling exists.** Play-callers deliberately stay unpredictable.
  Even a perfect model cannot exceed the genuine uncertainty in the situation;
  much of what decides a call (the coordinator's script, a coverage read, a
  sideline adjustment) is simply not in any free dataset.
- **No player-level context.** The model does not know the quarterback's
  identity or quality, offensive-line health beyond injury counts, or weather.
- **Head coach is a proxy for the play-caller.** See §14.
- **Motion and QB alignment only exist from 2022.** FTN charting covers 40.8% of
  the modelled plays. Earlier plays carry NaN, which XGBoost handles natively,
  but the model leans on those features less than it would with full coverage.
- **Formation and personnel are informative but not always known pre-snap in
  practice.** In the app they are optional; leaving them unknown is honest and
  the model handles it.
- **Team relocations are collapsed.** OAK→LV, SD→LAC and STL→LA are treated as
  continuous franchises so tendency history survives the move. This is usually
  right for scheme continuity and wrong if you consider the move a hard reset.
- **Tendencies are league-smoothed.** Early-season and small-sample situations
  are pulled toward the league average by design, which makes them conservative.
- **The model has no memory of the current drive's play sequence** beyond the
  expanding within-game tendency.
- **Retraining is required as seasons complete.** Tendencies drift; the saved
  model reflects the seasons it was trained on.
- **The photo reader is the weakest link in the chain.** See
  [photo-reading limitations](#photo-reading-limitations); always check what it
  extracted before trusting a prediction built on it.

---

## 13. Injury-data limitations

Injury features come from the official weekly injury/practice report and are
joined on `(season, week, team)` — pregame information, so they are pre-snap safe.

- **Team-week granularity, not play granularity.** A player ruled out mid-game
  is not reflected; the report is a Friday snapshot.
- **`report_status` is frequently blank.** Roughly a third of listed players
  have no game-status designation, so "players out" is a floor, not a census.
- **No indication of who actually played.** The report says who was *ruled out*,
  not who took snaps, and never says who was on the field for a given play.
- **Position labels are inconsistent across seasons** (`T`/`G`/`C` vs `OL`);
  these are grouped, but the grouping is approximate.
- **A missing team-week is treated as zero players out.** That is usually
  correct but conflates "nobody hurt" with "no report filed". An
  `inj_report_available` indicator is included so the model can tell them apart.
- **In the app, injury inputs default to zero.** If you want the model to
  account for a backup quarterback, you must set it yourself.
- Depth charts, which would have identified *which* players were starters, are
  unusable for the test season (see §3).

---

## 14. Coach-data limitations

nflverse publishes `home_coach` / `away_coach` per game, which joins to **100%
of modelled plays** across 95 distinct head coaches. That is genuinely useful —
but it is the **head coach, not the play-caller**.

- Many head coaches do not call plays; an offensive coordinator usually does.
  nflverse does not publish coordinator or play-caller identity, so there is no
  free way to correct this.
- Play-calling duties change mid-season without any signal in the data.
- A coach's tendency history follows the *person*, so it carries across teams —
  usually desirable, but it blends very different rosters and coordinators.
- Interim coaches and first-year coaches have thin histories; smoothing pulls
  them toward the league average, which is safe but uninformative.
- Coach identity is also included as a categorical feature; coaches unseen in
  training fall back to league-average behaviour via `handle_unknown`.

If coach data were unavailable, the code falls back to offensive team + season
as the play-caller proxy. That fallback is implemented but not currently needed.

---

## 15. Why you should not gamble on this

**This model estimates historical frequencies. It does not predict the future.**

- A "68% pass" output means *among historical situations resembling this one,
  about 68% were passes*. It says nothing certain about the next snap.
- The model is trained on aggregate tendencies and is systematically blind to
  the things that decide individual plays: the specific game plan, an audible, a
  defensive look, a coach's whim, information from the sideline.
- Play-callers are adversarial. Any tendency reliable enough to bet on is
  exactly the tendency a competent coordinator breaks.
- Betting markets already price in everything here and considerably more,
  including information this model cannot see. There is no edge in a public
  model built on public data.
- Accuracy is measured across tens of thousands of plays. Per-play variance is
  enormous, and a run of correct calls is not evidence of predictive power.
- The model degrades as the game changes, and it has no idea when it is wrong
  beyond the calibration reported above.

Use it to explore tendencies, to study situational football, and as a worked
example of leakage-safe sports modelling. Do not use it to bet money.

---

## License and attribution

Data © [nflverse](https://github.com/nflverse), used under its terms and freely
available. This project is unaffiliated with the NFL or nflverse.
