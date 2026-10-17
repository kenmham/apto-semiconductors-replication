"""Original pilot prompt templates and snapshot construction; offline only."""
from pathlib import Path
import numpy as np
import polars as pl
MODEL = 'claude-opus-4-8'
PROMPT_VERSION = 'v2'
RUNGS = ['R0_blind', 'R1_year']
_V1_BASES = ['recalled_specific_history', 'inferred_from_market_structure', 'base_rate_only', 'no_information']
_V1_CONFS = ['low', 'medium', 'high']
_V1_TASK = 'You are forecasting technological progress in an electronics component market.\n\nYou will be shown the competitive state of ONE capability metric within ONE product\nsubcategory, as of a reference point in time. Your job: judge how likely it is that the\nSTATE OF THE ART for that metric (the single best value achieved across all\nmanufacturers) makes a LARGE JUMP during the next year.\n\n"Large jump" means the best-in-class improves by more than about 25% in that one\nyear. Across this catalog, in any given year a given metric jumps only about 12-15%\nof the time, so a large jump is the exception, not the rule.\n\nReason as a market analyst who knows how these components advance: what drives the\nfrontier in this kind of part, whether the competitive structure shown looks like a\nmarket about to leap or one that is coasting.{year_line}\n\n{snapshot}\n\nThink first, then commit to a number. In your reasoning, weigh what the market\nstructure implies AND what you know about how this class of component tends to\nadvance. Then report:\n- reasoning: at most 4 sentences.\n- year_guess: your single best estimate of the calendar year of this reference point.\n- basis: exactly one of {bases}.\n- confidence: one of {confs}.\n- jump_likelihood: an integer 0-100, your probability that a large jump occurs next year.\n\nRespond with ONE line of strict JSON, keys in exactly this order:\n{{"reasoning": "...", "year_guess": <int>, "basis": "...", "confidence": "...", "jump_likelihood": <int>}}'
_V1_BATCH = 'You are forecasting technological progress in an electronics component market.\n\nBelow are SEVERAL independent cases. Treat each one entirely on its own; they are\nunrelated parts and reference points. For each, judge how likely it is that the\nSTATE OF THE ART for that metric makes a LARGE JUMP during the next year.\n\n"Large jump" means the best-in-class improves by more than about 25% in that one\nyear. In any given year a given metric jumps only about 12-15% of the time.{year_line}\n\nFor each case report: reasoning (<=4 sentences); year_guess; basis (one of {bases});\nconfidence (one of {confs}); jump_likelihood (integer 0-100).\n\nCASES:\n{cases}\n\nRespond with ONE JSON array of {n} objects, same order, each:\n{{"case": <int>, "reasoning": "...", "year_guess": <int>, "basis": "...", "confidence": "...", "jump_likelihood": <int>}}\nOutput only the JSON array.'
_V2_TASK = 'You are a market analyst assessing one capability metric in one category of electronic components.\n\nYou will see the competitive state of that metric at a reference point: how the\nbest-in-class has been moving, who leads and by how much, and how the field around\nthe leader is behaving. Read these specifics closely.{year_line}\n\n{snapshot}\n\nQuestion: reading the facts above, how primed does this particular state look for the\nsingle best-in-class value to take a LARGE upward step in the near future, meaning a\nstep clearly bigger than this metric\'s ordinary year-to-year drift?\n\nReason from the specific facts shown: what would push this frontier forward now, and\nwhether what you see points toward an imminent step or a settled, quiet stretch. Use\nwhat you know about how this kind of part advances only to interpret those facts, not\nto answer from the category alone.\n\nThen give:\n- reasoning: 2 to 4 sentences, grounded in the specific facts shown.\n- propensity: an integer 0-100 scoring how primed this state is for a near-term large\n  step (0 = no sign of one, 100 = every sign of one). This is a relative ranking score,\n  not a calibrated probability: use the full range, and let clearly different states get\n  clearly different scores.\n\nRespond with ONE line of strict JSON, keys in this order:\n{{"reasoning": "...", "propensity": <int>}}'
_V2_BATCH = 'You are a market analyst assessing capability metrics in electronic components.\n\nBelow are SEVERAL independent cases. Treat each entirely on its own; they are unrelated\nmetrics and reference points. For each you see the competitive state of one metric: how\nthe best-in-class has moved, who leads and by how much, how the field is behaving.{year_line}\n\nFor each case, reading its facts, score how primed that state looks for the single\nbest-in-class value to take a LARGE upward step soon (a step clearly bigger than that\nmetric\'s ordinary year-to-year drift). Reason from the specific facts shown.\n\nCASES:\n{cases}\n\nRespond with ONE JSON array of {n} objects, same order as the cases, each:\n{{"case": <int>, "reasoning": "<2-4 sentences on THIS case>", "propensity": <int 0-100>}}\npropensity is a relative ranking score, not a probability: use the full range so clearly\ndifferent states get clearly different scores. Output only the JSON array.'
_YEAR_BLIND = '\n\nThe reference year is deliberately withheld; infer everything from the market structure and your own knowledge.'

def _year_line(rung, ref_year):
    if rung == 'R1_year':
        return f'\n\nThe reference point is the end of calendar year {ref_year}; you are judging the following year, {ref_year + 1}.'
    return _YEAR_BLIND

def build_prompt(rung, snapshot, ref_year, version=PROMPT_VERSION):
    if version == 'v1':
        return _V1_TASK.format(year_line=_year_line(rung, ref_year), snapshot=snapshot, bases=' | '.join(_V1_BASES), confs='/'.join(_V1_CONFS))
    return _V2_TASK.format(year_line=_year_line(rung, ref_year), snapshot=snapshot)

def build_batch_prompt(rung, snapshots, ref_years, version=PROMPT_VERSION):
    cases = '\n\n'.join((f'[case {i + 1}]' + (f' (reference year: end of {ref_years[i]})' if rung == 'R1_year' else '') + f'\n{s}' for i, s in enumerate(snapshots)))
    yl = '\n\nEach case states its own reference year.' if rung == 'R1_year' else _YEAR_BLIND
    if version == 'v1':
        return _V1_BATCH.format(year_line=yl, cases=cases, n=len(snapshots), bases=' | '.join(_V1_BASES), confs='/'.join(_V1_CONFS))
    return _V2_BATCH.format(year_line=yl, cases=cases, n=len(snapshots))

PILOT_SUBS = {'FPGAs (Field Programmable Gate Array)': 'signal', 'LDO Voltage Regulators': 'control'}
METRIC_LABELS = {'numberOfGates': 'number of gates', 'numberOfIO': 'number of I/O pins', 'numberOfLabsClbs': 'number of logic array blocks / CLBs', 'numberOfLogicElementsCells': 'number of logic elements / cells', 'totalRamBits': 'total on-chip RAM bits', 'dropoutVoltageMax': 'dropout voltage (max)', 'dropoutVoltage': 'dropout voltage', 'operatingSupplyCurrent': 'operating supply current', 'outputCurrent': 'output current', 'psrrRippleRejectionTyp': 'power-supply ripple rejection (PSRR)', 'quiescentCurrent': 'quiescent current', 'voltageRegulationAccuracy': 'output-voltage regulation accuracy'}
BETTER = {'numberOfGates': 'higher', 'numberOfIO': 'higher', 'numberOfLabsClbs': 'higher', 'numberOfLogicElementsCells': 'higher', 'totalRamBits': 'higher', 'dropoutVoltageMax': 'lower', 'dropoutVoltage': 'lower', 'operatingSupplyCurrent': 'lower', 'outputCurrent': 'higher', 'psrrRippleRejectionTyp': 'higher', 'quiescentCurrent': 'lower', 'voltageRegulationAccuracy': 'lower'}

def _trend(x, up, down, flat='held roughly steady', thr=0.02):
    if x is None:
        return None
    if x > thr:
        return up
    if x < -thr:
        return down
    return flat

def _num(x):
    return None if x is None else float(x)

def verbalize(row):
    col = row['metric_col']
    metric = METRIC_LABELS.get(col, col)
    better = BETTER.get(col, 'higher')
    lines = [f"Product subcategory: {row['subcategory']}.", f'Capability metric: {metric} ({better} is better).']
    b = []
    rc = _num(row.get('recent_change'))
    if rc is not None:
        pct = (10 ** rc - 1) * 100
        if pct < 1:
            b.append('The best-in-class value has barely moved over the last 3 years.')
        else:
            b.append(f'Over the last 3 years the best-in-class improved by roughly {pct:.0f}%.')
    ysj = _num(row.get('yrs_since_jump'))
    if ysj is not None:
        b.append(f'The frontier last made a large jump about {ysj:.0f} year(s) ago.')
    held = _num(row.get('leader_held'))
    if held is not None:
        b.append(f'The current record has stood unbeaten for {held:.0f} year(s).')
    eff = _num(row.get('eff_mfrs'))
    if eff is not None:
        b.append(f'About {eff:.1f} manufacturers are effectively competing at the top tier.')
    margin = _num(row.get('leader_margin'))
    t = _trend(margin, 'The single record-holder sits well ahead of the rest of the leading pack.', 'The record-holder is barely ahead of the pack.', 'The record-holder leads the pack by a moderate margin.', thr=0.05)
    if t:
        b.append(t)
    nm = _num(row.get('new_mfr_rate_3y'))
    nl = _num(row.get('n_leaders_3y'))
    if nm is not None:
        s = f'Over the last 3 years {nm:.0f} new manufacturer(s) broke into the top tier'
        if nl is not None:
            s += f', and the lead changed hands among {nl:.0f} different maker(s)'
        b.append(s + '.')
    t = _trend(_num(row.get('gap_trend_3y')), 'The gap between the leader and the pack has been widening.', 'The pack has been compressing toward the leader.', 'The leader-to-pack gap has held roughly steady.')
    if t:
        b.append(t)
    t = _trend(_num(row.get('near_frontier_trend')), 'A growing share of parts now cluster near the frontier.', 'Fewer parts sit near the frontier than before.', 'The share of parts near the frontier is roughly flat.')
    if t:
        b.append(t)
    t = _trend(_num(row.get('median_slope_3y')), 'The typical (median) part has been improving briskly.', 'The typical (median) part has been slipping.', 'The typical (median) part has been roughly flat.')
    if t:
        b.append(t)
    t = _trend(_num(row.get('inflow_accel')), 'The inflow of new parts into the catalog is accelerating.', 'The inflow of new parts is decelerating.', 'New-part inflow is steady.')
    if t:
        b.append(t)
    sib_last = _num(row.get('sib_jumped_last'))
    sib_rec = _num(row.get('sib_jumps_recent'))
    if sib_rec:
        b.append(f'A related capability metric in this same subcategory jumped {int(sib_rec)} time(s) in the last 3 years' + (' (including last year).' if sib_last else '.'))
    cg = _num(row.get('challenger_gapclose_3y'))
    t = _trend(cg, 'The nearest challenger has been closing on the record-holder.', 'The nearest challenger has been falling further behind.', "The nearest challenger's gap to the record has held steady.")
    if t:
        b.append(t)
    return '\n'.join(lines + ['', 'Market structure as of the reference point:'] + [f'- {x}' for x in b])

PHRASINGS = {'yrs_since_jump': ('The frontier last made a large jump about 1 year ago.', 'The frontier last made a large jump about 4 years ago.', 'The frontier last made a large jump about 10 years ago.'), 'leader_held': ('The current record has stood unbeaten for 1 year.', 'The current record has stood unbeaten for 3 years.', 'The current record has stood unbeaten for 7 years.'), 'eff_mfrs': ('About 1.2 manufacturers are effectively competing at the top tier.', 'About 2.5 manufacturers are effectively competing at the top tier.', 'About 4.5 manufacturers are effectively competing at the top tier.'), 'leader_margin': ('The single record-holder is barely ahead of the leading pack.', 'The single record-holder leads the pack by a moderate margin.', 'The single record-holder sits well ahead of the leading pack.'), 'new_mfr_rate_3y': ('Over the last 3 years no new manufacturers broke into the top tier.', 'Over the last 3 years 1 new manufacturer broke into the top tier.', 'Over the last 3 years 3 new manufacturers broke into the top tier.'), 'near_frontier_trend': ('Fewer parts sit near the frontier than before.', 'The share of parts near the frontier is roughly flat.', 'A growing share of parts now cluster near the frontier.'), 'median_slope_3y': ('The typical (median) part has been slipping.', 'The typical (median) part has been roughly flat.', 'The typical (median) part has been improving briskly.'), 'challenger_gapclose_3y': ('The nearest challenger has been falling further behind.', "The nearest challenger's gap to the record has held steady.", 'The nearest challenger has been closing on the record-holder.'), 'packaging': ('The parts are predominantly offered in through-hole packaging.', 'The parts come in a mix of package formats.', 'The parts are predominantly offered in surface-mount packaging.')}
FEATURES = list(PHRASINGS)
REAL_FEATURES = [f for f in FEATURES if f != 'packaging']

def _bucket(v, lo, hi):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 1
    return 0 if v <= lo else 2 if v >= hi else 1

def base_cases(n, seed=0):
    df = pl.read_csv(Path(__file__).resolve().parents[1]/'data/pilots/panel.csv')
    pct = {f: (df[f].quantile(0.2), df[f].quantile(0.8)) for f in REAL_FEATURES}
    rng = np.random.default_rng(seed)
    picks = []
    for arm in ('signal', 'control'):
        ids = df.filter(pl.col('arm_kind') == arm)['part_id'].to_list()
        picks += list(rng.choice(ids, min(n // 2, len(ids)), replace=False))
    cases = []
    for pid in picks:
        r = df.filter(pl.col('part_id') == pid).row(0, named=True)
        levels = {f: _bucket(r[f], *pct[f]) for f in REAL_FEATURES}
        levels['packaging'] = 1
        cases.append(dict(part_id=int(pid), arm=r['arm_kind'], sub=r['subcategory'], metric=r['metric_col'], levels=levels))
    return cases

def build_variants(case, seed):
    base = dict(case['levels'])
    variants = []
    for f in FEATURES:
        for lvl, name in ((0, 'lo'), (2, 'hi')):
            lv = dict(base)
            lv[f] = lvl
            lines = [PHRASINGS[g][lv[g]] for g in FEATURES]
            variants.append(dict(feature=f, level=name, lines=lines))
    order = np.random.default_rng(seed).permutation(len(variants))
    return [variants[i] for i in order]

def build_ablation_prompt(case, variants):
    metric = METRIC_LABELS.get(case['metric'], case['metric'])
    better = BETTER.get(case['metric'], 'higher')
    head = f"You are a market analyst assessing one capability metric in electronic components.\nSubcategory: {case['sub']}. Metric: {metric} ({better} is better).\n\nBelow are several snapshots of this metric's competitive state. For each, score 0-100 how primed that state looks for the single best-in-class value to take a LARGE upward step in the near future (a step clearly bigger than this metric's ordinary year-to-year drift). Use the full 0-100 range so clearly different states get clearly different scores. Judge each state on its own facts.\n\nSTATES:"
    body = '\n\n'.join((f'[state {i + 1}]\n' + '\n'.join((f'- {ln}' for ln in v['lines'])) for i, v in enumerate(variants)))
    tail = f'\n\nRespond with ONE JSON array of {len(variants)} objects, same order as the states, each {{"case": <int>, "propensity": <int 0-100>}}. Output only the JSON array.'
    return head + '\n' + body + tail

def build_attribution_prompt(grp):
    head = 'You are an electronics technology historian. For each query below, judge whether the best-in-class value of the stated capability metric, across all manufacturers in that product category, made a LARGE step up around the stated year. If it did, name the one specific EXTERNAL development that drove it (a named process node, material system, industry standard, architecture, or end-market shift) and its approximate year. If nothing notable happened to that metric around then, say so plainly. Do not force an explanation, and do not invent a development you are unsure of.\n\nQUERIES:'
    body = '\n'.join((f"[q{i + 1}] category: {q['subcategory']}; metric: {METRIC_LABELS.get(q['metric_col'], q['metric_col'])} ({BETTER.get(q['metric_col'], 'higher')} is better); year: {q['year']}" for i, q in enumerate(grp)))
    tail = '\n\nRespond with ONE JSON array of ' + str(len(grp)) + ' objects, same order, each:\n{"case":<int>,"occurred":"yes"|"no"|"unsure","driver":"<specific name or none>","driver_year":<int or null>,"driver_type":"process_node|materials|standard|architecture|end_market|other|none","confidence":"low"|"medium"|"high","reasoning":"<=2 sentences"}\nOutput only the JSON array.'
    return head + '\n' + body + tail
