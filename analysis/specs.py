import os, re
import numpy as np, polars as pl

_SUFFIX = re.compile(r"[,\.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|"
                     r"limited|llc|gmbh|kk|plc|ab|nv|sa)\.?$", re.I)
def norm(m):

    if m is None:
        return "Unknown"
    m = re.sub(r"\s+", " ", str(m)).strip()
    prev = None
    while prev != m:
        prev = m; m = _SUFFIX.sub("", m).strip(" ,.")
    return m or "Unknown"


SRC = os.environ["APTO_DATA"]

Y0 = int(os.environ.get("WK3_Y0", "2000"))
Y1 = int(os.environ.get("WK3_Y1", "2024"))
MIN_PER_YEAR, MIN_YEARS = 40, 8


def tag():

    return "" if (Y0 <= 2000 and Y1 >= 2024) else f"_{Y0}-{Y1}"


SPECS = [
    ("clockFrequency",            "Memory",                                    "Memory clock (Hz)",        "hi"),
    ("accessTime",                "Memory",                                    "Memory access time (s)",   "lo"),
    ("totalRamBits",             "FPGAs (Field Programmable Gate Array)",      "FPGA RAM bits",            "hi"),
    ("numberOfLogicElementsCells","FPGAs (Field Programmable Gate Array)",     "FPGA logic cells",         "hi"),
    ("samplingRatePerSecond",     "Analog to Digital Converters (ADC)",        "ADC sample rate (S/s)",    "hi"),
    ("ippPeakPulseCurrent",       "Transient Voltage Suppression Diodes - TVS","TVS peak pulse current (A)","hi"),
    ("currentAverageRectifiedIo", "Single Diodes",                             "Diode rectified Io (A)",   "hi"),
    ("drainToSourceVoltageVdss",  "Single FETs, MOSFETs",                      "MOSFET Vdss (V)",          "hi"),
    ("frequencyTransition",       "Single Bipolar Transistors",                "BJT transition freq fT (Hz)","hi"),
    ("3dbBandwidth",              "Instrumentation, Op Amps, Buffer Amps",     "Op amp bandwidth (Hz)",    "hi"),
    ("voltageInputOffset",        "Instrumentation, Op Amps, Buffer Amps",     "Op amp input offset (V)",  "lo"),
    ("currentInputBias",          "Instrumentation, Op Amps, Buffer Amps",     "Op amp input bias I (A)",  "lo"),
    ("gain",                      "RF Amplifiers",                             "RF amplifier gain (dB)",   "hi"),
    ("millicandelaRating",        "LED Indication - Discrete",                 "LED brightness (mcd)",     "hi"),
    ("numberOfIO",                "Microcontrollers",                          "MCU I/O pins",             "hi"),
    ("currentQuiescentIq",        "Voltage Regulators - Linear, Low Drop Out (LDO) Regulators",
                                                                               "LDO quiescent Iq (A)",     "lo"),
    ("numberOfLabsClbs",          "FPGAs (Field Programmable Gate Array)",     "FPGA CLBs",                "hi"),
    ("slewRate",                  "Instrumentation, Op Amps, Buffer Amps",     "Op amp slew rate (V/s)",   "hi"),
    ("frequencySwitching",        "Voltage Regulators - DC DC Switching Regulators", "DC-DC switching freq (Hz)", "hi"),
    ("currentPeakPulse101000µs",  "TVS Diodes",                                "TVS pulse current 10/1000us (A)", "hi"),
    ("currentSensing",            "Current Sensors",                           "Current sensor range (A)", "hi"),
    ("rotationalLifeCyclesMin",   "Encoders",                                  "Encoder rotational life (cyc)", "hi"),
    ("powerMax",                  "Single Zener Diodes",                       "Zener power rating (W)",   "hi"),
    ("outputCurrent",             "LDO Voltage Regulators",                    "LDO output current (A)",   "hi"),
    ("currentOutput",             "Voltage Regulators - Linear, Low Drop Out (LDO) Regulators",
                                                                               "LDO current capability (A)", "hi"),
    ("voltageIsolation",          "Transistor, Photovoltaic Output Optoisolators", "Optoisolator isolation (V)", "hi"),
    ("responseTime",              "Photoelectric, Industrial",                 "Photoelectric response time (s)", "lo"),
    ("jitter",                    "Clock Generators & Support Products",       "Clock jitter (s)",         "lo"),
    ("currentLeakageIsOffMax",    "Analog Switches, Multiplexers, Demultiplexers", "Analog switch off-leakage (A)", "lo"),
]


def scan():

    return pl.scan_parquet(SRC).filter(pl.col("release_year").is_between(Y0, Y1))


def _cumbest(a, hi):
    return np.maximum.accumulate(a) if hi else np.minimum.accumulate(a)


def frontier_series(lf, col, sub, direction):


    hi = direction == "hi"
    qA, qB = (0.95, 0.99) if hi else (0.05, 0.01)
    agg = pl.col(col).max() if hi else pl.col(col).min()
    g = (lf.filter(pl.col("subcategory") == sub).select(["release_year", col])
            .drop_nulls().filter(pl.col(col) > 0)
            .group_by("release_year")
            .agg(pl.len().alias("n"),
                 pl.col(col).quantile(qA).alias("pA"),
                 pl.col(col).quantile(qB).alias("pB"),
                 agg.alias("pM"))
            .filter(pl.col("n") >= MIN_PER_YEAR)
            .sort("release_year").collect())
    if g.height < MIN_YEARS:
        return None
    yr, nn = g["release_year"].to_numpy(), g["n"].to_numpy()
    return dict(yr=yr, n=nn, hi=hi, N=int(nn.sum()),
                praw=g["pB"].to_numpy(),
                cA=_cumbest(g["pA"].to_numpy(), hi),
                cB=_cumbest(g["pB"].to_numpy(), hi),
                cM=_cumbest(g["pM"].to_numpy(), hi))


def step_gains(s):


    prog = np.log10(s["cB"]) * (1 if s["hi"] else -1)
    steps = np.clip(np.diff(prog), 0, None)
    return s["yr"][1:], steps, float(prog[-1] - prog[0])
