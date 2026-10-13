"""Column-name constants shared across modules (data columns are always referenced by name)."""

INSTRUMENT = "instrument_id"
DESK = "desk"
RISK_CLASS = "risk_class"
MEASURE = "measure"
BUCKET = "bucket"
RISK_FACTOR = "risk_factor"
TENOR = "tenor"
CURVE = "curve"
SENSITIVITY = "sensitivity"
GAMMA = "gamma"
CVR_UP = "cvr_up"
CVR_DOWN = "cvr_down"

# Risk classes
GIRR = "GIRR"
FX = "FX"
EQ = "EQ"
CSR = "CSR"

# Measures
DELTA = "delta"
VEGA = "vega"
CURVATURE = "curvature"
DRC = "DRC"
RRAO = "RRAO"

# DRC position table
OBLIGOR = "obligor"
SENIORITY = "seniority"
RATING = "rating"
DRC_BUCKET = "drc_bucket"
NOTIONAL = "notional"
PNL_TO_DATE = "pnl_to_date"
LGD = "lgd"
MATURITY = "maturity"
