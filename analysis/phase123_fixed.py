import numpy as np
import pandas as pd
import phase123 as p


def numeric_fixed(s: pd.Series) -> pd.Series:
    """Normalize the pandas/SAS-XPORT tiny sentinel used when numeric zero is decoded."""
    x = pd.to_numeric(s, errors="coerce")
    return x.mask(x.notna() & (x.abs() < 1e-50), 0.0)


p.numeric = numeric_fixed
p.main()
