"""
Downloads SPY daily OHLCV from yfinance and writes it into LEAN's expected
data format: data/equity/usa/daily/spy.zip
"""
import io
import zipfile
from pathlib import Path
import yfinance as yf

TICKER = "SPY"
START = "2018-01-01"
END = "2024-12-31"
OUT_PATH = Path(__file__).parent / "data" / "equity" / "usa" / "daily" / "spy.zip"


def main():
    print(f"Downloading {TICKER} {START} to {END} from yfinance...")
    df = yf.download(TICKER, start=START, end=END, auto_adjust=False, progress=False)

    if df.empty:
        raise RuntimeError("yfinance returned no data. Check your internet connection.")

    # LEAN daily format: Date,Open,High,Low,Close,Volume
    # Prices are scaled ×10000 (integer). Date is YYYYMMDD HH:MM (00:00).
    rows = []
    for dt, row in df.iterrows():
        date_str = dt.strftime("%Y%m%d 00:00")
        o = int(float(row["Open"].iloc[0]) * 10000)
        h = int(float(row["High"].iloc[0]) * 10000)
        l = int(float(row["Low"].iloc[0]) * 10000)
        c = int(float(row["Close"].iloc[0]) * 10000)
        v = int(float(row["Volume"].iloc[0]))
        rows.append(f"{date_str},{o},{h},{l},{c},{v}")

    csv_content = "\n".join(rows) + "\n"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{TICKER.lower()}.csv", csv_content)

    print(f"Written {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
