import numpy as np
import pandas as pd
from astropy.io import fits

from src.data import download_kepler as dk


def write_lc(path):
    cols = fits.ColDefs([
        fits.Column(name="TIME", format="D", array=np.arange(100.0)),
        fits.Column(name="PDCSAP_FLUX", format="E", array=np.ones(100)),
    ])
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")]
                 ).writeto(path)


def test_quarantine_unreadable_sets_aside_truncated_files(tmp_path):
    good, bad = tmp_path / "a_llc.fits", tmp_path / "sub" / "b_llc.fits"
    bad.parent.mkdir()
    write_lc(good)
    write_lc(bad)
    bad.write_bytes(bad.read_bytes()[:3000])  # cut mid-table, like an interrupted download
    assert dk.quarantine_unreadable(tmp_path) == [bad]
    assert good.exists() and not bad.exists()
    assert (bad.parent / "b_llc.fits.partial").exists()
    assert dk.quarantine_unreadable(tmp_path) == []


def test_saved_koi_table_reproduces_target_selection(tmp_path):
    table = pd.DataFrame({
        "kepid": [1, 1, 2, 3, 4, 5],
        "kepoi_name": ["K1.01", "K1.02", "K2.01", "K3.01", "K4.01", "K5.01"],
        "koi_disposition": ["CONFIRMED", "FALSE POSITIVE", "FALSE POSITIVE", "CANDIDATE",
                            "CONFIRMED", "FALSE POSITIVE"],
    })
    path = tmp_path / "kois.csv"
    table.to_csv(path, index=False)
    labels = dk.load_koi_table(path, ("CONFIRMED", "FALSE POSITIVE"))
    assert "K3.01" not in set(labels.kepoi_name)
    assert labels.set_index("kepoi_name").label.to_dict() == {
        "K1.01": 1, "K1.02": 0, "K2.01": 0, "K4.01": 1, "K5.01": 0}
    first = dk.select_targets(labels, 1, 42)
    assert first.equals(dk.select_targets(labels, 1, 42))
    # A star with any confirmed KOI counts as a confirmed host.
    assert set(dk.select_targets(labels, None, 42).query("kepid == 1").koi_disposition) == {
        "CONFIRMED"}


def test_cli_accepts_offline_options():
    args = dk.parse_args(["--koi-table", "x.csv", "--system-certs", "--limit", "100"])
    assert args.koi_table.name == "x.csv" and args.system_certs and args.limit == 100
