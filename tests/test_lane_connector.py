import json
import numpy as np
import pytest
from dual_tail_characterization.lane_connector import connector, midpoint_derivative, solve_kappa
from dual_tail_characterization.shared_endpoints import logistic_weight
from dual_tail_characterization.experiment_config import preset, ExperimentConfig
from dual_tail_characterization.lane_parameters import LaneDraft, effective_parameters
from dual_tail_characterization.experiment_backend import generate_experiment
from dual_tail_characterization.eyes import paired_eyes

@pytest.mark.parametrize("k",[1e-9,.01,3.,22.,100.])
@pytest.mark.parametrize("c",[.1,.46,.5,.54,.9])
def test_connector(k,c):
    assert np.array_equal(connector([0,c,1],k,c),[0,.5,1])
    assert np.all(np.diff(connector(np.linspace(0,1,1001),k,c))>=0)
    eps=1e-7
    numerical=(connector(c+eps,k,c)-connector(c-eps,k,c))/(2*eps)
    assert numerical==pytest.approx(midpoint_derivative(k,c),rel=2e-7)
    np.testing.assert_allclose(connector(np.linspace(0,1),k,.5),logistic_weight(np.linspace(0,1),k),atol=1e-14)

def test_inversion():
    for c in [.3,.5,.7]:
        for k in [.1,4,22,100]:
            assert solve_kappa(3*midpoint_derivative(k,c),3,c)==pytest.approx(k,rel=1e-8)
    for slope,delta,c in [(1,1,.5),(2,0,.5),(10,1,0),(float("nan"),1,.5)]:
        with pytest.raises(ValueError):solve_kappa(slope,delta,c)
    for k,c in [(0,.5),(1,0),(1,1),(1e7,.5)]:
        with pytest.raises(ValueError):connector(.5,k,c)

def edited(mode="shared_endpoints"):
    raw=preset(mode).to_dict();raw["base"].update(n_symbols=300,samples_per_ui=32)
    snapshot=ExperimentConfig.parse(raw);draft=LaneDraft(snapshot)
    draft.rows[3].update(kappa=12,phase_shift=.04)
    draft.rows[12].update(kappa=35,phase_shift=-.04)
    return snapshot,draft

@pytest.mark.parametrize("mode",["shared_endpoints","history_endpoints","independent_lanes"])
def test_draft_roundtrip_continuity(mode):
    old,draft=edited(mode)
    assert "lane_overrides" not in old.to_dict() # Cancel/discard cannot mutate source
    new=draft.apply()
    assert ExperimentConfig.parse(json.loads(new.snapshot_json))==new
    assert new!=old
    result=generate_experiment(new)
    if result.waveform is not None:
        w=result.waveform
        np.testing.assert_array_equal(w.final[::32],w.symbol_endpoint_values)
        one,two=paired_eyes(w)
        np.testing.assert_array_equal(one.traces,two.traces[:,:33])
    table=next(t for t in result.tables if t.source_csv=="effective_lane_parameters.csv")
    assert len(table.rows)==16
    assert "nominal_slope_ui" in table.columns

def test_invalid_atomic():
    old,draft=edited()
    draft.rows[0]["enabled"]=False
    with pytest.raises(ValueError,match="Disabled"):draft.apply()
    assert "lane_overrides" not in old.to_dict()
    old,draft=edited();draft.rows[0]["mean_start"]=99
    with pytest.raises(ValueError,match="read-only"):draft.apply()


def test_effective_csv_excel(tmp_path):
    from dual_tail_characterization.run_controller import RunRecord, now
    from dual_tail_characterization.experiment_plotting import PlotSettings
    from dual_tail_characterization.experiment_export import export_run
    from openpyxl import load_workbook
    import csv
    _,draft=edited();snapshot=draft.apply()
    data=generate_experiment(snapshot)
    record=RunRecord("test","token",snapshot,PlotSettings(time_bins_per_ui=16),str(tmp_path),"complete",now(),now(),data)
    export_run(record)
    with open(tmp_path/"effective_lane_parameters.csv",newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    row=next(r for r in rows if r["start"]=="0" and r["end"]=="3")
    assert float(row["kappa"])==12 and float(row["phase_shift"])==.04
    book=load_workbook(tmp_path/"validation_results.xlsx",data_only=True)
    sheet=book["Effective_Lane_Parameters"]
    headers=[c.value for c in sheet[1]]
    values=list(sheet.values)[1:]
    i,j=headers.index("kappa"),headers.index("phase_shift")
    assert any(r[i]==12 and r[j]==.04 for r in values)
    book.close()
