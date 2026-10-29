from dataclasses import replace
import json
import queue
import numpy as np
import pytest
from matplotlib.figure import Figure
from dual_tail_characterization.experiment_config import ExperimentConfig, preset, capabilities, lane_preview
from dual_tail_characterization.history_endpoints import generate_history_endpoints
from dual_tail_characterization.experiment_backend import generate_experiment, prescribed_context_table
from dual_tail_characterization.experiment_plotting import PlotSettings, render_run
from dual_tail_characterization.run_controller import RunController, RunRecord
from dual_tail_characterization.experiment_export import export_run
from dual_tail_characterization.lane_benchmark import original_lane_records


def small(mode="shared_endpoints",n=100):
    raw=preset(mode).to_dict()
    raw["base"].update(n_symbols=n,samples_per_ui=32)
    return ExperimentConfig.parse(raw)


def record(snapshot,tmp_path,view="1UI pattern"):
    return RunRecord("test_run","token",snapshot,PlotSettings(view=view,time_bins_per_ui=16),str(tmp_path),"complete","2026-10-28T00:00:00+00:00",data=generate_experiment(snapshot))


@pytest.mark.parametrize("mode",["independent_lanes","shared_endpoints","history_endpoints","continuous_filter"])
def test_modes_reproducibility_and_validated_conversion(mode):
    snapshot=small(mode)
    assert ExperimentConfig.parse(snapshot.to_dict())==snapshot
    a,b=generate_experiment(snapshot),generate_experiment(snapshot)
    np.testing.assert_array_equal(a.arrays["segments"],b.arrays["segments"])
    assert not a.arrays["segments"].flags.writeable
    assert a.mode==mode
    assert (a.waveform is None)==(mode=="independent_lanes")
    assert a.tables


def test_original_preset_is_one_authoritative_copy_and_editable_duplicate():
    source=preset("independent_lanes")
    raw=source.to_dict()
    assert [{k:r[k] for k in ("start","end","mean_start","mean_end","sigma")} for r in raw["lanes"]]==original_lane_records()
    raw["lanes"][0]["mean_start"]=.4
    assert ExperimentConfig.parse(raw)!=source
    assert preset("independent_lanes")==source


def test_mode_editability_matches_generator_support():
    assert capabilities("independent_lanes")["lane_edit"]
    for mode in ("shared_endpoints","history_endpoints"):
        cap=capabilities(mode)
        assert not cap["lane_edit"] and not cap["channel_edit"] and not cap["additive_edit"]
        assert not cap["per_lane_timing_edit"]
    assert capabilities("history_endpoints")["contexts_edit"]
    assert capabilities("continuous_filter")["channel_edit"]
    rows=lane_preview(small())
    assert rows[3]["mean_start"]==0 and rows[3]["mean_end"]==3
    assert rows[3]["center"]==.5


@pytest.mark.parametrize("mutation,match",[
    (lambda r:r.update(mode="unknown"),"unsupported"),
    (lambda r:r["levels"][1].update(mean=0),"strictly increasing"),
    (lambda r:r["levels"][0].update(sigma=-1),"row 1"),
    (lambda r:r["base"].update(channel_bandwidth_hz=1e9),"unsupported"),
    (lambda r:r["contexts"].extend([dict(previous=0,current=1,next=2,mean=1.,sigma=None)]*2),"duplicate context"),
    (lambda r:r["incoming"].pop(),"exactly one row"),
    (lambda r:r["levels"][0].update(probability=.5),"sum to one"),
])
def test_invalid_inputs_fail_with_fields(mutation,match):
    raw=small("history_endpoints").to_dict()
    mutation(raw)
    with pytest.raises(ValueError,match=match):
        ExperimentConfig.parse(raw)


def test_independent_lanes_complete_pairs_and_noise_definition():
    raw=small("independent_lanes").to_dict()
    raw["lanes"].append(raw["lanes"][0])
    with pytest.raises(ValueError,match="duplicate ordered pair"):
        ExperimentConfig.parse(raw)
    raw=small("independent_lanes").to_dict()
    raw["lanes"][0]["weight"]+=.1
    with pytest.raises(ValueError,match="occurrence weights"):
        ExperimentConfig.parse(raw)


def test_history_rule_overrides_no_wrap_and_exact_shared_continuity():
    raw=small("history_endpoints",8).to_dict()
    sequence=[3,0,1,0,0,2,3,1]
    raw["base"].update(symbol_mode="explicit",sequence=sequence)
    raw["contexts"]=[dict(previous=3,current=0,next=1,mean=.7,sigma=0.)]
    snapshot=ExperimentConfig.parse(raw)
    result=generate_history_endpoints(snapshot)
    np.testing.assert_array_equal(result.contexts[0],[3,3,0])
    np.testing.assert_array_equal(result.contexts[-1],[3,1,1])
    assert result.endpoint_means[1]==.7 and result.endpoint_sigmas[1]==0
    assert result.waveform.symbol_endpoint_values[1]==.7
    raw_dict=snapshot.to_dict()
    incoming={(r["start"],r["end"]):r["mean"] for r in raw_dict["incoming"]}
    outgoing={(r["start"],r["end"]):r["mean"] for r in raw_dict["outgoing"]}
    assert result.endpoint_means[3]==(incoming[1,0]+outgoing[0,0])/2
    np.testing.assert_array_equal(result.waveform.final[::32],result.waveform.symbol_endpoint_values)
    assert np.all(np.diff(result.waveform.time)>0)
    assert len(prescribed_context_table(snapshot))==64


def test_history_lane_mixtures_use_observed_context_weights():
    raw=small("history_endpoints",17).to_dict()
    raw["base"].update(symbol_mode="explicit",sequence=[0,0,0,0,0,1,0,1,0,1,0,2,0,1,0,1,0])
    for row in raw["levels"]:
        row["sigma"]=0.
    snapshot=ExperimentConfig.parse(raw)
    data=generate_experiment(snapshot)
    rows=next(t for t in data.tables if t.source_csv=="lane_validation.csv")
    dictionaries=[dict(zip(rows.columns,r)) for r in rows.rows]
    # With zero endpoint fluctuations, theoretical mixture moments (actual context
    # occurrence weights) exactly describe the empirical sample means.
    for row in dictionaries:
        if row["quantity"]=="mean" and row["empirical"] is not None:
            assert row["empirical"]==pytest.approx(row["theoretical"],abs=1e-14)
    components=next(t for t in data.tables if t.source_csv=="lane_components.csv")
    weights=[dict(zip(components.columns,r))["weight"] for r in components.rows]
    assert any(0<p<1 for p in weights)


def test_plot_changes_do_not_regenerate_or_change_counts_population(tmp_path):
    r=record(small(),tmp_path)
    original=r.data.waveform.final.copy()
    figure=Figure()
    a=render_run(figure,r,replace(r.plot_settings,max_traces=1))
    b=render_run(figure,r,replace(r.plot_settings,max_traces=99,normalization="linear",color="color"))
    np.testing.assert_array_equal(a["counts"],b["counts"])
    np.testing.assert_array_equal(original,r.data.waveform.final)
    assert a["counts"].sum()==r.data.arrays["segments"].size


def test_independent_two_ui_is_unavailable_not_fabricated(tmp_path):
    r=record(small("independent_lanes"),tmp_path)
    figure=Figure()
    output=render_run(figure,r,replace(r.plot_settings,view="2UI eye"))
    assert output=={}
    assert "Unavailable" in figure.axes[0].texts[0].get_text()


def test_controller_isolation_stale_and_late_result_handling(tmp_path):
    controller=RunController()
    try:
        snapshot=small()
        r=controller.start(snapshot,PlotSettings(time_bins_per_ui=16),tmp_path)
        raw=snapshot.to_dict()
        raw["levels"][0]["sigma"]+=.1
        newer=ExperimentConfig.parse(raw)
        controller.mark_inputs(newer)
        assert controller.is_stale(r)
        while controller.busy:
            controller.accept(controller.events.get(timeout=20))
        completed=controller.records[r.run_id]
        assert completed.snapshot==snapshot
        assert completed.data is not None
        assert not controller.accept((r.run_id,"wrong_token","generated",None))
        controller.remove(r.run_id)
        assert not controller.accept((r.run_id,r.token,"generated",completed.data))
        controller.close()
        assert not controller.accept((r.run_id,r.token,"generated",completed.data))
    finally:
        controller.close()


@pytest.mark.parametrize("mode",["independent_lanes","history_endpoints","continuous_filter"])
def test_export_stored_selected_run_complete_workbook(mode,tmp_path):
    r=record(small(mode,80),tmp_path)
    export_run(r)
    raw=json.loads((tmp_path/"gui_config.json").read_text())
    assert raw==r.snapshot.to_dict()
    mapping=json.loads((tmp_path/"workbook_mapping.json").read_text())
    assert {m["source_csv"] for m in mapping}=={p.name for p in tmp_path.glob("*.csv")}
    assert (tmp_path/"validation_results.xlsx").is_file()
    with np.load(tmp_path/"waveform.npz",allow_pickle=False) as archive:
        if mode=="history_endpoints":
            np.testing.assert_array_equal(archive["experiment_contexts"],r.data.arrays["contexts"])
        if mode=="independent_lanes":
            assert "segments" in archive and "time" not in archive
