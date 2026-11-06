"""Reproducible common and lane-specific midpoint demonstration."""
from pathlib import Path
from dataclasses import replace
from dual_tail_characterization.experiment_config import preset, ExperimentConfig
from dual_tail_characterization.lane_parameters import LaneDraft
from dual_tail_characterization.experiment_backend import generate_experiment
from dual_tail_characterization.experiment_export import export_run
from dual_tail_characterization.run_controller import RunRecord, now
from dual_tail_characterization.experiment_plotting import PlotSettings, render_run
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

root=Path("runs/lane_connector_demo")
raw=preset().to_dict();raw["base"].update(n_symbols=3000,samples_per_ui=128)
base=ExperimentConfig.parse(raw)
fig=Figure(figsize=(12,5));FigureCanvasAgg(fig)
for name,modified in (("common",False),("lane_specific",True)):
    draft=LaneDraft(base)
    if modified:
        draft.rows[3].update(kappa=12,phase_shift=.04)
        draft.rows[12].update(kappa=35,phase_shift=-.04)
    snapshot=draft.apply();data=generate_experiment(snapshot)
    record=RunRecord(name,name,snapshot,PlotSettings(style="traces",selection="lane",lane_start=0,lane_end=3),str(root/name),"complete",now(),now(),data)
    export_run(record)
    # Full export preserves counts; this compact comparison isolates nominal 0->3 and 3->0 connectors.
    ax=fig.add_subplot(1,2,1 if not modified else 2)
    from dual_tail_characterization.lane_parameters import effective_parameters
    from dual_tail_characterization.lane_connector import connector
    import numpy as np
    u=np.linspace(0,1,1001)
    for r in effective_parameters(snapshot):
        if (r["start"],r["end"]) in ((0,3),(3,0)):
            ax.plot(u,r["mean_start"]+(r["mean_end"]-r["mean_start"])*connector(u,r["kappa"],r["center"]),label=f'{r["start"]}->{r["end"]}, k={r["kappa"]:g}, c={r["center"]:g}')
            ax.axvline(r["center"],color="orange",ls=":",lw=.7)
    ax.axvline(.5,color="gray",ls="--");ax.set(xlim=(0,1),ylim=(-.2,3.2),xlabel="Segment phase [UI]",ylabel="Amplitude",title=name);ax.legend()
fig.tight_layout();fig.savefig(root/"connector_comparison.png",dpi=150)
print(root.resolve())
