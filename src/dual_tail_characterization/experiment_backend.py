"""Numerical run service and validation, independent of Tk and Matplotlib."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from scipy.special import expit
from .experiment_config import ExperimentConfig
from .waveform import generate_waveform
from .history_endpoints import generate_history_endpoints
from .eyes import extract_eye
from .shared_endpoints import logistic_weight
from .level_analysis import Components, LevelContext, Reference, analyze_level, component_records
from .validation_workbook import RunTable


@dataclass(frozen=True)
class TableData:
    source_csv: str
    sheet_name: str
    columns: tuple
    rows: tuple
    units: tuple = ()
    kind: str = "empirical"

    @classmethod
    def records(cls, source, sheet, records, *, units=None, kind="empirical"):
        table = RunTable.from_records(source,sheet,records,units=units,kind=kind)
        return cls(table.source_csv,table.sheet_name,tuple(table.columns),tuple(table.rows),tuple(table.units.items()),kind)

    def export_table(self):
        return RunTable(self.source_csv,self.sheet_name,list(self.columns),list(self.rows),dict(self.units),self.kind)


@dataclass(frozen=True)
class RunData:
    mode: str
    waveform: object
    arrays: object
    tables: tuple
    summary: str


def lane_weight(u,kappa,center=.5):
    u = np.asarray(u)
    if kappa < 1e-6:
        return u.copy()
    first,last = expit(-kappa*center),expit(kappa*(1-center))
    return np.where(u==0,0.,np.where(u==1,1.,(expit(kappa*(u-center))-first)/(last-first)))


def independent_population(snapshot):
    raw,c = snapshot.to_dict(),snapshot.config
    lanes = raw["lanes"]
    symbol_rng,fluctuation_rng,noise_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(3)]
    selected = symbol_rng.choice(len(lanes),c.n_symbols,p=[r["weight"] for r in lanes])
    u = np.arange(c.samples_per_ui+1)/c.samples_per_ui
    traces = np.empty((c.n_symbols,len(u)))
    labels = np.array([(lanes[k]["start"],lanes[k]["end"]) for k in selected],dtype=int)
    z = fluctuation_rng.standard_normal((c.n_symbols,1 if raw["noise_model"]=="trace_offset" else len(u)))
    for k,row in enumerate(lanes):
        mask = selected==k
        weight = lane_weight(u,row["kappa"],row["center"])
        traces[mask] = row["mean_start"] + weight*(row["mean_end"]-row["mean_start"]) + row["sigma"]*z[mask]
    traces += noise_rng.normal(0,c.noise_std,traces.shape)
    return dict(segments=traces,segment_labels=labels,lane_component_ids=selected,
                independent_phase_ui=u,independent_fluctuations=z)


def phase_samples(segments, phase):
    position = phase*(segments.shape[1]-1)
    left,right = int(np.floor(position)),int(np.ceil(position))
    return (1-(position-left))*segments[:,left] + (position-left)*segments[:,right]


def _stats(values):
    return dict(sample_count=len(values),mean=float(np.mean(values)) if len(values) else None,
                sigma=float(np.std(values,ddof=1)) if len(values)>1 else None,
                variance=float(np.var(values,ddof=1)) if len(values)>1 else None)


def _component_mixture(keys,means,sigmas):
    unique,inverse,counts = np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    first = [np.flatnonzero(inverse==k)[0] for k in range(len(unique))]
    return unique,counts/counts.sum(),means[first],sigmas[first],counts


def validation_tables(snapshot,w,arrays):
    raw,c = snapshot.to_dict(),snapshot.config
    mode,unit = snapshot.mode,c.amplitude_unit
    segments,labels = arrays["segments"],arrays["segment_labels"]
    lane_rows,component_rows,level_rows,diagnostics,status_rows = [],[],[],[],[]
    level_components = []
    lane_lookup = {(r["start"],r["end"]):r for r in raw["lanes"]}
    effective_rows=[]
    if mode=="history_endpoints":
        contexts=arrays["contexts"]
        unique,counts=np.unique(contexts,axis=0,return_counts=True)
        for key,count in zip(unique,counts):
            index=np.flatnonzero(np.all(contexts==key,axis=1))[0]
            effective_rows.append(dict(previous=int(key[0]),current=int(key[1]),next=int(key[2]),
                prescribed_mean=float(arrays["endpoint_means"][index]),prescribed_sigma=float(arrays["endpoint_sigmas"][index]),
                observed_count=int(count),population="retained symbol occurrences; context not amplitude classified"))
    for a in range(len(c.levels)):
        for b in range(len(c.levels)):
            mask=(labels[:,0]==a)&(labels[:,1]==b)
            indices=np.flatnonzero(mask)
            for phase in (0.,.5,1.):
                values=phase_samples(segments[mask],phase)
                empirical=_stats(values)
                mean=variance=sigma=None
                condition=f"lane {a}->{b}; u={phase:g} within center-to-center segment"
                reference_status="unavailable: continuous filtering has no prescribed conditional Gaussian mixture"
                if mode=="independent_lanes":
                    lane=lane_lookup[a,b]
                    weight=float(np.interp(phase,np.arange(c.samples_per_ui+1)/c.samples_per_ui,
                                           lane_weight(np.arange(c.samples_per_ui+1)/c.samples_per_ui,lane["kappa"],lane["center"])))
                    mean=lane["mean_start"]+weight*(lane["mean_end"]-lane["mean_start"])
                    # Interpolated off-grid sample-independent noise has reduced variance.
                    x=phase*c.samples_per_ui
                    fraction=x-np.floor(x)
                    factor=(1-fraction)**2+fraction**2
                    variance=lane["sigma"]**2*(1 if raw["noise_model"]=="trace_offset" else factor)+c.noise_std**2*factor
                    reference_status=f"independent lane, {raw['noise_model']}; additive noise included; one observation/realization/phase"
                elif mode in ("shared_endpoints","history_endpoints") and len(indices):
                    weight=float(np.interp(phase,np.arange(c.samples_per_ui+1)/c.samples_per_ui,logistic_weight(np.arange(c.samples_per_ui+1)/c.samples_per_ui,w.metadata["effective_kappa"])))
                    endpoint_means=arrays["endpoint_means"]
                    endpoint_sigmas=arrays["endpoint_sigmas"]
                    means=(1-weight)*endpoint_means[indices]+weight*endpoint_means[indices+1]
                    variances=(1-weight)**2*endpoint_sigmas[indices]**2+weight**2*endpoint_sigmas[indices+1]**2
                    keys=(np.concatenate((arrays["contexts"][indices],arrays["contexts"][indices+1]),axis=1)
                          if mode=="history_endpoints" else np.zeros((len(indices),1),dtype=int))
                    unique,weights,mus,sigmas,counts=_component_mixture(keys,means,np.sqrt(variances))
                    mean=float(np.dot(weights,mus))
                    variance=float(np.dot(weights,sigmas**2+(mus-mean)**2))
                    reference_status="actual observed context-pair weights; independent endpoint Z; sampled connector; no channel/additive noise"
                    for k,(key,p,mu,sd,count) in enumerate(zip(unique,weights,mus,sigmas,counts)):
                        component_rows.append(dict(population="conditional lane",lane_id=f"{a}->{b}",phase_u=phase,
                            component_id=str(k),context_pair="/".join(map(str,key)),weight=float(p),mean=float(mu),sigma=float(sd),observed_count=int(count)))
                if variance is not None:
                    sigma=float(np.sqrt(variance))
                for quantity,theory in (("mean",mean),("sigma",sigma),("variance",variance)):
                    measured=empirical[quantity]
                    lane_rows.append(dict(population=condition,quantity=quantity,prescribed=None,theoretical=theory,
                        empirical=measured,difference=measured-theory if measured is not None and theory is not None else None,
                        unit=f"{unit}**2" if quantity=="variance" else unit,sample_count=len(values),
                        status="not_observed" if not len(values) else "available" if measured is not None else "insufficient_samples",
                        reference_status=reference_status))
    for level in range(len(c.levels)):
        if mode=="independent_lanes":
            mask=labels[:,0]==level
            values=segments[mask,0]
            population="independent-lane start samples conditioned on start level; outgoing lanes pooled, not component sigma"
            present=[r for r in raw["lanes"] if r["start"]==level and np.any(np.all(labels==[level,r["end"]],axis=1))]
            if present:
                weights=np.array([np.sum(np.all(labels==[level,r["end"]],axis=1)) for r in present],dtype=float)
                weights/=weights.sum()
                parameters=Components(weights,[r["mean_start"] for r in present],[np.hypot(r["sigma"],c.noise_std) for r in present])
            else:
                parameters=None
        else:
            mask=w.symbol_indices==level
            values=arrays["level_samples"][mask]
            population="symbol-center amplitudes at 0.5 UI after nominal boundary; actual symbol labels"
            if mode=="continuous_filter":
                parameters=None
            else:
                keys=arrays["contexts"][mask] if mode=="history_endpoints" else np.zeros((mask.sum(),1),dtype=int)
                if mask.any():
                    _,weights,means,sigmas,_=_component_mixture(keys,arrays["endpoint_means"][mask],arrays["endpoint_sigmas"][mask])
                    parameters=Components(weights,means,sigmas)
                else:
                    parameters=None
        sample=_stats(values)
        for quantity in ("mean","sigma","variance"):
            level_rows.append(dict(level_id=str(level),population=population,quantity=quantity,
                prescribed=c.levels[level] if quantity=="mean" else None,theoretical=None,empirical=sample[quantity],difference=None,
                unit=f"{unit}**2" if quantity=="variance" else unit,sampling_phase_ui=.5 if mode!="independent_lanes" else 0.,
                sample_count=len(values),status="available" if sample[quantity] is not None else "insufficient_samples",
                note="nominal mean is a reference, not a claimed post-filter prediction; pooled sigma is not per-component sigma"))
        if parameters is not None:
            context=LevelContext("gui_run",str(level),unit,.5 if mode!="independent_lanes" else 0.)
            diagnostics.append(analyze_level(parameters,context,Reference(c.levels[level],reference_kind="configured_nominal")))
            level_components.extend(component_records(parameters,context))
    input_effective=[dict(quantity=k,input=v,effective=v,unit="see configuration",status="configured") for k,v in c.to_dict().items()]
    input_effective.append(dict(quantity="generator",input=mode,effective=mode,unit=None,status="explicit dispatch"))
    input_effective.append(dict(quantity="endpoint rule",input="incoming/outgoing average with explicit overrides" if mode=="history_endpoints" else mode,
        effective="controlled history model; context means differ from effective lane statistics" if mode=="history_endpoints" else mode,unit=None,status="available"))
    status_rows.extend([dict(quantity="generation",status="complete",value="no amplitude clipping"),
        dict(quantity="validation population",status="available",value="all eligible observations, independent of displayed trace limit"),
        dict(quantity="GMM fitting / EH / EW / BER",status="unavailable",value=None),
        dict(quantity="internal variance diagnostics",status="available" if diagnostics else "unavailable",value="prescribed mixture parameters, actual occurrence weights" if diagnostics else None)])
    tables=[TableData.records("input_effective.csv","Input_Effective",input_effective,kind="input/effective"),
            TableData.records("lane_validation.csv","Lane_Validation",lane_rows),
            TableData.records("level_statistics.csv","Level_Statistics",level_rows),
            TableData.records("validation_status.csv","Validation_Status",status_rows)]
    if component_rows:
        tables.append(TableData.records("lane_components.csv","Lane_Components",component_rows,kind="prescribed, actual conditional context weights"))
    if level_components:
        tables.append(TableData.records("level_components.csv","Component_Parameters",level_components,kind="prescribed, actual occurrence weights"))
    if diagnostics:
        tables.append(TableData.records("internal_variance.csv","Internal_Variance",diagnostics,kind="internal whole-level prescribed diagnostics"))
    if effective_rows:
        tables.append(TableData.records("observed_contexts.csv","Observed_Contexts",effective_rows,kind="prescribed context values / empirical occurrence counts"))
    return tuple(tables)


def generate_experiment(snapshot, progress=lambda message:None):
    progress(f"Generating {snapshot.mode}")
    c=snapshot.config
    w=None
    if snapshot.mode=="independent_lanes":
        arrays=independent_population(snapshot)
    else:
        if snapshot.mode=="history_endpoints":
            history=generate_history_endpoints(snapshot)
            w=history.waveform
        elif snapshot.mode in ("shared_endpoints","continuous_filter"):
            w=generate_waveform(c)
        else:
            raise ValueError("unsupported generator")
        eye=extract_eye(w)
        starts=eye.clock_start_symbols-1
        arrays=dict(segments=eye.traces,segment_labels=np.stack((w.symbol_indices[starts],w.symbol_indices[starts+1]),axis=1),
                    segment_start_symbols=starts,level_samples=np.interp((np.arange(c.n_symbols)+.5)*c.symbol_period,w.time,w.final))
        if snapshot.mode=="history_endpoints":
            arrays.update(contexts=history.contexts,endpoint_means=history.endpoint_means,endpoint_sigmas=history.endpoint_sigmas,context_guards=history.guards)
        elif snapshot.mode=="shared_endpoints":
            arrays.update(endpoint_means=w.symbol_amplitudes.copy(),endpoint_sigmas=np.asarray(c.endpoint_sigmas or [0.]*len(c.levels))[w.symbol_indices])
    progress("Computing full-population validation tables")
    tables=validation_tables(snapshot,w,arrays)
    for array in arrays.values():
        array.flags.writeable=False
    if w is not None:
        for value in vars(w).values():
            if isinstance(value,np.ndarray):
                value.flags.writeable=False
    return RunData(snapshot.mode,w,MappingProxyType(arrays),tables,
        f"{c.n_symbols:,} {'independent realizations' if w is None else 'retained symbols'}; {len(arrays['segments']):,} eligible 1UI observations; seed {c.seed}. "
        + ("Independent finite lanes: no continuous waveform or 2UI view." if w is None else "Continuous source waveform; shared joins retained."))


def prescribed_context_table(snapshot):
    from .history_endpoints import context_parameters
    n=len(snapshot.config.levels)
    keys=np.array([(a,b,c) for a in range(n) for b in range(n) for c in range(n)])
    means,sigmas=context_parameters(snapshot,*keys.T)
    overrides={(r["previous"],r["current"],r["next"]) for r in snapshot.to_dict()["contexts"]}
    return [dict(previous=int(key[0]),current=int(key[1]),next=int(key[2]),mean=float(mean),sigma=float(sigma),
                 source="explicit override" if tuple(key) in overrides else "incoming/outgoing average",
                 population_status="prescribed context; not an effective two-symbol lane statistic")
            for key,mean,sigma in zip(keys,means,sigmas)]
