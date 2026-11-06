"""Canonical lane records shared by the editor, generator and exports."""
import copy
import math
from .lane_connector import validate, midpoint_derivative, solve_kappa
from .shared_endpoints import kappa_for_duration

def effective_parameters(snapshot):
    raw,c=snapshot.to_dict(),snapshot.config
    overrides={(r["start"],r["end"]):r for r in raw.get("lane_overrides",[])}
    independent={(r["start"],r["end"]):r for r in raw["lanes"]}
    rows=[]
    for a in range(len(c.levels)):
        for b in range(len(c.levels)):
            legacy=independent[a,b]
            r=dict(enabled=True,start=a,end=b,mean_start=c.levels[a],mean_end=c.levels[b],
                   sigma_start=raw["levels"][a]["sigma"],sigma_end=raw["levels"][b]["sigma"],
                   kappa=c.shared_kappa or kappa_for_duration(min(c.transition_10_90_ui,.799999)),
                   phase_shift=0.,slope_mode="kappa",slope_magnitude=None,weight="sequence derived")
            if snapshot.mode=="independent_lanes":
                r.update(mean_start=legacy["mean_start"],mean_end=legacy["mean_end"],sigma_start=legacy["sigma"],sigma_end=legacy["sigma"],kappa=legacy["kappa"],phase_shift=legacy["center"]-.5,weight=legacy["weight"])
            if snapshot.mode=="history_endpoints" and not raw.get("pattern"):
                import numpy as np
                from .history_endpoints import context_parameters
                if c.symbol_mode=="random":
                    ids=np.arange(len(c.levels));weights=np.asarray(c.probabilities)
                    means0,sigmas0=context_parameters(snapshot,ids,np.full(len(ids),a),np.full(len(ids),b))
                    means1,sigmas1=context_parameters(snapshot,np.full(len(ids),a),np.full(len(ids),b),ids)
                else:
                    seq=np.asarray(c.sequence);extended=np.r_[seq[0],seq,seq[-1]]
                    if raw.get("pattern"):
                        extended=np.r_[result.before[-1],seq,result.after[0]]
                    contexts=np.stack((extended[:-2],extended[1:-1],extended[2:]),axis=1)
                    idx=np.flatnonzero((seq[:-1]==a)&(seq[1:]==b))
                    weights=np.full(len(idx),1/len(idx)) if len(idx) else np.array([])
                    means0,sigmas0=context_parameters(snapshot,*contexts[idx].T)
                    means1,sigmas1=context_parameters(snapshot,*contexts[idx+1].T)
                if len(weights):
                    for side,means,sigmas in (("start",means0,sigmas0),("end",means1,sigmas1)):
                        mean=float(np.dot(weights,means))
                        r["mean_"+side]=mean
                        r["sigma_"+side]=float(np.sqrt(np.dot(weights,sigmas**2+(means-mean)**2)))
            r.update(overrides.get((a,b),{}))
            r["center"]=.5+r["phase_shift"]
            r["nominal_delta"]=c.levels[b]-c.levels[a] if snapshot.mode=="history_endpoints" else r["mean_end"]-r["mean_start"]
            r["nominal_slope_ui"]=r["nominal_delta"]*midpoint_derivative(r["kappa"],r["center"])
            r["nominal_slope_per_second"]=r["nominal_slope_ui"]/c.symbol_period
            r["status"]=("ordered-pattern context summaries available after generation; displayed endpoints are nominal references" if snapshot.mode=="history_endpoints" and raw.get("pattern") else "context-dependent endpoints; nominal reference only" if snapshot.mode=="history_endpoints" else "nominal amplitude midpoint undefined" if r["mean_start"]==r["mean_end"] else "valid nominal reference")
            r["amplitude_unit"]=c.amplitude_unit
            r["phase_unit"]="UI"
            r["slope_unit"]=c.amplitude_unit+"/UI"
            r["analytic_midpoint_mean"]=(r["mean_start"]+r["mean_end"])/2 if snapshot.mode=="shared_endpoints" else None
            r["analytic_midpoint_variance"]=(r["sigma_start"]**2+r["sigma_end"]**2)/4 if snapshot.mode=="shared_endpoints" else None
            r["analytic_reference_status"]="independent endpoint pre-impairment continuum reference" if snapshot.mode=="shared_endpoints" else "unavailable as single independent endpoint model; see empirical/context tables"
            r["sigma"]=r["sigma_start"]
            rows.append(r)
    return rows

FIELDS=("enabled","start","end","mean_start","mean_end","sigma_start","sigma_end","kappa","phase_shift","slope_mode","slope_magnitude","weight")

def validate_overrides(raw,rows):
    from .experiment_config import ExperimentConfig
    if raw["mode"]=="continuous_filter":
        raise ValueError("Continuous filter does not support per-lane connectors")
    base=ExperimentConfig.parse({k:v for k,v in raw.items() if k!="lane_overrides"})
    nominal={(r["start"],r["end"]):r for r in effective_parameters(base)}
    seen=set(); result=[]
    for source in rows:
        r={k:source[k] for k in FIELDS}
        pair=(int(r["start"]),int(r["end"]))
        if pair not in nominal or pair in seen or pair!=(r["start"],r["end"]):
            raise ValueError("Invalid or duplicate fixed lane IDs")
        seen.add(pair)
        if r["enabled"] not in (True,1,"True","true"):
            raise ValueError("Disabled lanes are unsupported; edit sequence probabilities instead")
        r["enabled"]=True
        for k in ("mean_start","mean_end","sigma_start","sigma_end","kappa","phase_shift"):
            r[k]=float(r[k])
            if not math.isfinite(r[k]): raise ValueError(f"{k} must be finite")
        if min(r["sigma_start"],r["sigma_end"])<0: raise ValueError("Sigmas must be nonnegative")
        if raw["mode"]!="independent_lanes":
            for k in ("mean_start","mean_end","sigma_start","sigma_end","weight"):
                if r[k]!=nominal[pair][k]: raise ValueError(f"{k}: derived read-only in continuous modes")
        else:
            r["weight"]=float(r["weight"])
            if not math.isfinite(r["weight"]) or r["weight"]<0: raise ValueError("Invalid occurrence weight")
        if r["slope_mode"]=="slope":
            r["slope_magnitude"]=float(r["slope_magnitude"])
            r["kappa"]=solve_kappa(r["slope_magnitude"],(base.config.levels[pair[1]]-base.config.levels[pair[0]]) if raw["mode"]=="history_endpoints" else r["mean_end"]-r["mean_start"],.5+r["phase_shift"])
        elif r["slope_mode"]!="kappa": raise ValueError("Slope mode must be kappa or slope")
        else: r["slope_magnitude"]=None
        validate(r["kappa"],.5+r["phase_shift"])
        result.append(r)
    if seen!=set(nominal): raise ValueError("Exactly n*n lane rows required")
    if raw["mode"]=="independent_lanes" and abs(sum(r["weight"] for r in result)-1)>1e-12:
        raise ValueError("Occurrence weights must sum to one")
    return result

class LaneDraft:
    def __init__(self,snapshot):
        self.snapshot=snapshot
        self.rows=copy.deepcopy(effective_parameters(snapshot))
    def apply(self):
        from .experiment_config import ExperimentConfig
        raw=self.snapshot.to_dict();raw["lane_overrides"]=self.rows
        return ExperimentConfig.parse(raw)


def update_shared_levels(snapshot, updates):
    """Edit global level endpoints while retaining per-lane connectors and reuse."""
    from .experiment_config import ExperimentConfig, number
    if snapshot.mode not in ("shared_endpoints","history_endpoints"):
        raise ValueError("Shared level editing requires a shared endpoint mode")
    raw=snapshot.to_dict();overrides=raw.pop("lane_overrides",None)
    for level_id,changes in updates.items():
        if not isinstance(level_id,int) or not 0<=level_id<len(raw["levels"]):
            raise ValueError("Invalid level ID")
        for key,value in changes.items():
            if key not in ("mean","sigma"):raise ValueError("Only mean and sigma can be edited here")
            raw["levels"][level_id][key]=number(value,f"level {level_id} {key}")
    base=ExperimentConfig.parse(raw)
    if overrides:
        derived={(r["start"],r["end"]):r for r in effective_parameters(base)}
        for row in overrides:
            for key in ("mean_start","mean_end","sigma_start","sigma_end","weight"):
                row[key]=derived[row["start"],row["end"]][key]
        raw=base.to_dict();raw["lane_overrides"]=overrides
        return ExperimentConfig.parse(raw)
    return base


def independent_copy(snapshot):
    """Explicit copy into finite independent lanes; never join these as a waveform."""
    from .experiment_config import ExperimentConfig
    rows=effective_parameters(snapshot);raw=snapshot.to_dict()
    n=len(raw["levels"])
    for row in rows:row["weight"]=1/(n*n)
    raw.pop("pattern",None)
    raw["mode"]="independent_lanes"
    raw["base"].update(symbol_mode="random",sequence=None,channel_bandwidth_hz=None)
    raw["lanes"]=[dict(start=r["start"],end=r["end"],mean_start=r["mean_start"],mean_end=r["mean_end"],sigma=r["sigma_start"],kappa=r["kappa"],center=r["center"],weight=r["weight"]) for r in rows]
    raw["lane_overrides"]=rows
    raw["preset_name"]="Editable independent-lane copy"
    return ExperimentConfig.parse(raw)
