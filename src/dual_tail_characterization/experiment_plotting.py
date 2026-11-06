"""Figure rendering over stored runs; no generation and no widget dependencies."""
from dataclasses import dataclass
import threading
import weakref
import copy
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from .eyes import extract_eye
from .experiment_backend import phase_samples

PLOT_LOCK=threading.RLock()
VIEWS=("Waveform","1UI pattern","2UI eye","Level histogram","Lane histogram","Eye Height")


@dataclass(frozen=True)
class PlotSettings:
    view: str="2UI eye"
    style: str="combined"
    color: str="grayscale"
    normalization: str="log"
    density_edges: str="smooth"  # legacy display setting retained for JSON compatibility
    eye_rendering: str="Raw Counts"
    supersampling: int=2
    max_traces: int=300
    time_bins_per_ui: int=64
    amplitude_bins: int=96
    reference_lines: bool=True
    eh_overlay: bool=False
    selection: str="full"
    level: int=0
    lane_start: int=0
    lane_end: int=3
    sample_phase: float=.5
    display_phase: float=0.

    def __post_init__(self):
        if self.view not in VIEWS or self.style not in ("density","traces","combined") or self.color not in ("grayscale","color") or self.normalization not in ("linear","log"):
            raise ValueError("invalid plot view, rendering style, colormap or normalization")
        if self.eye_rendering not in ("Raw Counts","Smooth Eye"):
            raise ValueError("Eye Rendering must be Raw Counts or Smooth Eye")
        if isinstance(self.supersampling,bool) or not isinstance(self.supersampling,int) or self.supersampling not in (2,4):
            raise ValueError("Supersampling must be 2 or 4")
        if self.density_edges not in ("smooth","bins"):
            raise ValueError("density_edges must be smooth or bins")
        if self.selection not in ("full","lane","level"):
            raise ValueError("invalid conditioning selection")
        for key in ("max_traces","time_bins_per_ui","amplitude_bins"):
            if isinstance(getattr(self,key),bool) or not isinstance(getattr(self,key),int) or getattr(self,key)<1:
                raise ValueError(f"{key} must be a positive integer")
        if self.amplitude_bins>2048:
            raise ValueError("amplitude_bins must be <=2048")
        for key in ("level","lane_start","lane_end"):
            if not isinstance(getattr(self,key),int) or getattr(self,key)<0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not np.isfinite(self.sample_phase) or not 0<=self.sample_phase<=1 or not np.isfinite(self.display_phase):
            raise ValueError("Histogram phase must be in [0,1]; display phase must be finite")


def _selection(labels,settings):
    if settings.selection=="lane":
        return (labels[:,0]==settings.lane_start)&(labels[:,1]==settings.lane_end)
    if settings.selection=="level":
        return labels[:,0]==settings.level
    return np.ones(len(labels),dtype=bool)


def eye_data(record,length,settings):
    data=record.data
    if data.waveform is None:
        if length!=1:
            raise ValueError("Independent finite lanes have no 2UI waveform; no traces are joined or duplicated")
        if settings.display_phase!=0:
            raise ValueError("Independent finite lanes support zero display offset only; no wrapping or shifted support is fabricated")
        return np.linspace(0,1,data.arrays["segments"].shape[1]),data.arrays["segments"],data.arrays["segment_labels"]
    w=data.waveform
    eye=extract_eye(w,length,phase_ui=settings.display_phase)
    indices=eye.clock_start_symbols-1
    labels=np.full((len(indices),2),-1,dtype=int)
    valid=(indices>=0)&(indices<len(w.symbol_indices)-1)
    labels[valid]=np.stack((w.symbol_indices[indices[valid]],w.symbol_indices[indices[valid]+1]),axis=1)
    return eye.relative_time_ui,eye.traces,labels


def raw_eye_histogram(traces,relative,limits,settings):
    time_bins=int(relative[-1])*settings.time_bins_per_ui
    if time_bins>len(relative):
        raise ValueError("Time bins must not exceed the number of numerical sample phases")
    xedges=np.linspace(0,relative[-1],time_bins+1)
    yedges=np.linspace(*limits,settings.amplitude_bins+1)
    counts=np.zeros((time_bins,settings.amplitude_bins),dtype=np.int64)
    positions=np.minimum(np.searchsorted(xedges,relative,side="right")-1,time_bins-1)
    for k,values in enumerate(traces.T):
        counts[positions[k]]+=np.histogram(values,yedges)[0]
    return counts,xedges,yedges


def render_run(figure,record,settings=None):
    """Return numerical histogram arrays, separate from all display normalization."""
    settings=settings or record.plot_settings
    if record.data is None:
        raise ValueError("Run has no completed numerical data")
    c=record.snapshot.config
    if max(settings.level,settings.lane_start,settings.lane_end)>=len(c.levels):
        raise ValueError("Selected level/lane lies outside the configured PAM alphabet")
    if settings.time_bins_per_ui>c.samples_per_ui:
        raise ValueError("Time bins per UI must be <= samples per UI")
    arrays={}
    with PLOT_LOCK:
        figure.clear()
        # Reset the layout starting point so repeated redraws have identical
        # final pixel geometry (and can reuse their display coverage cache).
        figure.subplots_adjust(left=.125,right=.9,bottom=.11,top=.88,wspace=.2,hspace=.2)
        figure._eye_render_metadata={}
        smooth_pending=None
        ax=figure.add_subplot(111)
        data=record.data
        w=data.waveform
        if settings.view=="Eye Height":
            from .eye_height_plotting import plot_height
            plot_height(ax,record)
        elif settings.view=="Waveform":
            if w is None:
                ax.text(.5,.5,"Unavailable: independent-lane benchmark has no continuous time-domain waveform",ha="center",va="center",wrap=True,transform=ax.transAxes)
                return arrays
            cutoff=min(w.time[-1],w.time[0]+16*c.symbol_period)
            mask=w.time<=cutoff
            ax.step(w.time[mask]/c.symbol_period,w.ideal[mask],where="post",label="Ideal nominal",color="gray",alpha=.6)
            ax.plot(w.time[mask]/c.symbol_period,w.final[mask],lw=1,label="Final waveform")
            if w.symbol_endpoint_values is not None:
                keep=w.symbol_centers<=cutoff
                ax.scatter(w.symbol_centers[keep]/c.symbol_period,w.symbol_endpoint_values[keep],s=14,label="Saved endpoints")
            if settings.reference_lines:
                for k in range(int(w.time[0]/c.symbol_period),int(cutoff/c.symbol_period)+1):
                    ax.axvline(k,color="gray",lw=.4,alpha=.4)
                for level in c.levels:
                    ax.axhline(level,color="gray",lw=.4,alpha=.4)
            ax.set(xlabel="Time (UI)",ylabel=f"Amplitude ({c.amplitude_unit})",title="Continuous source waveform")
            ax.legend(fontsize=8)
        elif settings.view in ("1UI pattern","2UI eye"):
            length=1 if settings.view=="1UI pattern" else 2
            if length==2 and w is None:
                ax.text(.5,.5,"Unavailable: independent finite lanes are not joined into a 2UI eye",ha="center",va="center",transform=ax.transAxes)
                return arrays
            relative,traces,labels=eye_data(record,length,settings)
            eligible_rows=np.flatnonzero(_selection(labels,settings))
            traces=traces[eligible_rows]
            display_rows=eligible_rows[:settings.max_traces]
            source=w.final if w is not None else data.arrays["segments"]
            low,high=min(float(source.min()),min(c.levels)),max(float(source.max()),max(c.levels))
            pad=.08*(high-low) or .1
            limits=(low-pad,high+pad)
            cache=getattr(figure,"_eye_hist_cache",{})
            source_id=id(w.final) if w is not None else id(data.arrays["segments"])
            key=(id(data),source_id,length,settings.display_phase,settings.selection,settings.level,settings.lane_start,settings.lane_end,
                 settings.amplitude_bins,settings.time_bins_per_ui,limits)
            entry=cache.get(key)
            if entry is not None and entry[0]() is data:
                counts,x,y=entry[1:]
            else:
                counts,x,y=raw_eye_histogram(traces,relative,limits,settings)
                for array in (counts,x,y):array.flags.writeable=False
                if len(cache)>=2:cache.pop(next(iter(cache)))
                cache[key]=(weakref.ref(data),counts,x,y)
            figure._eye_hist_cache=cache
            arrays=dict(counts=counts,time_edges_ui=x,amplitude_edges=y)
            ax.set_facecolor("black")
            cmap="gray" if settings.color=="grayscale" else "viridis"
            if settings.eye_rendering=="Raw Counts":
                scale=LogNorm(vmin=1,vmax=max(2,int(counts.max()))) if settings.normalization=="log" else Normalize(vmin=0,vmax=max(1,int(counts.max())))
                mesh=ax.imshow(np.ma.masked_equal(counts.T,0),origin="lower",extent=(x[0],x[-1],y[0],y[-1]),
                    aspect="auto",interpolation="nearest",resample=False,cmap=cmap,norm=scale)
                figure.colorbar(mesh,ax=ax,label="Sample count"+(" (log display)" if settings.normalization=="log" else ""))
                figure._eye_render_metadata=dict(eye_rendering="Raw Counts",eligible_traces=len(traces),
                    displayed_traces=len(traces),display_phase_ui=settings.display_phase,window_ui=length,
                    interpolation="nearest",count_source="all eligible numerical samples",measurement_source=False)
            else:
                scale=LogNorm(vmin=1e-4,vmax=1) if settings.normalization=="log" else Normalize(vmin=0,vmax=1)
                mesh=ax.imshow(np.ma.masked_equal(np.zeros((2,2)),0),origin="lower",extent=(0,length,*limits),
                    aspect="auto",interpolation="nearest",resample=False,cmap=cmap,norm=scale)
                figure.colorbar(mesh,ax=ax,label="Relative display intensity")
                smooth_pending=(mesh,length,display_rows,limits)
            if settings.reference_lines:
                refs=[.5] if length==1 else [.5,1.,1.5]
                for value in refs:
                    ax.axvline(value-settings.display_phase,color="gray",lw=.6,linestyle="--")
                if record.snapshot.to_dict().get("lane_overrides"):
                    from .lane_parameters import effective_parameters
                    lookup={(r["start"],r["end"]):r for r in effective_parameters(record.snapshot)}
                    if w is None:
                        classes=[(labels,0)]
                    else:
                        eye=extract_eye(w,length,phase_ui=settings.display_phase)
                        indices=eye.clock_start_symbols-1
                        classes=[]
                        chosen=_selection(labels,settings)
                        for segment in range(length):
                            idx=indices[chosen]+segment
                            idx=idx[(idx>=0)&(idx<len(w.symbol_indices)-1)]
                            classes.append((np.stack((w.symbol_indices[idx],w.symbol_indices[idx+1]),axis=1),segment))
                    for pairs,segment in classes:
                        for a,b in np.unique(pairs,axis=0):
                            if (int(a),int(b)) in lookup:
                                r=lookup[int(a),int(b)]
                                ax.axvline(segment+r["center"]-settings.display_phase,color="tab:orange",lw=.7,alpha=.55,linestyle=":")
                for value in c.levels:
                    ax.axhline(value,color="gray",lw=.4)
            ax.set(xlabel="Relative time (UI)",ylabel=f"Amplitude ({c.amplitude_unit})",xlim=(0,length),ylim=limits,
                title=f"{'1UI transition pattern' if length==1 else '2UI eye diagram'} · {len(traces):,} eligible traces\n"
                      f"Display phase {settings.display_phase:g} UI; " +
                      (f"Smooth Eye {settings.supersampling}× · displayed {len(display_rows):,} / eligible {len(traces):,}" if settings.eye_rendering=="Smooth Eye" else "Raw Counts · all eligible numerical samples"))
            if not len(traces):
                ax.text(.5,.5,"No observations for this conditioning population",ha="center",transform=ax.transAxes,color="white")
        else:
            labels=data.arrays["segment_labels"]
            values=phase_samples(data.arrays["segments"],settings.sample_phase)
            chosen=values[_selection(labels,settings)]
            counts,edges=np.histogram(chosen,bins=settings.amplitude_bins)
            arrays=dict(counts=counts,amplitude_edges=edges)
            ax.stairs(counts,edges,color="black" if settings.color=="grayscale" else "tab:blue")
            if settings.reference_lines:
                for level in c.levels:
                    ax.axvline(level,color="gray",lw=.5)
            population=("conditioned on starting-symbol level; outgoing lanes may be pooled" if settings.view=="Level histogram" else "actual ordered lane labels")
            ax.set(xlabel=f"Amplitude ({c.amplitude_unit})",ylabel="Sample count",
                   title=f"{settings.view} · segment phase u={settings.sample_phase:g} · n={len(chosen):,}\n{population}; pooled spread is not a component sigma")
        if settings.eh_overlay and settings.view in ("1UI pattern","2UI eye"):
            from .eye_height_plotting import overlay_height
            overlay_height(ax,record,settings)
        figure.tight_layout()
        if smooth_pending is not None:
            from .smooth_eye import smooth_coverage
            mesh,length,rows,limits=smooth_pending
            coverage_cache=getattr(figure,"_eye_coverage_cache",{})
            from .smooth_eye import MAX_WIDTH,MAX_HEIGHT
            axis_width=max(1,int(round(ax.bbox.width)));axis_height=max(1,int(round(ax.bbox.height)))
            pixel_scale=min(1.,MAX_WIDTH/axis_width,MAX_HEIGHT/axis_height)
            size=(max(1,int(round(axis_width*pixel_scale))),max(1,int(round(axis_height*pixel_scale))))
            coverage_key=(id(data),source_id,record.snapshot.snapshot_json,length,settings.display_phase,tuple(map(int,rows)),limits,size,pixel_scale,settings.supersampling)
            cached=coverage_cache.get(coverage_key)
            if cached is not None and cached[0]() is data:
                intensity=cached[1];metadata=copy.deepcopy(cached[2])
                metadata.update(coverage_cached=True,initial_coverage_seconds=metadata["coverage_seconds"],coverage_seconds=0.)
            else:
                intensity,metadata=smooth_coverage(record,length,settings,rows,limits,*size,pixel_scale=pixel_scale)
                intensity.flags.writeable=False
                if len(coverage_cache)>=2:coverage_cache.pop(next(iter(coverage_cache)))
                coverage_cache[coverage_key]=(weakref.ref(data),intensity,copy.deepcopy(metadata))
                metadata["coverage_cached"]=False
            figure._eye_coverage_cache=coverage_cache
            mesh.set_data(np.ma.masked_equal(intensity,0))
            metadata.update(eligible_traces=len(traces),intensity_scale="fraction of displayed traces; linear coverage before downsample; fixed 0..1 mapping")
            figure._eye_render_metadata=metadata
    return arrays
