"""Tkinter front end over immutable run snapshots and numerical services."""
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from .experiment_config import ExperimentConfig, MODE_NAMES, preset, number, capabilities, lane_preview
from .experiment_backend import prescribed_context_table
from .experiment_plotting import PlotSettings, VIEWS, render_run
from .run_controller import RunController
from .gui_widgets import ScrollFrame, DataTable, RecordDialog

GENERAL_FIELDS=(
    ("level_count","PAM level count","4"),("symbol_period_ps","Symbol period (ps)","100"),
    ("samples_per_ui","Samples per UI","128"),("n_symbols","Symbols / benchmark realizations","2000"),
    ("seed","Seed","123"),("sequence","Explicit symbol indices (JSON or comma-separated)",""),
    ("transition_10_90_ui","Logistic 10–90 duration (UI)","0.2"),("shared_kappa","Shared connector κ (blank = solve duration)",""),
    ("channel_bandwidth_ghz","Channel bandwidth (GHz; blank = disabled)",""),
    ("noise_std","Additional sample amplitude-noise σ","0"),("amplitude_unit","Amplitude unit","arbitrary units"),
    ("output_root","Output root directory","runs/gui"),
)


class PamApplication:
    def __init__(self,root):
        self.root=root
        root.title("PAM waveform and level validation")
        root.geometry("1420x900")
        root.minsize(820,600)
        self.controller=RunController()
        self.loading=False
        self.closed=False
        self.last_busy=None
        self.pending_replot=False
        self.debounce=None
        self.after_id=None
        self.custom_presets={}
        self.fields={key:tk.StringVar(value=default) for key,_,default in GENERAL_FIELDS}
        self.symbol_mode=tk.StringVar(value="random")
        self.mode_var=tk.StringVar(value=MODE_NAMES["shared_endpoints"])
        self.preset_var=tk.StringVar(value="Option 1 PAM4")
        self.noise_model=tk.StringVar(value="trace_offset")
        self.probability_text=tk.StringVar()
        self.input_status=tk.StringVar()
        self.summary=tk.StringVar(value="Generate a run to inspect data. Configuration edits never modify completed runs.")
        self.busy_text=tk.StringVar(value="Ready")
        self.plot_vars={key:(tk.BooleanVar(value=value) if isinstance(value,bool) else tk.StringVar(value=str(value))) for key,value in asdict(PlotSettings()).items()}
        self._build_layout()
        self._menus()
        for variable in self.fields.values():
            variable.trace_add("write",self._inputs_changed)
        self.symbol_mode.trace_add("write",self._inputs_changed)
        self.noise_model.trace_add("write",self._inputs_changed)
        self.load_snapshot(preset())
        self.log("Ready. References app.py/screenshot were not attached; using the requested split-pane layout.")
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.after_id=root.after(100,self._poll)

    def _build_layout(self):
        split=ttk.Panedwindow(self.root,orient="horizontal")
        split.pack(fill="both",expand=True,padx=5,pady=5)
        left=ttk.Panedwindow(split,orient="vertical")
        right=ttk.Frame(split)
        split.add(left,weight=3)
        split.add(right,weight=2)
        plot_frame=ttk.Frame(left)
        log_frame=ttk.Frame(left)
        left.add(plot_frame,weight=5)
        left.add(log_frame,weight=1)
        viewbar=ttk.Frame(plot_frame)
        viewbar.pack(fill="x")
        ttk.Label(viewbar,text="Plot view").pack(side="left",padx=4)
        view=ttk.Combobox(viewbar,textvariable=self.plot_vars["view"],values=VIEWS,state="readonly",width=20)
        view.pack(side="left",padx=4)
        view.bind("<<ComboboxSelected>>",lambda event:self.replot())
        self.replot_button=ttk.Button(viewbar,text="Replot stored data",command=self.replot)
        self.replot_button.pack(side="right",padx=4)
        self.figure=Figure(figsize=(7,5),dpi=100)
        self.figure.add_subplot(111).text(.5,.5,"Generate or select a completed run",ha="center",transform=self.figure.axes[0].transAxes)
        self.canvas=FigureCanvasTkAgg(self.figure,master=plot_frame)
        toolbar=NavigationToolbar2Tk(self.canvas,plot_frame,pack_toolbar=False)
        toolbar.pack(side="bottom",fill="x")
        self.canvas.get_tk_widget().pack(fill="both",expand=True)
        ttk.Label(log_frame,textvariable=self.summary,wraplength=650,justify="left").pack(fill="x",padx=5,pady=4)
        self.log_widget=ScrolledText(log_frame,height=6,wrap="word",state="disabled")
        self.log_widget.pack(fill="both",expand=True)
        log_frame.bind("<Configure>",lambda event:self._summary_width(event.width))
        controls=ttk.Frame(right)
        controls.pack(fill="x",padx=4,pady=3)
        ttk.Label(controls,text="Generation mode").grid(row=0,column=0,sticky="w")
        mode=ttk.Combobox(controls,textvariable=self.mode_var,values=list(MODE_NAMES.values()),state="readonly")
        mode.grid(row=0,column=1,columnspan=3,sticky="ew",padx=4)
        mode.bind("<<ComboboxSelected>>",lambda event:self.load_snapshot(preset(self.mode_key())))
        self.generate_button=ttk.Button(controls,text="Generate",command=self.generate)
        self.export_button=ttk.Button(controls,text="Export selected",command=self.export_selected)
        self.remove_button=ttk.Button(controls,text="Remove run",command=self.remove_selected)
        self.generate_button.grid(row=1,column=0,columnspan=2,sticky="ew",padx=2,pady=3)
        self.export_button.grid(row=1,column=2,columnspan=2,sticky="ew",padx=2,pady=3)
        self.remove_button.grid(row=2,column=0,columnspan=2,sticky="ew",padx=2,pady=3)
        ttk.Button(controls,text="Load run inputs",command=self.load_run_inputs).grid(row=2,column=2,columnspan=2,sticky="ew",padx=2,pady=3)
        ttk.Label(controls,textvariable=self.busy_text,wraplength=230).grid(row=3,column=0,columnspan=2,sticky="w")
        self.progress=ttk.Progressbar(controls,mode="indeterminate")
        self.progress.grid(row=3,column=2,columnspan=2,sticky="ew")
        for col in range(4):
            controls.columnconfigure(col,weight=1)
        history_frame=ttk.LabelFrame(right,text="Run history (immutable inputs; created time UTC)")
        history_frame.pack(fill="x",padx=4,pady=4)
        self.history=ttk.Treeview(history_frame,columns=("mode","status","created"),show="tree headings",height=4,selectmode="browse")
        self.history.heading("#0",text="Run")
        self.history.column("#0",width=70,stretch=False)
        for key,width in (("mode",145),("status",70),("created",110)):
            self.history.heading(key,text=key.title())
            self.history.column(key,width=width,minwidth=60)
        bar=ttk.Scrollbar(history_frame,orient="horizontal",command=self.history.xview)
        vertical=ttk.Scrollbar(history_frame,orient="vertical",command=self.history.yview)
        self.history.configure(xscrollcommand=bar.set,yscrollcommand=vertical.set)
        self.history.grid(row=0,column=0,sticky="nsew")
        vertical.grid(row=0,column=1,sticky="ns")
        bar.grid(row=1,column=0,sticky="ew")
        history_frame.columnconfigure(0,weight=1)
        self.history.bind("<<TreeviewSelect>>",self._select_history)
        self.notebook=ttk.Notebook(right)
        self.notebook.pack(fill="both",expand=True,padx=4)
        config_host=ttk.Frame(self.notebook)
        validation_host=ttk.Frame(self.notebook)
        self.notebook.add(config_host,text="Configuration")
        self.notebook.add(validation_host,text="Validation tables")
        self.config_section=tk.StringVar(value="General")
        config_selector=ttk.Combobox(config_host,textvariable=self.config_section,state="readonly",
            values=("General","Level Properties","Lane Properties","Incoming proposals","Outgoing proposals","Context overrides","Prescribed contexts","Plot controls"))
        config_selector.pack(fill="x",padx=4,pady=4)
        config_selector.bind("<<ComboboxSelected>>",lambda event:self._show_config())
        self.config_body=ttk.Frame(config_host)
        self.config_body.pack(fill="both",expand=True)
        self.config_body.rowconfigure(0,weight=1)
        self.config_body.columnconfigure(0,weight=1)
        self.pages={}
        self._build_general()
        self.edit_tables={}
        definitions=(("Level Properties","levels",True),("Lane Properties","lanes",False),("Incoming proposals","incoming",False),
                     ("Outgoing proposals","outgoing",False),("Context overrides","contexts",True))
        for title,key,add_delete in definitions:
            table=DataTable(self.config_body,edit=lambda index,k=key:self.edit_row(k,index),
                add=(lambda k=key:self.add_row(k)) if add_delete else None,
                delete=(lambda k=key:self.delete_row(k)) if add_delete else None)
            table.grid(row=0,column=0,sticky="nsew")
            self.pages[title]=table
            self.edit_tables[key]=table
        context_table=DataTable(self.config_body)
        context_table.grid(row=0,column=0,sticky="nsew")
        self.pages["Prescribed contexts"]=context_table
        self._build_plot_controls()
        self.input_status_label=ttk.Label(config_host,textvariable=self.input_status,wraplength=480,justify="left")
        self.input_status_label.pack(fill="x",padx=5,pady=4)
        config_host.bind("<Configure>",lambda event:self.input_status_label.configure(wraplength=max(180,event.width-15)))
        io=ttk.Frame(config_host)
        io.pack(fill="x",padx=4,pady=4)
        for index,(text,command) in enumerate((("Import table CSV",self.import_csv),("Export table CSV",self.export_csv),("Save JSON",self.save_json),("Load JSON",self.load_json))):
            ttk.Button(io,text=text,command=command).grid(row=index//2,column=index%2,sticky="ew",padx=2,pady=2)
        io.columnconfigure(0,weight=1)
        io.columnconfigure(1,weight=1)
        self.validation_section=tk.StringVar(value="Input / effective")
        selector=ttk.Combobox(validation_host,textvariable=self.validation_section,state="readonly",
            values=("Input / effective","Lane theory / empirical","Level statistics","Component parameters","Internal variance","Observed contexts","Validation status"))
        selector.pack(fill="x",padx=4,pady=4)
        selector.bind("<<ComboboxSelected>>",lambda event:self._show_validation())
        self.validation_table=DataTable(validation_host)
        self.validation_table.pack(fill="both",expand=True)
        self._show_config()

    def _summary_width(self,width):
        for widget in self.log_widget.master.winfo_children():
            if isinstance(widget,ttk.Label):
                widget.configure(wraplength=max(200,width-15))

    def _build_general(self):
        page=ScrollFrame(self.config_body)
        page.grid(row=0,column=0,sticky="nsew")
        self.pages["General"]=page
        body=page.body
        ttk.Label(body,text="Preset").grid(row=0,column=0,sticky="w",padx=5)
        self.preset_box=ttk.Combobox(body,textvariable=self.preset_var,state="readonly",
            values=("Original customized PAM4","Option 1 PAM4","Option 2 PAM4","Continuous PAM4"))
        self.preset_box.grid(row=0,column=1,sticky="ew",padx=5)
        self.preset_box.bind("<<ComboboxSelected>>",self.choose_preset)
        tools=ttk.Frame(body)
        tools.grid(row=1,column=0,columnspan=2,sticky="ew",padx=5,pady=4)
        ttk.Button(tools,text="Duplicate preset",command=self.duplicate_preset).grid(row=0,column=0,sticky="ew",padx=2,pady=2)
        ttk.Button(tools,text="Reset to preset",command=lambda:self.load_snapshot(preset(self.mode_key()))).grid(row=0,column=1,sticky="ew",padx=2,pady=2)
        ttk.Button(tools,text="Rebuild N levels",command=self.rebuild_levels).grid(row=1,column=0,columnspan=2,sticky="ew",padx=2,pady=2)
        tools.columnconfigure(0,weight=1)
        tools.columnconfigure(1,weight=1)
        self.entries={}
        for row,(key,label,_) in enumerate(GENERAL_FIELDS,2):
            ttk.Label(body,text=label,wraplength=235).grid(row=row,column=0,sticky="w",padx=5,pady=5)
            entry=ttk.Entry(body,textvariable=self.fields[key])
            entry.grid(row=row,column=1,sticky="ew",padx=5,pady=5)
            self.entries[key]=entry
        row=len(GENERAL_FIELDS)+2
        ttk.Label(body,text="Symbol selection").grid(row=row,column=0,sticky="w",padx=5,pady=5)
        self.symbol_mode_box=ttk.Combobox(body,textvariable=self.symbol_mode,values=("random","explicit"),state="readonly")
        self.symbol_mode_box.grid(row=row,column=1,sticky="ew",padx=5)
        ttk.Label(body,text="Symbol probabilities (edit Level Properties)").grid(row=row+1,column=0,sticky="w",padx=5)
        ttk.Entry(body,textvariable=self.probability_text,state="readonly").grid(row=row+1,column=1,sticky="ew",padx=5)
        ttk.Label(body,text="Independent lane noise correlation").grid(row=row+2,column=0,sticky="w",padx=5,pady=5)
        self.noise_model_box=ttk.Combobox(body,textvariable=self.noise_model,values=("trace_offset","sample_independent"),state="readonly")
        self.noise_model_box.grid(row=row+2,column=1,sticky="ew",padx=5)
        ttk.Button(body,text="Browse output root…",command=self.browse_output).grid(row=row+3,column=1,sticky="ew",padx=5,pady=5)
        self.support_note=tk.StringVar()
        ttk.Label(body,textvariable=self.support_note,wraplength=430,justify="left").grid(row=row+4,column=0,columnspan=2,sticky="ew",padx=5,pady=8)
        ttk.Label(body,text="Rebuilding level count rebuilds complete lane/proposal tables and resets probabilities uniformly. Completed runs remain unchanged.",wraplength=430).grid(row=row+5,column=0,columnspan=2,sticky="ew",padx=5,pady=8)
        body.columnconfigure(1,weight=1)

    def _build_plot_controls(self):
        page=ScrollFrame(self.config_body)
        page.grid(row=0,column=0,sticky="nsew")
        self.pages["Plot controls"]=page
        self.plot_entries={}
        choices=dict(view=VIEWS,style=("density","traces","combined"),color=("grayscale","color"),normalization=("linear","log"),selection=("full","lane","level"))
        labels=dict(view="View",style="Rendering",color="Colormap",normalization="Count normalization",max_traces="Maximum displayed traces",
            time_bins_per_ui="Time bins per UI (<= samples/UI)",amplitude_bins="Amplitude bins",reference_lines="Reference lines",selection="Conditioning population",
            level="Selected starting level",lane_start="Selected lane start",lane_end="Selected lane end",sample_phase="Histogram phase u in segment [0,1]",display_phase="Display offset (UI; positive shifts left)")
        for row,(key,var) in enumerate(self.plot_vars.items()):
            ttk.Label(page.body,text=labels[key],wraplength=230).grid(row=row,column=0,sticky="w",padx=5,pady=5)
            if key=="reference_lines":
                widget=ttk.Checkbutton(page.body,variable=var)
            elif key in choices:
                widget=ttk.Combobox(page.body,textvariable=var,values=choices[key],state="readonly")
            else:
                widget=ttk.Entry(page.body,textvariable=var)
            widget.grid(row=row,column=1,sticky="ew",padx=5,pady=5)
            self.plot_entries[key]=widget
        row=len(self.plot_vars)
        ttk.Button(page.body,text="Replot selected run (no generation)",command=self.replot).grid(row=row,column=0,columnspan=2,sticky="ew",padx=5,pady=8)
        ttk.Label(page.body,text="Histograms use one observation per segment at phase u. Level conditioning uses the starting symbol; pooled level spread is not a component sigma. Validation always uses all eligible observations.",wraplength=430).grid(row=row+1,column=0,columnspan=2,sticky="ew",padx=5,pady=8)
        page.body.columnconfigure(1,weight=1)

    def _menus(self):
        menu=tk.Menu(self.root)
        file_menu=tk.Menu(menu,tearoff=False)
        for label,command in (("Load configuration JSON…",self.load_json),("Save configuration JSON…",self.save_json),("Import active table CSV…",self.import_csv),("Export active table CSV…",self.export_csv),("Export selected run",self.export_selected)):
            file_menu.add_command(label=label,command=command)
        file_menu.add_separator()
        file_menu.add_command(label="Exit",command=self.close)
        menu.add_cascade(label="File",menu=file_menu)
        run_menu=tk.Menu(menu,tearoff=False)
        for label,command in (("Generate",self.generate),("Export selected run",self.export_selected),("Load selected run inputs",self.load_run_inputs),("Remove selected run",self.remove_selected)):
            run_menu.add_command(label=label,command=command)
        menu.add_cascade(label="Run",menu=run_menu)
        plot_menu=tk.Menu(menu,tearoff=False)
        for view in VIEWS:
            plot_menu.add_command(label=view,command=lambda name=view:self.change_view(name))
        plot_menu.add_command(label="Replot",command=self.replot)
        menu.add_cascade(label="Plot",menu=plot_menu)
        help_menu=tk.Menu(menu,tearoff=False)
        help_menu.add_command(label="Model and controls",command=lambda:messagebox.showinfo("PAM validation", "Option 1 reuses saved per-symbol endpoints. Option 2 uses context proposals/overrides, not arbitrary independent lane endpoints. All shared connectors have a common symmetric center. Independent lanes have no time waveform or 2UI view. See docs/gui.md for model rules, exports and limitations.",parent=self.root))
        menu.add_cascade(label="Help",menu=help_menu)
        self.root.configure(menu=menu)

    def mode_key(self):
        return next(key for key,value in MODE_NAMES.items() if value==self.mode_var.get())

    def load_snapshot(self,snapshot):
        self.loading=True
        self.raw=snapshot.to_dict()
        raw,c=self.raw,snapshot.config
        self.mode_var.set(MODE_NAMES[snapshot.mode])
        names=dict(independent_lanes="Original customized PAM4",shared_endpoints="Option 1 PAM4",history_endpoints="Option 2 PAM4",continuous_filter="Continuous PAM4")
        self.preset_var.set(names.get(snapshot.mode,raw["preset_name"]))
        values=dict(level_count=len(c.levels),symbol_period_ps=c.symbol_period*1e12,samples_per_ui=c.samples_per_ui,n_symbols=c.n_symbols,seed=c.seed,
            sequence="" if c.sequence is None else json.dumps(c.sequence),transition_10_90_ui=c.transition_10_90_ui,shared_kappa="" if c.shared_kappa is None else c.shared_kappa,
            channel_bandwidth_ghz="" if c.channel_bandwidth_hz is None else c.channel_bandwidth_hz/1e9,noise_std=c.noise_std,amplitude_unit=c.amplitude_unit,
            output_root=raw["output_root"])
        for key,value in values.items():
            self.fields[key].set(str(value))
        self.symbol_mode.set(c.symbol_mode)
        self.noise_model.set(raw["noise_model"])
        for key,table in self.edit_tables.items():
            table.set_rows(copy.deepcopy(raw[key]),columns=("previous","current","next","mean","sigma") if key=="contexts" else None)
        self._set_capabilities()
        self.loading=False
        self._validate_inputs()
        if snapshot.mode=="independent_lanes":
            self.plot_vars["view"].set("1UI pattern")
            self.plot_vars["display_phase"].set("0")
        self.log(f"Loaded editable inputs: {raw['preset_name']}. Completed run records were not changed.")

    def _set_capabilities(self):
        mode=self.mode_key()
        cap=capabilities(mode)
        for key,enabled in (("channel_bandwidth_ghz",cap["channel_edit"]),("noise_std",cap["additive_edit"]),
            ("sequence",cap["explicit_edit"]),("shared_kappa",mode in ("shared_endpoints","history_endpoints")),("transition_10_90_ui",mode!="independent_lanes")):
            self.entries[key].configure(state="normal" if enabled else "disabled")
        self.symbol_mode_box.configure(state="readonly" if cap["explicit_edit"] else "disabled")
        self.noise_model_box.configure(state="readonly" if mode=="independent_lanes" else "disabled")
        self.plot_entries["display_phase"].configure(state="disabled" if mode=="independent_lanes" else "normal")
        for key,enabled in (("lanes",cap["lane_edit"]),("incoming",cap["proposals_edit"]),("outgoing",cap["proposals_edit"]),("contexts",cap["contexts_edit"])):
            for button in self.edit_tables[key].action_buttons:
                button.state(["!disabled"] if enabled else ["disabled"])
        notes=dict(independent_lanes="Independent finite lanes. σ is lane noise; trace_offset shares one Gaussian offset along a realization, sample_independent draws each sample. Lane κ/center and occurrence weights are editable. No channel, sequence or 2UI waveform.",
            shared_endpoints="Option 1: lane endpoints are derived read-only from Level Properties. Common symmetric connector only. Channel and additive sample noise are disabled; endpoint amplitude σ is separate from timing jitter.",
            history_endpoints="Option 2: edit incoming/outgoing proposals and optional context overrides. Shared M=(incoming+outgoing)/2 endpoints are controlled history dependence, not an exact reconstruction of arbitrary independent lanes. Common symmetric connector; no channel or additive sample noise.",
            continuous_filter="Continuous logistic smoothing and optional causal low-pass channel. Endpoint-sigma/lane-proposal edits are unavailable. Channel ISI, additive sample noise and display phase are separate settings.")
        self.support_note.set(notes[mode])

    def gather_snapshot(self):
        raw=copy.deepcopy(self.raw)
        raw["mode"]=self.mode_key()
        raw["output_root"]=self.fields["output_root"].get().strip()
        n=number(self.fields["level_count"].get(),"PAM level count",True)
        raw["levels"]=copy.deepcopy(self.edit_tables["levels"].rows)
        if n!=len(raw["levels"]):
            raise ValueError("PAM level count differs from Level Properties rows; use Rebuild N levels")
        for key in ("incoming","outgoing","contexts"):
            raw[key]=copy.deepcopy(self.edit_tables[key].rows)
        if raw["mode"]=="independent_lanes":
            raw["lanes"]=copy.deepcopy(self.edit_tables["lanes"].rows)
        base=raw["base"]
        for key in ("samples_per_ui","n_symbols","seed"):
            base[key]=number(self.fields[key].get(),key,True)
        base["symbol_period"]=number(self.fields["symbol_period_ps"].get(),"Symbol period (ps)")*1e-12
        base["symbol_mode"]=self.symbol_mode.get()
        base["amplitude_unit"]=self.fields["amplitude_unit"].get()
        base["noise_std"]=number(self.fields["noise_std"].get(),"Additional sample amplitude-noise sigma")
        base["transition_10_90_ui"]=number(self.fields["transition_10_90_ui"].get(),"Logistic transition duration")
        for field,key,scale in (("shared_kappa","shared_kappa",1),("channel_bandwidth_ghz","channel_bandwidth_hz",1e9)):
            text=self.fields[field].get().strip()
            base[key]=None if not text else number(text,field)*scale
        if base["symbol_mode"]=="explicit":
            text=self.fields["sequence"].get().strip()
            try:
                values=json.loads(text) if text.startswith("[") else text.replace(","," ").split()
                if not isinstance(values,list):
                    raise ValueError()
                base["sequence"]=[number(value,f"Explicit sequence index {i}",True) for i,value in enumerate(values)]
            except (ValueError,TypeError) as exc:
                raise ValueError(f"Explicit sequence: provide integer indices; {exc}") from exc
        else:
            base["sequence"]=None
        raw["noise_model"]=self.noise_model.get()
        return ExperimentConfig.parse(raw)

    def _inputs_changed(self,*args):
        if self.loading or self.closed:
            return
        if self.debounce:
            self.root.after_cancel(self.debounce)
        self.debounce=self.root.after(180,self._validate_inputs)

    def _validate_inputs(self):
        self.debounce=None
        try:
            snapshot=self.gather_snapshot()
            self.controller.mark_inputs(snapshot)
            self.input_status.set("Inputs valid. Generate creates a new immutable run; replot uses stored arrays.")
            if snapshot.mode!="independent_lanes":
                self.edit_tables["lanes"].set_rows(lane_preview(snapshot))
            if snapshot.mode=="history_endpoints":
                self.pages["Prescribed contexts"].set_rows(prescribed_context_table(snapshot))
            else:
                self.pages["Prescribed contexts"].set_rows([dict(status="unavailable",reason="Context prescriptions apply only to Option 2")])
            self.probability_text.set("Lane occurrence weights apply" if snapshot.mode=="independent_lanes" else ", ".join(str(r["probability"]) for r in snapshot.to_dict()["levels"]))
        except (ValueError,TypeError) as exc:
            self.controller.mark_inputs(None)
            self.input_status.set(f"Invalid input: {exc}")
        self._update_summary()

    def _show_config(self):
        page=self.pages[self.config_section.get()]
        page.tkraise()

    def _active_table_key(self):
        return {"Level Properties":"levels","Lane Properties":"lanes","Incoming proposals":"incoming","Outgoing proposals":"outgoing","Context overrides":"contexts"}.get(self.config_section.get())

    def _row_editable(self,key):
        cap=capabilities(self.mode_key())
        return key=="levels" or key=="lanes" and cap["lane_edit"] or key in ("incoming","outgoing") and cap["proposals_edit"] or key=="contexts" and cap["contexts_edit"]

    def edit_row(self,key,index):
        if not self._row_editable(key):
            self.log("Derived/unsupported cells are read-only in this mode.")
            return
        row=self.edit_tables[key].rows[index]
        editable=set(row)-({"level_id"} if key=="levels" else {"start","end"} if key in ("lanes","incoming","outgoing") else set())
        cap=capabilities(self.mode_key())
        if key=="levels":
            if not cap["endpoint_sigma_edit"]:
                editable.discard("sigma")
            if not cap["probability_edit"]:
                editable.discard("probability")
        def validate(values):
            result={}
            for field,value in values.items():
                result[field]=None if key=="contexts" and field=="sigma" and not value.strip() else number(value,f"{key} row {index+1} {field}",field in ("level_id","start","end","previous","current","next"))
            for field in ("sigma","probability","weight"):
                if result.get(field) is not None and result[field]<0:
                    raise ValueError(f"{field} must be nonnegative")
            return result
        def apply(result):
            self.edit_tables[key].rows[index]=result
            self.edit_tables[key].redraw()
            self._inputs_changed()
        RecordDialog(self.root,f"Edit {key} row {index+1}",row,editable,validate,apply)

    def add_row(self,key):
        if key=="levels":
            self._resize_levels(len(self.edit_tables[key].rows)+1)
        elif key=="contexts" and self._row_editable(key):
            self.edit_tables[key].rows.append(dict(previous=0,current=0,next=0,mean=0.,sigma=None))
            self.edit_tables[key].redraw()
            self.edit_row(key,len(self.edit_tables[key].rows)-1)
            self._inputs_changed()

    def delete_row(self,key):
        if key=="levels" and self.edit_tables[key].selected_index() is not None:
            self._resize_levels(len(self.edit_tables[key].rows)-1,remove_index=self.edit_tables[key].selected_index())
        elif key=="contexts" and self._row_editable(key):
            index=self.edit_tables[key].selected_index()
            if index is not None:
                self.edit_tables[key].rows.pop(index)
                self.edit_tables[key].redraw()
                self._inputs_changed()

    def rebuild_levels(self):
        try:
            self._resize_levels(number(self.fields["level_count"].get(),"Level count",True))
        except ValueError as exc:
            messagebox.showerror("Invalid level count",str(exc),parent=self.root)

    def _resize_levels(self,n,remove_index=None):
        if not 2<=n<=16:
            raise ValueError("GUI preset rebuilding supports 2–16 levels; the numerical backend supports arbitrary PAM-n")
        previous_count=self.fields["level_count"].get()
        self.loading=True
        self.fields["level_count"].set(str(len(self.edit_tables["levels"].rows)))
        try:
            current=self.gather_snapshot().to_dict()
        finally:
            self.fields["level_count"].set(previous_count)
            self.loading=False
        updated=preset(self.mode_key(),n).to_dict()
        updated["base"]=current["base"]
        updated["output_root"]=current["output_root"]
        old=copy.deepcopy(self.edit_tables["levels"].rows)
        if remove_index is not None:
            old.pop(remove_index)
        for index in range(min(n,len(old))):
            updated["levels"][index].update(mean=old[index]["mean"],sigma=old[index]["sigma"])
        for index in range(len(old),n):
            updated["levels"][index]["mean"]=updated["levels"][index-1]["mean"]+1
        self.load_snapshot(ExperimentConfig.parse(updated))
        self.log(f"Rebuilt {n} levels; complete lane/proposal tables and uniform probabilities reset explicitly.")

    def choose_preset(self,event=None):
        name=self.preset_var.get()
        if name in self.custom_presets:
            self.load_snapshot(self.custom_presets[name])
            self.preset_var.set(name)
            return
        mode={"Original customized PAM4":"independent_lanes","Option 1 PAM4":"shared_endpoints","Option 2 PAM4":"history_endpoints","Continuous PAM4":"continuous_filter"}[name]
        self.load_snapshot(preset(mode))

    def duplicate_preset(self):
        try:
            raw=self.gather_snapshot().to_dict()
            name=simpledialog.askstring("Duplicate preset","Name for editable preset copy:",parent=self.root)
            if not name:
                return
            if name in self.preset_box["values"]:
                raise ValueError("Choose a unique preset name")
            raw["preset_name"]=name
            self.custom_presets[name]=ExperimentConfig.parse(raw)
            self.preset_box.configure(values=(*self.preset_box["values"],name))
            self.preset_var.set(name)
            self.raw["preset_name"]=name
            self._inputs_changed()
        except ValueError as exc:
            messagebox.showerror("Cannot duplicate",str(exc),parent=self.root)

    def browse_output(self):
        path=filedialog.askdirectory(parent=self.root)
        if path:
            self.fields["output_root"].set(path)

    def save_json(self):
        try:
            snapshot=self.gather_snapshot()
            path=filedialog.asksaveasfilename(parent=self.root,defaultextension=".json",filetypes=[("JSON","*.json")])
            if path:
                Path(path).write_text(json.dumps(snapshot.to_dict(),indent=2),encoding="utf-8")
        except (ValueError,OSError) as exc:
            messagebox.showerror("Configuration save failed",str(exc),parent=self.root)

    def load_json(self):
        path=filedialog.askopenfilename(parent=self.root,filetypes=[("GUI configuration JSON","*.json")])
        if not path:
            return
        try:
            self.load_snapshot(ExperimentConfig.parse(json.loads(Path(path).read_text(encoding="utf-8"))))
        except (ValueError,OSError,TypeError) as exc:
            messagebox.showerror("Configuration load failed",str(exc),parent=self.root)

    def import_csv(self):
        key=self._active_table_key()
        if key is None or not self._row_editable(key):
            messagebox.showinfo("Read-only table","Select an editable configuration table. Derived lanes cannot be imported as contradictory endpoints.",parent=self.root)
            return
        path=filedialog.askopenfilename(parent=self.root,filetypes=[("CSV","*.csv")])
        if not path:
            return
        try:
            with Path(path).open(newline="",encoding="utf-8-sig") as file:
                rows=list(csv.DictReader(file))
            raw=self.gather_snapshot().to_dict()
            # Validate imported complete tables/unique keys before committing.
            if key=="lanes" and rows and set(rows[0])=={"start","end","mean_start","mean_end","sigma"}:
                rows=[dict(row,kappa=22.,center=.5,weight=1/len(rows)) for row in rows]
                self.log("Original five-column benchmark imported with explicit κ=22, center=0.5 and uniform lane weights.")
            raw[key]=rows
            snapshot=ExperimentConfig.parse(raw)
            self.load_snapshot(snapshot)
        except (ValueError,OSError,TypeError) as exc:
            messagebox.showerror("CSV import failed",str(exc),parent=self.root)

    def export_csv(self):
        key=self._active_table_key()
        if key is None:
            messagebox.showinfo("Select a table","Select a configuration table first.",parent=self.root)
            return
        table=self.edit_tables[key]
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension=".csv",filetypes=[("CSV","*.csv")])
        if path:
            try:
                with Path(path).open("w",newline="",encoding="utf-8") as file:
                    writer=csv.DictWriter(file,fieldnames=table.columns)
                    writer.writeheader()
                    writer.writerows(table.rows)
            except OSError as exc:
                messagebox.showerror("CSV export failed",str(exc),parent=self.root)

    def plot_settings(self):
        raw={key:var.get() for key,var in self.plot_vars.items()}
        for key in ("max_traces","time_bins_per_ui","amplitude_bins","level","lane_start","lane_end"):
            raw[key]=number(raw[key],key,True)
        for key in ("sample_phase","display_phase"):
            raw[key]=number(raw[key],key)
        return PlotSettings(**raw)

    def generate(self):
        try:
            snapshot=self.gather_snapshot()
            settings=self.plot_settings()
            if settings.time_bins_per_ui>snapshot.config.samples_per_ui:
                raise ValueError("Time bins per UI must not exceed samples per UI")
            record=self.controller.start(snapshot,settings,snapshot.to_dict()["output_root"])
            self.history.insert("","end",iid=record.run_id,text=record.run_id[:8],values=(snapshot.mode,"running",record.created_at[11:19]))
            self.history.selection_set(record.run_id)
            self._update_busy()
            self.log(f"Started {record.run_id[:8]} with an immutable configuration snapshot.")
        except (ValueError,RuntimeError) as exc:
            messagebox.showerror("Cannot generate",str(exc),parent=self.root)

    def export_selected(self):
        try:
            self.controller.export_selected()
            self._update_busy()
            self.log("Exporting the selected run's stored configuration, data and plot settings.")
        except (ValueError,RuntimeError) as exc:
            messagebox.showerror("Cannot export",str(exc),parent=self.root)

    def remove_selected(self):
        if self.controller.busy:
            return
        run_id=self.controller.selected_id
        if run_id:
            self.controller.remove(run_id)
            self.history.delete(run_id)
            if self.controller.selected_id:
                self.history.selection_set(self.controller.selected_id)
            else:
                self.figure.clear()
                self.figure.add_subplot(111).text(.5,.5,"No selected run",ha="center",transform=self.figure.axes[0].transAxes)
                self.canvas.draw_idle()
            self._show_validation()
            self._update_summary()

    def load_run_inputs(self):
        record=self.controller.records.get(self.controller.selected_id)
        if record:
            self.load_snapshot(record.snapshot)

    def _select_history(self,event=None):
        selection=self.history.selection()
        if not selection or selection[0] not in self.controller.records:
            return
        self.controller.selected_id=selection[0]
        record=self.controller.records[selection[0]]
        for key,value in asdict(record.plot_settings).items():
            self.plot_vars[key].set(value)
        self._show_validation()
        self._update_summary()
        if record.data is not None:
            self.replot()

    def _show_validation(self):
        record=self.controller.records.get(self.controller.selected_id)
        if record is None or record.data is None:
            self.validation_table.set_rows([dict(status="unavailable",reason="Select a completed run")])
            return
        wanted={"Input / effective":["input_effective.csv"],"Lane theory / empirical":["lane_validation.csv"],"Level statistics":["level_statistics.csv"],
            "Component parameters":["lane_components.csv","level_components.csv"],"Internal variance":["internal_variance.csv"],
            "Observed contexts":["observed_contexts.csv"],"Validation status":["validation_status.csv"]}[self.validation_section.get()]
        rows=[]
        for table in record.data.tables:
            if table.source_csv in wanted:
                rows.extend(dict(zip(table.columns,row)) for row in table.rows)
        self.validation_table.set_rows(rows or [dict(status="unavailable",reason="This run does not produce this analysis; no placeholder measurement")])

    def change_view(self,name):
        self.plot_vars["view"].set(name)
        self.replot()

    def replot(self):
        if self.controller.active_operation=="export":
            self.pending_replot=True
            self.log("Replot queued until export's isolated figure rendering finishes; no samples regenerated.")
            return
        record=self.controller.records.get(self.controller.selected_id)
        if record is None or record.data is None:
            return
        try:
            settings=self.plot_settings()
            render_run(self.figure,record,settings)
            self.controller.set_plot(record.run_id,settings)
            self.canvas.draw_idle()
        except ValueError as exc:
            self.log(f"Plot settings invalid: {exc}")
            self.busy_text.set(str(exc))

    def _update_summary(self):
        record=self.controller.records.get(self.controller.selected_id)
        if record:
            stale="INPUTS DIFFER — selected stored result remains valid" if self.controller.is_stale(record) else "Inputs match selected snapshot"
            self.summary.set(f"{record.snapshot.mode} · {record.status} · {stale}\n{record.data.summary if record.data else record.error or 'Worker in progress'}\nOutput: {record.output_directory}")

    def _update_busy(self):
        busy=self.controller.busy
        if busy==self.last_busy:
            return
        self.last_busy=busy
        for button in (self.generate_button,self.export_button,self.remove_button):
            button.state(["disabled"] if busy else ["!disabled"])
        if busy:
            self.busy_text.set("Worker busy; numerical/export actions are serialized")
            self.progress.start(15)
        else:
            self.busy_text.set("Ready")
            self.progress.stop()
            if self.pending_replot:
                self.pending_replot=False
                self.replot()

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                event=self.controller.events.get_nowait()
                accepted=self.controller.accept(event)
                if not accepted:
                    continue
                run_id,_,kind,payload=event
                if kind=="log":
                    self.log(str(payload))
                elif kind in ("generated","failed","exported","export_failed"):
                    record=self.controller.records[run_id]
                    self.history.item(run_id,values=(record.snapshot.mode,record.status,record.created_at[11:19]))
                    self.log(f"{run_id[:8]} {kind}: {payload if kind!='generated' else record.data.summary}")
                    if self.controller.selected_id==run_id:
                        self._show_validation()
                        self._update_summary()
                        if kind=="generated":
                            self.replot()
        except queue.Empty:
            pass
        self._update_busy()
        self.after_id=self.root.after(120,self._poll)

    def log(self,message):
        if self.closed:
            return
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end",message+"\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def close(self):
        if self.closed:
            return
        self.closed=True
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if self.debounce:
            self.root.after_cancel(self.debounce)
        self.controller.close()
        self.root.destroy()


def main():
    root=tk.Tk()
    PamApplication(root)
    root.mainloop()


if __name__=="__main__":
    main()
