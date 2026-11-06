"""Modal lane grid with cell, field-selector and scrollable row editing."""
import copy
import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .lane_parameters import LaneDraft, FIELDS, effective_parameters
from .lane_connector import connector
from .experiment_config import ExperimentConfig, clear_preset as preset
from .gui_widgets import RecordDialog

COLUMNS=FIELDS+("nominal_slope_ui","nominal_slope_per_second","status")

class LaneEditor(tk.Toplevel):
    def __init__(self,parent,snapshot,commit):
        super().__init__(parent)
        self.title("Edit Lane Parameters — draft");self.geometry("1100x700+15+15");self.minsize(740,480)
        self.draft=LaneDraft(snapshot);self.commit=commit
        self.anchor=(0,0);self.corner=None;self.cell_editor=None
        self.columnconfigure(0,weight=1);self.rowconfigure(1,weight=1)
        header=ttk.Frame(self);header.grid(row=0,column=0,sticky="ew")
        description=("Endpoint means, sigmas and weights are editable." if snapshot.mode=="independent_lanes" else
            "Endpoint means/sigmas are read-only here: use Level Properties or history rule editors.")
        self.mode_note=ttk.Label(header,text=f"Mode: {snapshot.mode}. {description}\nMore columns are to the RIGHT of kappa. Use the horizontal scrollbar, field selector, or Edit row fields.",wraplength=1000)
        self.mode_note.pack(anchor="w")
        shortcuts=ttk.Frame(header);shortcuts.pack(fill="x",pady=4)
        ttk.Button(shortcuts,text="Edit row fields…",command=self.edit_row_fields).pack(side="left",padx=3)
        ttk.Label(shortcuts,text="Editable field:").pack(side="left",padx=3)
        self.field_choice=tk.StringVar(value="kappa")
        self.field_selector=ttk.Combobox(shortcuts,textvariable=self.field_choice,state="readonly",
            values=[k for k in FIELDS if self.editable(k)],width=22)
        self.field_selector.pack(side="left",padx=3)
        if snapshot.mode!="independent_lanes":
            ttk.Button(shortcuts,text="Edit shared means/sigmas…",command=self.edit_shared_levels).pack(side="left",padx=3)
            ttk.Button(shortcuts,text="Create editable lane copy…",command=self.create_independent_copy).pack(side="left",padx=3)
        ttk.Button(shortcuts,text="Edit field",command=self.edit_chosen_field).pack(side="left",padx=3)
        frame=ttk.Frame(self);frame.grid(row=1,column=0,sticky="nsew");frame.columnconfigure(0,weight=1);frame.rowconfigure(0,weight=1)
        self.tree=ttk.Treeview(frame,columns=COLUMNS,show="headings",selectmode="extended",height=8)
        for key in COLUMNS:
            label=key.replace("phase_shift","midpoint shift [UI]").replace("nominal_slope_ui","signed slope [amplitude/UI]").replace("nominal_slope_per_second","signed slope [amplitude/s]")
            if not self.editable(key):label+=" [read-only]"
            self.tree.heading(key,text=label);self.tree.column(key,width=140,stretch=False)
        self.tree.grid(row=0,column=0,sticky="nsew")
        for orient in ("vertical","horizontal"):
            bar=ttk.Scrollbar(frame,orient=orient,command=self.tree.yview if orient=="vertical" else self.tree.xview)
            bar.grid(row=0 if orient=="vertical" else 1,column=1 if orient=="vertical" else 0,sticky="ns" if orient=="vertical" else "ew")
            self.tree.configure(**{("yscrollcommand" if orient=="vertical" else "xscrollcommand"):bar.set})
        self.tree.bind("<F2>",lambda event:self.edit_selected_cell())
        self.tree.bind("<Double-1>",self.edit);self.tree.bind("<Button-1>",self.locate)
        self.tree.bind("<Shift-Button-1>",self.rectangle)
        self.tree.bind("<Control-c>",self.copy);self.tree.bind("<Control-v>",self.paste)
        tools=ttk.Frame(self);tools.grid(row=2,column=0,sticky="ew")
        actions=(("Edit cell (F2)",self.edit_selected_cell),("Apply value to selected",self.bulk),("Copy",self.copy),("Paste rectangle",self.paste),
            ("Import CSV",self.import_csv),("Export CSV",self.export_csv),("Load JSON",self.load_json),("Save JSON",self.save_json),
            ("Reset selected",lambda:self.reset(False)),("Reset all",lambda:self.reset(True)),("Preview",self.preview),("Apply",self.apply),("Cancel",self.destroy))
        for i,(label,action) in enumerate(actions):ttk.Button(tools,text=label,command=action).grid(row=i//6,column=i%6,sticky="ew")
        self.figure=Figure(figsize=(8,1.6));self.canvas=FigureCanvasTkAgg(self.figure,master=self)
        self.canvas.get_tk_widget().grid(row=3,column=0,sticky="ew")
        self.status=tk.StringVar(value="Draft only; Apply validates every row.")
        ttk.Label(self,textvariable=self.status,wraplength=1000).grid(row=4,column=0,sticky="w")
        self.refresh();self.transient(parent);self.grab_set()

    def editable(self,key):
        if key in ("start","end","nominal_slope_ui","nominal_slope_per_second","status"):return False
        if self.draft.snapshot.mode!="independent_lanes" and key in ("mean_start","mean_end","sigma_start","sigma_end","weight"):return False
        return True

    def refresh(self):
        try:
            derived=effective_parameters(self.draft.apply())
            for row,value in zip(self.draft.rows,derived):
                for key in ("nominal_slope_ui","nominal_slope_per_second","status","center","kappa"):row[key]=value[key]
        except Exception as exc:
            for row in self.draft.rows:row["status"]=str(exc)
        selected=self.tree.selection();position=self.tree.xview()[0]
        self.tree.delete(*self.tree.get_children())
        for i,row in enumerate(self.draft.rows):self.tree.insert("","end",iid=str(i),values=[row.get(k,"") for k in COLUMNS])
        for i in selected:
            if self.tree.exists(i):self.tree.selection_add(i)
        self.tree.xview_moveto(position)

    def locate(self,event):
        row=self.tree.identify_row(event.y);col=self.tree.identify_column(event.x)
        if row and col:self.anchor=(int(row),int(col[1:])-1);self.corner=None

    def rectangle(self,event):
        row=self.tree.identify_row(event.y);col=self.tree.identify_column(event.x)
        if row and col:
            self.corner=(int(row),int(col[1:])-1)
            self.tree.selection_set(*[str(i) for i in range(min(self.anchor[0],self.corner[0]),max(self.anchor[0],self.corner[0])+1)])
        return "break"

    def parse_value(self,key,value):
        if not self.editable(key):raise ValueError(f"{key} is read-only in this mode")
        if key=="enabled":
            if str(value).lower() not in ("true","1"):raise ValueError("Lane disabling is unsupported; enabled must remain True")
            return True
        if key=="slope_mode":
            if value not in ("kappa","slope"):raise ValueError("slope_mode must be kappa or slope")
            return value
        return None if value in (None,"","None") else float(value)

    def set_value(self,i,key,value):
        self.draft.rows[i][key]=self.parse_value(key,value)
        self.draft.rows[i]["status"]="draft; validation pending"

    def cancel_cell(self,event=None):
        if self.cell_editor is not None:
            box=self.cell_editor[0];self.cell_editor=None;box.destroy()
        return "break"

    def commit_cell(self):
        if self.cell_editor is None:return True
        box,i,key=self.cell_editor
        try:self.set_value(i,key,box.get())
        except (ValueError,TypeError) as exc:
            self.status.set(f"{key}: {exc}. Correct the value or press Escape.");return False
        self.cell_editor=None;box.destroy();self.refresh()
        self.status.set("Cell saved to draft. Apply commits all rows; Cancel discards the draft.")
        return True

    def edit(self,event):
        if not self.tree.identify_row(event.y):return
        self.locate(event);self.edit_selected_cell()

    def edit_selected_cell(self):
        if not self.commit_cell():return
        i,col=self.anchor;key=COLUMNS[col]
        if not self.editable(key):
            self.status.set(f"{key} is read-only. Edit Level Properties/history rules, or use Independent-lane benchmark for direct endpoint editing.");return
        rect=self.tree.bbox(str(i),f"#{col+1}")
        if not rect:
            self.status.set("Scroll the column into view or use the Editable field selector.");return
        box=ttk.Entry(self.tree);value=self.draft.rows[i].get(key,"")
        box.insert(0,"" if value is None else str(value));box.place(x=rect[0],y=rect[1],width=rect[2],height=rect[3])
        self.cell_editor=(box,i,key);box.focus_set();box.select_range(0,"end")
        for event in ("<Return>","<Tab>","<FocusOut>"):box.bind(event,lambda event:self.commit_cell())
        box.bind("<Escape>",self.cancel_cell)

    def edit_chosen_field(self):
        if not self.commit_cell():return
        selection=self.tree.selection();row=int(selection[0]) if selection else self.anchor[0]
        col=COLUMNS.index(self.field_choice.get());self.anchor=(row,col)
        self.tree.see(str(row))
        total=sum(self.tree.column(key,"width") for key in COLUMNS)
        offset=sum(self.tree.column(key,"width") for key in COLUMNS[:col])
        self.tree.xview_moveto(offset/total);self.update_idletasks();self.edit_selected_cell()

    def edit_row_fields(self):
        if not self.commit_cell():return
        selection=self.tree.selection();index=int(selection[0]) if selection else self.anchor[0]
        row=self.draft.rows[index];fields={key:row.get(key) for key in FIELDS}
        editable=[key for key in fields if self.editable(key)]
        def validate(values):return {key:self.parse_value(key,values[key].strip()) for key in editable}
        def save(values):
            self.draft.rows[index].update(values);self.refresh()
            self.status.set("Row saved to draft. Apply commits all rows; Cancel discards the draft.")
        dialog=RecordDialog(self,f"Lane {row['start']} -> {row['end']} — draft fields",fields,editable,validate,save)
        dialog.bind("<Destroy>",lambda event:self.grab_set() if event.widget==dialog and self.winfo_exists() else None,add="+")

    def edit_shared_levels(self):
        if not self.commit_cell():return
        selection=self.tree.selection();index=int(selection[0]) if selection else self.anchor[0]
        row=self.draft.rows[index];ids=sorted({row["start"],row["end"]})
        fields={}
        for level in ids:
            for key in ("mean","sigma"):
                fields[f"level_{level}_{key}"]=self.draft.snapshot.to_dict()["levels"][level][key]
        fields["scope"]="Shared levels: updates every lane using these IDs; endpoint draws remain reused. History overrides still take precedence."
        def validate(values):
            updates={level:{key:values[f"level_{level}_{key}"] for key in ("mean","sigma")} for level in ids}
            from .lane_parameters import update_shared_levels
            return update_shared_levels(self.draft.apply(),updates)
        def save(snapshot):
            self.draft=LaneDraft(snapshot);self.refresh()
            self.status.set("Shared means/sigmas updated in draft for all affected lanes. Apply to commit.")
        dialog=RecordDialog(self,"Edit shared endpoint means/sigmas (all affected lanes)",fields,list(fields)[:-1],validate,save)
        dialog.bind("<Destroy>",lambda event:self.grab_set() if event.widget==dialog and self.winfo_exists() else None,add="+")

    def create_independent_copy(self):
        if not self.commit_cell():return
        if not messagebox.askyesno("Create editable independent-lane copy?",
            "This draft copy allows separate lane endpoint means/sigmas. It generates finite independent lanes, not a continuous waveform or 2UI eye. Shared endpoint mode remains available via presets. Create the copy?",parent=self):return
        try:
            from .lane_parameters import independent_copy
            self.draft=LaneDraft(independent_copy(self.draft.apply()))
            self.mode_note.configure(text="Independent-lane draft copy: endpoint means/sigmas and weights are editable. No continuous waveform or 2UI view.")
            self.field_selector.configure(values=[key for key in FIELDS if self.editable(key)])
            for key in COLUMNS:
                if self.editable(key):self.tree.heading(key,text=key)
            self.refresh();self.status.set("Independent copy is a draft only. Edit row fields, then Apply to activate.")
        except Exception as exc:messagebox.showerror("Cannot create lane copy",str(exc),parent=self)

    def guard(self,action):
        if not self.commit_cell():return
        try:action();self.refresh();self.status.set("Draft updated; Apply validates all rows")
        except Exception as exc:self.status.set(str(exc))

    def bulk(self):
        if not self.commit_cell():return
        key=COLUMNS[self.anchor[1]];value=simpledialog.askstring("Apply to selected",key,parent=self)
        if value is not None:self.guard(lambda:[self.set_value(int(i),key,value) for i in self.tree.selection()])

    def copy(self,event=None):
        if not self.commit_cell():return "break"
        indices=sorted(map(int,self.tree.selection())) or [self.anchor[0]]
        first,last=self.anchor[1],len(COLUMNS)-1
        if self.corner is not None:first,last=sorted((first,self.corner[1]))
        self.clipboard_clear();self.clipboard_append("\n".join("\t".join(str(self.draft.rows[i].get(k,"")) for k in COLUMNS[first:last+1]) for i in indices))
        return "break"

    def paste(self,event=None):
        def action():
            before=copy.deepcopy(self.draft.rows);i,col=self.anchor
            try:
                for di,line in enumerate(self.clipboard_get().splitlines()):
                    for dc,value in enumerate(line.split("\t")):self.set_value(i+di,COLUMNS[col+dc],value)
            except Exception:self.draft.rows=before;raise
        self.guard(action);return "break"

    def import_csv(self):
        path=filedialog.askopenfilename(parent=self,filetypes=[("CSV","*.csv")])
        if path:
            def action():
                with open(path,newline="",encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
                for row in rows:
                    for key in ("start","end"):row[key]=int(row[key])
                raw=self.draft.snapshot.to_dict();raw["lane_overrides"]=rows
                self.draft=LaneDraft(ExperimentConfig.parse(raw))
            self.guard(action)

    def export_csv(self):
        path=filedialog.asksaveasfilename(parent=self,defaultextension=".csv")
        if path:
            def action():
                rows=effective_parameters(self.draft.apply())
                with open(path,"w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=COLUMNS,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
            self.guard(action)

    def load_json(self):
        path=filedialog.askopenfilename(parent=self,filetypes=[("JSON","*.json")])
        if path:
            def action():
                self.draft=LaneDraft(ExperimentConfig.parse(json.loads(Path(path).read_text(encoding="utf-8"))))
                self.field_selector.configure(values=[key for key in FIELDS if self.editable(key)])
            self.guard(action)

    def save_json(self):
        path=filedialog.asksaveasfilename(parent=self,defaultextension=".json")
        if path:self.guard(lambda:Path(path).write_text(json.dumps(self.draft.apply().to_dict(),indent=2),encoding="utf-8"))

    def reset(self,all_rows):
        if not self.commit_cell():return
        rows=effective_parameters(preset(self.draft.snapshot.mode,len(self.draft.snapshot.config.levels)))
        for i in range(len(rows)) if all_rows else map(int,self.tree.selection()):
            for key in FIELDS:
                if self.editable(key):self.draft.rows[i][key]=rows[i][key]
        self.refresh()

    def preview(self):
        def action():
            r=effective_parameters(self.draft.apply())[int((self.tree.selection() or [str(self.anchor[0])])[0])]
            if self.draft.snapshot.mode=="history_endpoints":
                r["mean_start"]=self.draft.snapshot.config.levels[r["start"]];r["mean_end"]=self.draft.snapshot.config.levels[r["end"]]
            u=np.linspace(0,1,1001);c=r["center"];delta=r["mean_end"]-r["mean_start"];middle=(r["mean_start"]+r["mean_end"])/2
            self.figure.clear();ax=self.figure.subplots();ax.plot(u,r["mean_start"]+delta*connector(u,r["kappa"],c))
            ax.scatter([0,c,1],[r["mean_start"],middle,r["mean_end"]]);t=np.array([c-.08,c+.08]);ax.plot(t,middle+r["nominal_slope_ui"]*(t-c),"--")
            ax.set(xlabel="Segment phase [UI]",ylabel=self.draft.snapshot.config.amplitude_unit,title=r["status"]);self.figure.tight_layout();self.canvas.draw_idle()
        self.guard(action)

    def apply(self):
        if not self.commit_cell():return
        try:candidate=self.draft.apply();self.commit(candidate);self.destroy()
        except Exception as exc:self.status.set(str(exc));messagebox.showerror("Lane validation",str(exc),parent=self)
