"""Smoke-test our Tk lane editor, including draft cancellation and successful Apply."""
import json
from pathlib import Path
import tkinter as tk
from dual_tail_characterization.gui import PamApplication
from dual_tail_characterization.lane_editor import LaneEditor
from dual_tail_characterization.experiment_config import preset
out=Path("runs/lane_connector_demo");out.mkdir(parents=True,exist_ok=True)
root=tk.Tk();app=PamApplication(root);errors=[]
root.report_callback_exception=lambda kind,value,trace:errors.append(str(value))
assert app.gather_snapshot().config.shared_kappa==10
assert all(value==.02 for value in app.gather_snapshot().config.endpoint_sigmas)
app.preset_var.set("PAM2 editable independent lanes");app.choose_preset()
assert all(row["sigma"]==.02 and row["kappa"]==10 for row in app.gather_snapshot().to_dict()["lanes"])
app.load_snapshot(preset());before=app.gather_snapshot();commits=[]
editor=LaneEditor(root,before,commits.append);root.update()
editor.set_value(3,"phase_shift",".04");editor.destroy();root.update()
assert not commits and app.gather_snapshot()==before
editor=LaneEditor(root,before,commits.append)
editor.set_value(3,"phase_shift",".04")
# Exercise an actual Entry and Apply without Enter, rather than bypassing widgets.
from dual_tail_characterization.lane_editor import COLUMNS
editor.tree.xview_moveto(.35);root.update()
editor.anchor=(3,COLUMNS.index("kappa"))
editor.edit_selected_cell();root.update()
assert editor.cell_editor is not None
entry=editor.cell_editor[0];entry.delete(0,"end");entry.insert(0,"12")
assert editor.commit_cell()
assert editor.draft.rows[3]["kappa"]==12
editor.edit_selected_cell();root.update()
entry=editor.cell_editor[0];entry.delete(0,"end");entry.insert(0,"13")
# Apply must commit this still-open Entry, even without an Enter key press.
editor.apply();root.update()
assert len(commits)==1 and commits[0].to_dict()["lane_overrides"][3]["kappa"]==13
commits.clear()
editor=LaneEditor(root,before,commits.append)
editor.set_value(3,"phase_shift",".04");editor.set_value(3,"kappa","12")
editor.tree.selection_set("3");editor.preview();root.update()
import time
time.sleep(.5);root.update()
from PIL import ImageGrab
ImageGrab.grab(bbox=(editor.winfo_rootx(),editor.winfo_rooty(),editor.winfo_rootx()+editor.winfo_width(),editor.winfo_rooty()+editor.winfo_height())).save(out/"lane_editor.png")
editor.apply();root.update();assert len(commits)==1
app.load_snapshot(commits[0]);assert app.gather_snapshot()==commits[0]
# Hidden columns are reachable without manually scrolling.
editor=LaneEditor(root,before,lambda candidate:None)
editor.tree.selection_set("3");editor.field_choice.set("phase_shift")
editor.edit_chosen_field();root.update()
assert editor.cell_editor is not None and editor.cell_editor[2]=="phase_shift"
entry=editor.cell_editor[0];entry.delete(0,"end");entry.insert(0,".06")
assert editor.commit_cell() and editor.draft.rows[3]["phase_shift"]==.06
editor.destroy();root.update()
# Direct endpoint editing is available through the independent-mode row form.
editor=LaneEditor(root,preset("independent_lanes"),lambda candidate:None)
editor.tree.selection_set("3");editor.edit_row_fields();root.update()
from dual_tail_characterization.gui_widgets import RecordDialog
from tkinter import ttk
form=next(w for w in editor.winfo_children() if isinstance(w,RecordDialog))
def walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from walk(child)
entries=[w for w in walk(form) if isinstance(w,ttk.Entry)]
mean_entry=entries[list(__import__("dual_tail_characterization.lane_parameters",fromlist=["FIELDS"]).FIELDS).index("mean_start")]
assert "readonly" not in mean_entry.state()
mean_entry.delete(0,"end");mean_entry.insert(0,".15")
next(w for w in walk(form) if isinstance(w,ttk.Button) and w.cget("text")=="Apply").invoke();root.update()
assert editor.draft.rows[3]["mean_start"]==.15
editor.destroy();root.update()
# PAM2 preset resets formerly invalid PAM4 lane selection and generates a real 2UI view.
app.preset_var.set("NRZ / PAM2 alternating");app.choose_preset();root.update()
assert len(app.gather_snapshot().config.levels)==2
assert app.plot_settings().lane_end==1
app.plot_vars["view"].set("2UI eye");app.generate()
deadline=time.monotonic()+30
while app.controller.busy and time.monotonic()<deadline:
    root.update();time.sleep(.02)
assert not app.controller.busy
assert any(record.data is not None and len(record.snapshot.config.levels)==2 for record in app.controller.records.values())
# Shared endpoint mean/sigma dialog changes shared levels without converting mode.
editor=LaneEditor(root,app.gather_snapshot(),lambda candidate:None)
editor.tree.selection_set("1");editor.edit_shared_levels();root.update()
form=next(w for w in editor.winfo_children() if isinstance(w,RecordDialog))
entries=[w for w in walk(form) if isinstance(w,ttk.Entry)]
entries[0].delete(0,"end");entries[0].insert(0,"-.2")
entries[1].delete(0,"end");entries[1].insert(0,".03")
next(w for w in walk(form) if isinstance(w,ttk.Button) and w.cget("text")=="Apply").invoke();root.update()
assert editor.draft.snapshot.config.levels[0]==-.2
assert editor.draft.rows[0]["mean_start"]==-.2
assert editor.draft.rows[1]["sigma_start"]==.03
editor.destroy();root.update()
assert not errors
(out/"gui_smoke_report.json").write_text(json.dumps(dict(status="pass",checks=["Cancel preserves active inputs","Apply commits validated draft","selected lane preview","configuration round trip","actual Entry editing and Apply without Enter","hidden midpoint field selector","independent endpoint row form","PAM2 preset and 2UI generation","shared mean/sigma dialog","GUI defaults sigma 0.02 and kappa 10"],callback_errors=errors),indent=2))
app.close();print("Lane editor GUI smoke passed")
