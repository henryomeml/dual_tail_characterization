"""Interactive-display smoke test of our own Tk application (no external UI control).

Run explicitly: python scripts/gui_smoke_test.py
Outputs are confined to runs/gui_smoke. Screenshots cover this app's window only.
"""
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
from dual_tail_characterization.gui import PamApplication
from dual_tail_characterization.experiment_config import ExperimentConfig, preset
from dual_tail_characterization.gui_widgets import RecordDialog

OUTPUT=Path("runs/gui_smoke")
OUTPUT.mkdir(parents=True,exist_ok=True)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def main():
    try:
        root=tk.Tk()
    except tk.TclError as exc:
        (OUTPUT/"smoke_report.json").write_text(json.dumps(dict(status="unavailable",reason=str(exc)),indent=2))
        print(f"GUI execution unavailable: {exc}")
        return
    app=PamApplication(root)
    errors=[]
    app.root.report_callback_exception=lambda kind,value,trace:errors.append(f"{kind.__name__}: {value}")
    def fail_dialog(title,message,**kwargs):
        raise AssertionError(f"Unexpected GUI error dialog: {title}: {message}")
    import dual_tail_characterization.gui as gui_module
    import dual_tail_characterization.gui_widgets as widgets_module
    gui_module.messagebox.showerror=fail_dialog
    widgets_module.messagebox.showerror=fail_dialog
    def pump_until(predicate,timeout=120):
        deadline=time.monotonic()+timeout
        while not predicate():
            root.update()
            if errors:
                raise AssertionError(errors)
            if time.monotonic()>deadline:
                raise TimeoutError("Tk smoke-test operation timed out")
            time.sleep(.02)
        root.update()
    checks=[]
    try:
        root.geometry("1280x820")
        root.update()
        root.tk.call("tk","scaling",1.5)
        ids=[]
        for mode in ("shared_endpoints","history_endpoints","continuous_filter","independent_lanes"):
            raw=preset(mode).to_dict()
            raw["base"].update(n_symbols=160,samples_per_ui=32)
            raw["output_root"]=str(OUTPUT)
            app.load_snapshot(ExperimentConfig.parse(raw))
            app.plot_vars["time_bins_per_ui"].set("16")
            if mode=="shared_endpoints":
                app.edit_row("levels",0)
                root.update()
                dialog=next(w for w in root.winfo_children() if isinstance(w,RecordDialog))
                sigma_entry=next(w for w in descendants(dialog) if isinstance(w,ttk.Entry) and w.get()=="0.2")
                sigma_entry.delete(0,"end")
                sigma_entry.insert(0,"0.23")
                next(w for w in descendants(dialog) if isinstance(w,ttk.Button) and w.cget("text")=="Apply").invoke()
                assert app.gather_snapshot().config.endpoint_sigmas[0]==.23
                assert preset().config.endpoint_sigmas[0]==.2
                checks.append("dedicated row editing preserves original preset")
            if mode=="history_endpoints":
                app.edit_tables["contexts"].set_rows([dict(previous=0,current=0,next=0,mean=.4,sigma=.05)])
            app.generate()
            pump_until(lambda:not app.controller.busy)
            record=app.controller.records[app.controller.selected_id]
            assert record.status=="complete",record.error
            ids.append(record.run_id)
            checks.append(f"generate {mode}")
            before=record.data.arrays["segments"].copy()
            app.change_view("Lane histogram")
            app.plot_vars["selection"].set("lane")
            app.replot()
            np.testing.assert_array_equal(before,record.data.arrays["segments"])
            checks.append(f"histogram / no-regeneration replot {mode}")
        # Select the original shared run while editor inputs still contain the
        # independent preset. Export must retain the stored Option-1 settings.
        app.history.selection_set(ids[0])
        root.update()
        app._select_history()
        assert app.controller.selected_id==ids[0]
        assert app.controller.is_stale(app.controller.records[ids[0]])
        app.export_selected()
        pump_until(lambda:not app.controller.busy)
        selected=app.controller.records[ids[0]]
        saved=json.loads((Path(selected.output_directory)/"gui_config.json").read_text())
        assert saved["mode"]=="shared_endpoints" and saved["levels"][0]["sigma"]==.23
        assert (Path(selected.output_directory)/"validation_results.xlsx").is_file()
        checks.append("run switching restores data; selected-run Excel export is isolated from edited inputs")
        # History run export includes its rule, overrides and observed contexts.
        app.history.selection_set(ids[1])
        root.update()
        app._select_history()
        app.change_view("2UI eye")
        app.plot_vars["selection"].set("full")
        app.replot()
        app.export_selected()
        pump_until(lambda:not app.controller.busy)
        selected=app.controller.records[ids[1]]
        assert (Path(selected.output_directory)/"contexts_prescribed.csv").is_file()
        checks.append("history context Excel export")
        app.load_run_inputs()
        config_path=OUTPUT/"saved_gui_config.json"
        gui_module.filedialog.asksaveasfilename=lambda **kwargs:str(config_path)
        app.save_json()
        gui_module.filedialog.askopenfilename=lambda **kwargs:str(config_path)
        app.load_json()
        assert app.gather_snapshot()==selected.snapshot
        checks.append("configuration JSON round trip")
        # Inspect compact layout and take a screenshot of this window only.
        root.geometry("1000x700")
        root.update()
        app.notebook.select(1)
        app.validation_section.set("Lane theory / empirical")
        app._show_validation()
        root.update()
        assert app.generate_button.winfo_ismapped() and app.export_button.winfo_ismapped()
        screenshot_status="unavailable"
        try:
            from PIL import ImageGrab
            left,top=root.winfo_rootx(),root.winfo_rooty()
            ImageGrab.grab(bbox=(left,top,left+root.winfo_width(),top+root.winfo_height())).save(OUTPUT/"gui_smoke.png")
            screenshot_status="saved own application window"
        except (ImportError,OSError) as exc:
            screenshot_status=str(exc)
        report=dict(status="pass",checks=checks,callback_errors=errors,screenshot=screenshot_status,
                    generated_runs=[app.controller.records[k].output_directory for k in ids])
        (OUTPUT/"smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2))
    except Exception as exc:
        (OUTPUT/"smoke_report.json").write_text(json.dumps(dict(status="failed",reason=str(exc),checks=checks,callback_errors=errors),indent=2),encoding="utf-8")
        raise
    finally:
        app.close()


if __name__=="__main__":
    main()
