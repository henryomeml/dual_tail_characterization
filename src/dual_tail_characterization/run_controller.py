"""Run ownership and worker coordination. No widget calls on worker threads."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import queue
import uuid
from .experiment_backend import generate_experiment


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    token: str
    snapshot: object
    plot_settings: object
    output_directory: str
    status: str
    created_at: str
    completed_at: str | None = None
    data: object = None
    error: str | None = None


class RunController:
    def __init__(self):
        self.records={}
        self.selected_id=None
        self.events=queue.Queue()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="pam-worker")
        self.closed=False
        self.busy=False
        self.active_job=None
        self.active_operation=None
        self.current_snapshot_json=None

    def mark_inputs(self,snapshot):
        self.current_snapshot_json=snapshot.snapshot_json if snapshot is not None else None

    def is_stale(self,record):
        return self.current_snapshot_json!=record.snapshot.snapshot_json

    def start(self,snapshot,plot_settings,output_root):
        if self.busy or self.closed:
            raise RuntimeError("A worker is already active or the application is closed")
        run_id=uuid.uuid4().hex
        output=str(Path(output_root)/f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id[:8]}")
        record=RunRecord(run_id,uuid.uuid4().hex,snapshot,plot_settings,output,"running",now())
        self.records[run_id]=record
        self.selected_id=run_id
        self.mark_inputs(snapshot)
        self.busy=True
        self.active_job=(run_id,record.token)
        self.active_operation="generation"
        def worker():
            try:
                result=generate_experiment(snapshot,lambda message:self.events.put((run_id,record.token,"log",message)))
                self.events.put((run_id,record.token,"generated",result))
            except Exception as exc:
                self.events.put((run_id,record.token,"failed",f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put((run_id,record.token,"idle",None))
        self.executor.submit(worker)
        return record

    def export_selected(self):
        if self.busy or self.closed:
            raise RuntimeError("A worker is already active or the application is closed")
        record=self.records.get(self.selected_id)
        if record is None or record.data is None:
            raise ValueError("Select a successfully generated run first")
        self.busy=True
        self.active_job=(record.run_id,record.token)
        self.active_operation="export"
        def worker():
            try:
                from .experiment_export import export_run
                export_run(record)
                self.events.put((record.run_id,record.token,"exported",record.output_directory))
            except Exception as exc:
                self.events.put((record.run_id,record.token,"export_failed",f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put((record.run_id,record.token,"idle",None))
        self.executor.submit(worker)

    def accept(self,event):
        run_id,token,kind,payload=event
        if self.closed:
            return False
        if kind=="idle" and self.active_job==(run_id,token):
            self.busy=False
            self.active_job=None
            self.active_operation=None
        record=self.records.get(run_id)
        if record is None or record.token!=token:
            return False
        if kind=="generated":
            self.records[run_id]=replace(record,data=payload,status="complete",completed_at=now())
        elif kind=="failed":
            self.records[run_id]=replace(record,status="failed",error=payload,completed_at=now())
        elif kind=="exported":
            self.records[run_id]=replace(record,status="exported",error=None)
        elif kind=="export_failed":
            self.records[run_id]=replace(record,error=payload)
        return True

    def set_plot(self,run_id,settings):
        record=self.records[run_id]
        self.records[run_id]=replace(record,plot_settings=settings)

    def remove(self,run_id):
        self.records.pop(run_id,None)
        if self.selected_id==run_id:
            self.selected_id=next(reversed(self.records),None)

    def close(self):
        self.closed=True
        self.records.clear()
        self.executor.shutdown(wait=False,cancel_futures=True)
