"""Reusable scrollable tables and edit dialogs. No scientific calculations."""
import tkinter as tk
from tkinter import ttk, messagebox


class ScrollFrame(ttk.Frame):
    def __init__(self,parent):
        super().__init__(parent)
        self.canvas=tk.Canvas(self,highlightthickness=0)
        scrollbar=ttk.Scrollbar(self,orient="vertical",command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.body=ttk.Frame(self.canvas)
        self.window=self.canvas.create_window(0,0,anchor="nw",window=self.body)
        self.body.bind("<Configure>",lambda event:self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",lambda event:self.canvas.itemconfigure(self.window,width=event.width))
        self.canvas.grid(row=0,column=0,sticky="nsew")
        scrollbar.grid(row=0,column=1,sticky="ns")
        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)
        def wheel(event):
            self.canvas.yview_scroll(-int(event.delta/120),"units")
        self.canvas.bind("<MouseWheel>",wheel)


class RecordDialog(tk.Toplevel):
    def __init__(self,parent,title,row,editable,validate,apply):
        super().__init__(parent)
        self.title(title)
        self.transient(parent.winfo_toplevel())
        self.geometry("460x520")
        self.minsize(340,280)
        scroll=ScrollFrame(self)
        scroll.pack(fill="both",expand=True,padx=10,pady=8)
        variables={}
        for index,(key,value) in enumerate(row.items()):
            ttk.Label(scroll.body,text=key).grid(row=index,column=0,sticky="w",pady=4)
            variable=tk.StringVar(value="" if value is None else value)
            variables[key]=variable
            entry=ttk.Entry(scroll.body,textvariable=variable)
            entry.grid(row=index,column=1,sticky="ew",padx=8,pady=4)
            if key not in editable:
                entry.state(["readonly"])
        scroll.body.columnconfigure(1,weight=1)
        def save():
            try:
                values=validate({key:var.get() for key,var in variables.items()})
                apply(values)
                self.destroy()
            except ValueError as exc:
                messagebox.showerror("Invalid row",str(exc),parent=self)
        buttons=ttk.Frame(self)
        buttons.pack(fill="x",padx=10,pady=10)
        ttk.Button(buttons,text="Apply",command=save).pack(side="right")
        ttk.Button(buttons,text="Cancel",command=self.destroy).pack(side="right",padx=6)
        self.bind("<Return>",lambda event:save())
        self.bind("<Escape>",lambda event:self.destroy())
        self.grab_set()


class DataTable(ttk.Frame):
    """Paged table keeps large endpoint/context tables responsive and complete."""
    def __init__(self,parent,edit=None,add=None,delete=None):
        super().__init__(parent)
        self.rows=[]
        self.columns=[]
        self.page=0
        self.page_size=100
        self.edit_callback=edit
        self.tree=ttk.Treeview(self,show="headings",selectmode="browse")
        xbar=ttk.Scrollbar(self,orient="horizontal",command=self.tree.xview)
        ybar=ttk.Scrollbar(self,orient="vertical",command=self.tree.yview)
        self.tree.configure(xscrollcommand=xbar.set,yscrollcommand=ybar.set)
        self.tree.grid(row=0,column=0,sticky="nsew")
        ybar.grid(row=0,column=1,sticky="ns")
        xbar.grid(row=1,column=0,sticky="ew")
        controls=ttk.Frame(self)
        controls.grid(row=2,column=0,columnspan=2,sticky="ew")
        self.note=ttk.Label(controls)
        self.note.pack(side="left",padx=4)
        ttk.Button(controls,text="‹",width=3,command=lambda:self.turn(-1)).pack(side="right")
        ttk.Button(controls,text="›",width=3,command=lambda:self.turn(1)).pack(side="right")
        self.action_buttons=[]
        for text,command in (("Edit",self.edit_selected if edit else None),("Add",add),("Delete",delete)):
            if command:
                button=ttk.Button(controls,text=text,command=command)
                button.pack(side="right",padx=2)
                self.action_buttons.append(button)
        self.tree.bind("<Double-1>",lambda event:self.edit_selected())
        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)

    def set_rows(self,rows,columns=None):
        self.rows=list(rows)
        self.columns=list(columns or dict.fromkeys(key for row in self.rows for key in row))
        self.page=0
        self.tree.configure(columns=self.columns)
        for key in self.columns:
            self.tree.heading(key,text=key)
            self.tree.column(key,width=max(100,min(240,9*len(key)+16)),minwidth=70,stretch=False)
        self.redraw()

    def redraw(self):
        self.tree.delete(*self.tree.get_children())
        first=self.page*self.page_size
        for index,row in enumerate(self.rows[first:first+self.page_size],first):
            self.tree.insert("","end",iid=str(index),values=["" if row.get(k) is None else row.get(k) for k in self.columns])
        self.note.configure(text=f"Rows {first+1 if self.rows else 0}–{min(len(self.rows),first+self.page_size)} / {len(self.rows)}")

    def turn(self,delta):
        self.page=max(0,min(max(0,(len(self.rows)-1)//self.page_size),self.page+delta))
        self.redraw()

    def selected_index(self):
        selection=self.tree.selection()
        return int(selection[0]) if selection else None

    def edit_selected(self):
        index=self.selected_index()
        if index is not None and self.edit_callback:
            self.edit_callback(index)
