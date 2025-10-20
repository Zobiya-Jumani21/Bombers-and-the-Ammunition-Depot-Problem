import customtkinter as ctk  # type: ignore
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path
import time

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DEFAULT_POLYGON_TEXT = """-504,198
-250,-550
700,-550
700,-150
552,18
700,400
0,950
-500,400
"""
def parse_polygon(text):
    pts = []
    for line in text.strip().splitlines():
        parts = line.replace(',', ' ').split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            pts.append((x, y))
        except:
            continue
    if len(pts) < 3:
        raise ValueError("Polygon must have at least 3 vertices.")
    return np.array(pts)

def points_in_polygon(x_arr, y_arr, polygon):
    pts = np.vstack((x_arr.ravel(), y_arr.ravel())).T
    p = Path(polygon)
    mask_flat = p.contains_points(pts)
    return mask_flat.reshape(x_arr.shape)

def run_simulation(sigma_x, sigma_y, bombs, polygon, seed=None):
    if seed is not None:
        np.random.seed(int(seed))
    Zx = np.random.normal(size=bombs)
    Zy = np.random.normal(size=bombs)
    Xs = Zx * sigma_x
    Ys = Zy * sigma_y
    hit_mask = points_in_polygon(Xs, Ys, polygon) #Tests whether pont is inside polygon
    hits = hit_mask.sum()
    p_hit = hits / bombs
    return {
        "Zx": Zx, "Zy": Zy, "Xs": Xs, "Ys": Ys,
        "hit_mask": hit_mask, "hits": hits, "p_hit": p_hit
    }






#UI
class BombingUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("💣 Monte Carlo Bombing Simulation")
        self.geometry("1300x880")
        self.configure(fg_color="#00bfff")
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, bg="#2a4770", highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(container, orientation="vertical", command=canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(canvas, fg_color="#2a4770")
        frame_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        def resize_scrollable_frame(event):
          canvas.itemconfig(frame_window, width=event.width)
        canvas.bind("<Configure>", resize_scrollable_frame)
        self.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * int(e.delta / 120), "units"))

        
        title = ctk.CTkLabel(self.scrollable_frame, text="Monte Carlo Bombing Simulation",
                             font=("Segoe UI", 26, "bold"), text_color="#4a81ce")
        title.pack(pady=15)

        input_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="#4a81ce", corner_radius=10)
        input_frame.pack(fill="x", padx=20, pady=10)

        # Input Labels 
        ctk.CTkLabel(input_frame, text="σx (m)", text_color="#0b1a2d" ,font=("Segoe UI", 13, "bold")).grid(row=0, column=0, padx=(10, 2), pady=10, sticky="e")
        self.sigma_x = ctk.CTkEntry(input_frame, placeholder_text="σx (m)", text_color="#0b1a2d", fg_color="#CEC4C4",width=100)
        self.sigma_x.grid(row=0, column=1, padx=(0, 15), pady=10)

        ctk.CTkLabel(input_frame, text="σy (m)", text_color="#0b1a2d",font=("Segoe UI", 13, "bold")).grid(row=0, column=2, padx=(10, 2), pady=10, sticky="e")
        self.sigma_y = ctk.CTkEntry(input_frame, placeholder_text="σy (m)", text_color="#0b1a2d",fg_color="#CEC4C4", width=100)
        self.sigma_y.grid(row=0, column=3, padx=(0, 15), pady=10)

        ctk.CTkLabel(input_frame, text="Bombs", text_color="#0b1a2d",font=("Segoe UI", 13, "bold")).grid(row=0, column=4, padx=(10, 2), pady=10, sticky="e")
        self.bombs = ctk.CTkEntry(input_frame, placeholder_text="Bombs",text_color="#0b1a2d", fg_color="#CEC4C4", width=100)
        self.bombs.grid(row=0, column=5, padx=(0, 15), pady=10)

        ctk.CTkLabel(input_frame, text="Random Seed", text_color="#0b1a2d",font=("Segoe UI", 13, "bold")).grid(row=0, column=6, padx=(10, 2), pady=10, sticky="e")
        self.seed = ctk.CTkEntry(input_frame, placeholder_text="Seed", fg_color="#CEC4C4", text_color="#0b1a2d",width=100)
        self.seed.grid(row=0, column=7, padx=(0, 15), pady=10)

        # self.run_btn = ctk.CTkButton(input_frame, text="▶ Run Simulation",
        #                      fg_color="#11AF2C", hover_color="#0099cc",
        #                      command=self.simulate)
        # self.run_btn.grid(row=4, column=8, padx=10, pady=10)

# Default values for inputs
        self.sigma_x.insert(0, "500")
        self.sigma_y.insert(0, "350")
        self.bombs.insert(0, "10")
        self.seed.insert(0, "42")

        # === POLYGON INPUT + DRAWING ===
        poly_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="#10233d", corner_radius=10)
        poly_frame.pack(fill="both", padx=20, pady=10, expand=False)

        text_area_frame = ctk.CTkFrame(poly_frame, fg_color="#0b1a2d")
        text_area_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(text_area_frame, text="Enter Polygon Coordinates:",
                     text_color="#4a81ce", font=("Segoe UI", 19, "bold")).pack(anchor="w", padx=5)
        self.poly_text = tk.Text(text_area_frame, height=10, width=50, font=("Consolas", 13),
                                 bg="white", fg="#4a81ce", insertbackground="white")
        self.poly_text.insert("1.0", DEFAULT_POLYGON_TEXT)
        self.poly_text.pack(fill="x", padx=10, pady=5)

        draw_frame = ctk.CTkFrame(poly_frame, fg_color="#0b1a2d")
        draw_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(draw_frame, text="Draw Polygon (click to add points):",
                     text_color="#4a81ce", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        self.draw_canvas = tk.Canvas(draw_frame, width=480, height=350,
                                     bg="white", highlightthickness=2)
        self.draw_canvas.pack(pady=5)

        btns = ctk.CTkFrame(draw_frame, fg_color="#10233d")
        btns.pack(pady=5)
        ctk.CTkButton(btns, text="Undo Line", fg_color="#11AF2C", hover_color="#007acc",
                      command=self.undo_last, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btns, text="Finish Polygon", fg_color="#11AF2C", hover_color="#007acc",
                      command=self.close_polygon, width=120).pack(side="left", padx=5)
        ctk.CTkButton(btns, text="Clear Drawing", fg_color="#11AF2C", hover_color="#007acc",
                      command=self.clear_drawing, width=120).pack(side="left", padx=5)
        
      # Frame for checkbox + button
        checkbox_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        checkbox_frame.pack(anchor="w", padx=30, pady=(5, 10), fill="x")

# Canvas and polygon logic
        self.draw_canvas.bind("<Button-1>", self.add_point)
        self.points = []
        self.lines = []

# Checkbox
        self.use_drawing = tk.BooleanVar(value=False)
        self.use_drawn_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Use drawn polygon instead of coordinates",variable=self.use_drawing,text_color="white"
)
        self.use_drawn_checkbox.pack(side="left")

#--------------------------- Run Simulation button------------------------------------
        self.run_button = ctk.CTkButton(
    checkbox_frame,
    text="▶ Run Simulation",
    command=self.simulate ,fg_color="#11AF2C"
)
        self.run_button.pack(side="left", padx=15)

        main_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="#10233d", corner_radius=10)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Plot Frame
        plot_frame = ctk.CTkFrame(main_frame, fg_color="#ffffff", corner_radius=10)
        plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.ax.set_facecolor("white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


        # Table Frame
        table_frame = ctk.CTkFrame(main_frame, fg_color="#0B1625", corner_radius=10)
        table_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

       
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background="#152131",
                        foreground="#1E90FF",
                        fieldbackground="#152131",
                        rowheight=25,
                        font=("Consolas", 11))
        style.configure("Treeview.Heading",
                        background="#0B1625",
                        foreground="#1E90FF",
                        font=("Consolas", 12, "bold"))
        style.map("Treeview", background=[("selected", "#1E90FF")])

        

        columns = ("Bomb", "RNNx", "X", "RNNy", "Y", "Result")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=80, anchor="center")
        self.table.pack(fill="both", expand=True, padx=10, pady=10)
        self.table.bind("<Key>", lambda e: "break")

       
        self.stats_label = ctk.CTkLabel(self.scrollable_frame, text="", text_color="#51A0EE", font=("Segoe UI", 15,"bold"))
        self.stats_label.pack(pady=(5, 10))

    # Canvas Buttons
    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        dot = self.draw_canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red")
        if len(self.points) > 1:
            line = self.draw_canvas.create_line(self.points[-2], self.points[-1], fill="black", width=2) 
            self.lines.append(line)

    def undo_last(self):
        if self.lines:
            self.draw_canvas.delete(self.lines.pop())
            self.points.pop()

    def close_polygon(self):
        if len(self.points) > 2:
            line = self.draw_canvas.create_line(self.points[-1], self.points[0], fill="black", width=2)
            self.lines.append(line)

    def clear_drawing(self):
        self.points.clear()
        self.lines.clear()
        self.draw_canvas.delete("all")
    
    def get_polygon_coords(self):
        if len(self.points) < 3: 
            raise ValueError("Please draw at least 3 points.")

        pts = np.array(self.points)
        canvas_w, canvas_h = 480, 350

        x_scaled = (pts[:, 0] / canvas_w) * 1400 - 700   
        y_scaled = ((canvas_h - pts[:, 1]) / canvas_h) * 1500 - 550  

        poly = np.column_stack((x_scaled, y_scaled))
        return poly


    # def get_polygon_coords(self):
    #     if len(self.points) < 3:
    #         raise ValueError("Please draw at least 3 points.")
    #     scale_x, scale_y = 5, 5
    #     poly = [(x * scale_x, (350 - y) * scale_y) for (x, y) in self.points]
    #     return np.array(poly)

    def simulate(self):
        try:
            sigma_x = float(self.sigma_x.get())
            sigma_y = float(self.sigma_y.get())
            bombs = int(self.bombs.get())
            seed = self.seed.get().strip()
            seed = int(seed) if seed else None

            polygon = self.get_polygon_coords() if self.use_drawing.get() \
                else parse_polygon(self.poly_text.get("1.0", "end"))
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        start = time.time()
        stats = run_simulation(sigma_x, sigma_y, bombs, polygon, seed)
        elapsed = time.time() - start

        for row in self.table.get_children():
            self.table.delete(row)
        for i in range(bombs):
            res = "Hit" if stats["hit_mask"][i] else "Miss"
            self.table.insert("", "end", values=(
                i + 1,
                f"{stats['Zx'][i]:.2f}",
                f"{stats['Xs'][i]:.0f}",
                f"{stats['Zy'][i]:.2f}",
                f"{stats['Ys'][i]:.0f}",
                res
            ))

        self.ax.clear()
        poly = np.vstack((polygon, polygon[0]))
        self.ax.fill(poly[:, 0], poly[:, 1], color="lightblue", alpha=0.4, label="Depot Area")
        self.ax.plot(poly[:, 0], poly[:, 1], 'b-', linewidth=1.5, label='Depot Boundary')
        self.ax.scatter(0, 0, color='black', marker='+', s=100, label='Aiming Point')
        self.ax.scatter(stats["Xs"][stats["hit_mask"]], stats["Ys"][stats["hit_mask"]],
                        color='red', s=60, edgecolors='k', label='Hits')
        self.ax.scatter(stats["Xs"][~stats["hit_mask"]], stats["Ys"][~stats["hit_mask"]],
                        color='blue', s=60, edgecolors='k', label='Misses')
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel("Horizontal (m)")
        self.ax.set_ylabel("Vertical (m)")
        self.ax.set_title("Bombing Simulation Scatter Plot", fontsize=13)
        self.canvas.draw()

        self.stats_label.configure(
            text=f"Simulated {bombs} bombs | Hits = {stats['hits']} | "
                 f"Hit Probability = {stats['p_hit']:.2f} | Time = {elapsed:.3f}s"
        )
        print(stats)


if __name__ == "__main__":
    app = BombingUI()
    app.mainloop()

