import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button

class WorkspaceApp:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height

        self.obstacles = []      # Lista obstacolelor poligonale finalizate
        self.current_pts = []    # Vârfurile obstacolului în curs de desenare

        # Inițializarea ferestrei grafice
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.15)

        # Conectarea evenimentului de click al mouse-ului
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Crearea butoanelor pentru alegerea metodei și resetare
        ax_hom = plt.axes([0.10, 0.03, 0.28, 0.06])
        ax_tree = plt.axes([0.41, 0.03, 0.28, 0.06])
        ax_clear = plt.axes([0.72, 0.03, 0.18, 0.06])

        self.btn_hom = Button(ax_hom, '1. Descompunere Omogenă', color='#e0f0ff', hovercolor='#b0d8ff')
        self.btn_tree = Button(ax_tree, '2. Metoda Arborelui', color='#ffe0e0', hovercolor='#ffb0b0')
        self.btn_clear = Button(ax_clear, 'Șterge Tot', color='#f0f0f0', hovercolor='#d0d0d0')

        self.btn_hom.on_clicked(self.select_homogeneous)
        self.btn_tree.on_clicked(self.select_tree)
        self.btn_clear.on_clicked(self.clear_all)

        self.redraw_canvas()

    def setup_axes(self, title):
        """Configurează limitele și aspectul spațiului de lucru."""
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title, fontsize=11, fontweight='bold')

    def redraw_canvas(self, title=None):
        """Redesenează spațiul de lucru și obstacolele."""
        if title is None:
            title = "Click Stânga: adaugă vârf | Click Dreapta: închide obstacolul"
        self.setup_axes(title)

        # Desenăm obstacolele finalizate
        for obs in self.obstacles:
            poly = MplPolygon(obs, closed=True, facecolor='#0080ff',
                              edgecolor='black', linewidth=1.2)
            self.ax.add_patch(poly)

        # Desenăm liniile obstacolului aflat în curs de definire
        if self.current_pts:
            pts = np.array(self.current_pts)
            self.ax.plot(pts[:, 0], pts[:, 1], 'ro--', linewidth=1.2, markersize=5)

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Gestionează desenarea obstacolelor cu mouse-ul."""
        if event.inaxes != self.ax:
            return

        # Click Stânga: adaugă un vârf nou
        if event.button == 1:
            self.current_pts.append([event.xdata, event.ydata])
            self.redraw_canvas()

        # Click Dreapta: închide poligonul (minim 3 vârfuri)
        elif event.button == 3:
            if len(self.current_pts) >= 3:
                self.obstacles.append(np.array(self.current_pts))
                self.current_pts = []
                self.redraw_canvas()
            else:
                print("Un obstacol are nevoie de cel puțin 3 vârfuri!")

    # =================================================================
    # FUNCȚIILE DE SELECȚIE (Urmează să fie completate cu algoritmii)
    # =================================================================

    def select_homogeneous(self, event=None):
        self.current_pts = []
        self.redraw_canvas(title="Mod selectat: Descompunere Omogenă")
        print("A fost selectată Descompunerea Omogenă.")
        # TODO: Aici se va adăuga codul pentru descompunerea omogenă

    def select_tree(self, event=None):
        self.current_pts = []
        self.redraw_canvas(title="Mod selectat: Metoda Arborelui")
        print("A fost selectată Metoda Arborelui.")
        # TODO: Aici se va adăuga codul pentru metoda arborelui

    def clear_all(self, event=None):
        """Resetează spațiul de lucru."""
        self.obstacles = []
        self.current_pts = []
        self.redraw_canvas()


if __name__ == "__main__":
    app = WorkspaceApp(width=100, height=100)
    plt.show()