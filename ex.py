"""
Polygonal Obstacle Pathfinding Visualizer + Quad-Tree Method
===========================================================
Interaction modes:
  ✏ Draw      : left click = add vertex | right click = close polygon
  ✥ Move      : drag obstacle to move it
  ⬡ Edit      : drag a polygon vertex
  ✖ Delete    : click obstacle to remove it
  🔵 Start     : click to place the start point
  🔴 Goal      : click to place the goal point

Algorithms: BFS, Dijkstra, A* (with/without diagonals), Quad-Tree Method
Buttons: ▶ Run Full | ⏭ Step | ⏯ Auto | ↺ Reset | 🗑 Clear All
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Polygon as MplPolygon, Circle, FancyArrowPatch
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
from matplotlib.path import Path as MplPath
import heapq
from collections import deque

# ═════════════════════════════════════════════════════════════════════════════
# Constante globale
# ═════════════════════════════════════════════════════════════════════════════
GRID_W, GRID_H = 30, 30
CELL = 1.0
WORLD_W, WORLD_H = GRID_W * CELL, GRID_H * CELL

COL_GRID     = '#cccccc'
COL_FREE     = '#ffffff'
COL_BLOCKED  = '#2255bb'
COL_VISITED  = '#ffe0a0'
COL_FRONTIER = '#ffcc00'
COL_PATH     = '#00cc44'
COL_START    = '#1144ff'
COL_GOAL     = '#dd0000'
COL_OBS      = '#2255cc'
COL_OBS_EDGE = '#000033'
COL_HOVER    = '#44aaff'
COL_DRAW     = '#ff8800'

# Quad-tree culori (liber / parțial / plin)
QT_FREE    = '#e8ffe8'
QT_MIXED   = '#fff0a0'
QT_FULL    = '#ffaaaa'

# ═════════════════════════════════════════════════════════════════════════════
# ─── Geometrie: intersecție segment–celulă (robustă pentru poligoane subțiri)
# ═════════════════════════════════════════════════════════════════════════════

def _segments_intersect(p1, p2, p3, p4):
    """True dacă segmentul p1-p2 intersectează segmentul p3-p4."""
    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross2d(p3, p4, p1)
    d2 = cross2d(p3, p4, p2)
    d3 = cross2d(p1, p2, p3)
    d4 = cross2d(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # coliniare / capete pe segment
    def on_seg(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))
    if d1 == 0 and on_seg(p3, p1, p4): return True
    if d2 == 0 and on_seg(p3, p2, p4): return True
    if d3 == 0 and on_seg(p1, p3, p2): return True
    if d4 == 0 and on_seg(p1, p4, p2): return True
    return False


def poly_blocks_cell(poly_pts, col, row, cell_size=CELL):
    """
    Returnează True dacă poligonul blochează celula (col, row).
    Verifică:
      1. Centrul celulei în poligon (ray-casting)
      2. Oricare colț al celulei în poligon
      3. Oricare latură a poligonului intersectează oricare latură a celulei
      4. Oricare vârf al poligonului în celulă
    Acoperă și poligoane subțiri / linii.
    """
    x0 = col * cell_size
    y0 = row * cell_size
    x1 = min(x0 + cell_size, WORLD_W)
    y1 = min(y0 + cell_size, WORLD_H)

    # 1 & 2 — ray-casting cu matplotlib Path
    mpl_path = MplPath(poly_pts)
    test_pts = [
        ((x0 + x1) / 2, (y0 + y1) / 2),  # centru
        (x0, y0), (x1, y0), (x1, y1), (x0, y1),  # colțuri
        # mijlocul fiecărei laturi a celulei
        ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1),
        (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2),
    ]
    if any(mpl_path.contains_point(p) for p in test_pts):
        return True

    # 3 — laturile poligonului vs. laturile celulei
    cell_edges = [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]
    n = len(poly_pts)
    for i in range(n):
        pa = tuple(poly_pts[i])
        pb = tuple(poly_pts[(i + 1) % n])
        for (ca, cb) in cell_edges:
            if _segments_intersect(pa, pb, ca, cb):
                return True

    # 4 — vârfurile poligonului în interiorul celulei (inclusiv margini)
    for pt in poly_pts:
        if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1:
            return True

    return False


def point_in_polygon(px, py, poly_pts):
    return MplPath(poly_pts).contains_point((px, py))


def dist_to_vertices(px, py, poly_pts):
    dists = [np.hypot(px - p[0], py - p[1]) for p in poly_pts]
    return min(dists), int(np.argmin(dists))


# ═════════════════════════════════════════════════════════════════════════════
# ─── Quad-Tree (Metoda Arborelui)
# ═════════════════════════════════════════════════════════════════════════════

class QuadNode:
    """Un nod în quad-tree-ul de descompunere."""
    FREE   = 0  # celulă complet liberă
    FULL   = 1  # celulă complet blocată
    MIXED  = 2  # celulă mixtă (se subdivizează)

    def __init__(self, x0, y0, x1, y1, depth=0):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.depth = depth
        self.state = QuadNode.FREE
        self.children = []   # 4 copii dacă MIXED

    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2


def _cell_fully_blocked(poly_pts, x0, y0, x1, y1, samples=4):
    """Verifică dacă o celulă rectangulară e complet în interiorul poligonului."""
    xs = np.linspace(x0, x1, samples + 1)
    ys = np.linspace(y0, y1, samples + 1)
    mpl_path = MplPath(poly_pts)
    for xi in xs:
        for yi in ys:
            if not mpl_path.contains_point((xi, yi)):
                return False
    return True


def _rect_intersects_poly(poly_pts, x0, y0, x1, y1):
    """True dacă dreptunghiul [x0,x1]×[y0,y1] intersectează poligonul."""
    # Folosim aceeași logică ca poly_blocks_cell dar pentru dreptunghi arbitrar
    mpl_path = MplPath(poly_pts)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    test_pts = [
        (cx, cy),
        (x0, y0), (x1, y0), (x1, y1), (x0, y1),
        ((x0 + x1)/2, y0), ((x0 + x1)/2, y1),
        (x0, (y0 + y1)/2), (x1, (y0 + y1)/2),
    ]
    if any(mpl_path.contains_point(p) for p in test_pts):
        return True
    cell_edges = [
        ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)),
    ]
    n = len(poly_pts)
    for i in range(n):
        pa = tuple(poly_pts[i])
        pb = tuple(poly_pts[(i + 1) % n])
        for (ca, cb) in cell_edges:
            if _segments_intersect(pa, pb, ca, cb):
                return True
    for pt in poly_pts:
        if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1:
            return True
    return False


def build_quadtree(obstacles, x0, y0, x1, y1, max_depth=6, depth=0):
    """Construiește recursiv un arbore cu celule libere / ocupate / mixte."""
    node = QuadNode(x0, y0, x1, y1, depth)

    if not obstacles:
        node.state = QuadNode.FREE
        return node

    def rect_has_intersection():
        return any(_rect_intersects_poly(obs, x0, y0, x1, y1) for obs in obstacles)

    def rect_fully_blocked():
        xs = np.linspace(x0, x1, 5)
        ys = np.linspace(y0, y1, 5)
        for xi in xs:
            for yi in ys:
                if not any(MplPath(obs).contains_point((xi, yi)) for obs in obstacles):
                    return False
        return True

    if not rect_has_intersection():
        node.state = QuadNode.FREE
        return node

    if rect_fully_blocked():
        node.state = QuadNode.FULL
        return node

    if depth >= max_depth:
        node.state = QuadNode.MIXED
        return node

    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    node.state = QuadNode.MIXED
    node.children = [
        build_quadtree(obstacles, x0, y0, mx, my, max_depth, depth + 1),
        build_quadtree(obstacles, mx, y0, x1, my, max_depth, depth + 1),
        build_quadtree(obstacles, x0, my, mx, y1, max_depth, depth + 1),
        build_quadtree(obstacles, mx, my, x1, y1, max_depth, depth + 1),
    ]

    states = [c.state for c in node.children]
    if all(s == QuadNode.FREE for s in states):
        node.state = QuadNode.FREE
        node.children = []
    elif all(s == QuadNode.FULL for s in states):
        node.state = QuadNode.FULL
        node.children = []
    return node


def collect_qt_leaves(node, leaves=None):
    if leaves is None:
        leaves = []
    if not node.children:
        leaves.append(node)
    else:
        for c in node.children:
            collect_qt_leaves(c, leaves)
    return leaves


def collect_qt_nodes_to_depth(node, max_depth, nodes=None):
    if nodes is None:
        nodes = []
    if node.depth >= max_depth or not node.children:
        nodes.append(node)
    else:
        for child in node.children:
            collect_qt_nodes_to_depth(child, max_depth, nodes)
    return nodes


def find_qt_leaf(node, x, y):
    if not (node.x0 <= x <= node.x1 and node.y0 <= y <= node.y1):
        return None
    if not node.children:
        return node
    for child in node.children:
        leaf = find_qt_leaf(child, x, y)
        if leaf is not None:
            return leaf
    return None


def run_quadtree(root, start, goal, cell_size=CELL):
    leaves = collect_qt_leaves(root)
    start_leaf = find_qt_leaf(
        root, (start[0] + 0.5) * cell_size, (start[1] + 0.5) * cell_size)
    goal_leaf = find_qt_leaf(
        root, (goal[0] + 0.5) * cell_size, (goal[1] + 0.5) * cell_size)
    if (start_leaf is None or goal_leaf is None
            or start_leaf.state != QuadNode.FREE
            or goal_leaf.state != QuadNode.FREE):
        return {}, [], []

    free_leaves = [leaf for leaf in leaves if leaf.state == QuadNode.FREE]
    neighbors = {leaf: [] for leaf in free_leaves}
    epsilon = 1e-9
    for index, first in enumerate(free_leaves):
        for second in free_leaves[index + 1:]:
            vertical_touch = (
                abs(first.x1 - second.x0) < epsilon
                or abs(second.x1 - first.x0) < epsilon
            )
            horizontal_overlap = min(first.y1, second.y1) - max(first.y0, second.y0)
            horizontal_touch = (
                abs(first.y1 - second.y0) < epsilon
                or abs(second.y1 - first.y0) < epsilon
            )
            vertical_overlap = min(first.x1, second.x1) - max(first.x0, second.x0)
            if vertical_touch and horizontal_overlap > epsilon:
                neighbors[first].append(second)
                neighbors[second].append(first)
            elif horizontal_touch and vertical_overlap > epsilon:
                neighbors[first].append(second)
                neighbors[second].append(first)

    def heuristic(leaf):
        return np.hypot(leaf.cx - goal_leaf.cx, leaf.cy - goal_leaf.cy)

    costs = {start_leaf: 0.0}
    came_from = {start_leaf: None}
    visited_order = []
    frontier_snap = []
    queue = [(heuristic(start_leaf), 0.0, 0, start_leaf)]
    sequence = 0
    visited = set()

    while queue:
        _, cost, _, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        visited_order.append(current)
        if current is goal_leaf:
            break
        for neighbor in neighbors[current]:
            next_cost = cost + np.hypot(
                current.cx - neighbor.cx, current.cy - neighbor.cy)
            if next_cost < costs.get(neighbor, float('inf')):
                costs[neighbor] = next_cost
                came_from[neighbor] = current
                sequence += 1
                heapq.heappush(
                    queue,
                    (next_cost + heuristic(neighbor), next_cost, sequence, neighbor))
                frontier_snap.append((len(visited_order), neighbor))
    return came_from, visited_order, frontier_snap


# ═════════════════════════════════════════════════════════════════════════════
# ─── Algoritmi de căutare
# ═════════════════════════════════════════════════════════════════════════════

def get_neighbors(col, row, grid_blocked, diagonal=True):
    grid_h = len(grid_blocked)
    grid_w = len(grid_blocked[0]) if grid_h else 0
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diagonal:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    result = []
    for dc, dr in dirs:
        nc, nr = col+dc, row+dr
        if 0 <= nc < grid_w and 0 <= nr < grid_h and not grid_blocked[nr][nc]:
            # Verificare colț tăiat pentru diagonale
            if dc != 0 and dr != 0:
                if grid_blocked[row][nc] or grid_blocked[nr][col]:
                    continue  # nu traversa colț blocat
            cost = 1.4142 if (dc != 0 and dr != 0) else 1.0
            result.append((nc, nr, cost))
    return result


def run_bfs(grid_blocked, start, goal, diagonal):
    queue = deque([start])
    came_from = {start: None}
    visited_order, frontier_snap = [], []
    while queue:
        cur = queue.popleft()
        visited_order.append(cur)
        if cur == goal:
            break
        for nc, nr, _ in get_neighbors(cur[0], cur[1], grid_blocked, diagonal):
            nb = (nc, nr)
            if nb not in came_from:
                came_from[nb] = cur
                queue.append(nb)
                frontier_snap.append((len(visited_order), nb))
    return came_from, visited_order, frontier_snap


def run_dijkstra(grid_blocked, start, goal, diagonal):
    g = {start: 0.0}
    came_from = {start: None}
    visited_order, frontier_snap = [], []
    pq = [(0.0, start)]
    visited_set = set()
    while pq:
        d, cur = heapq.heappop(pq)
        if cur in visited_set:
            continue
        visited_set.add(cur)
        visited_order.append(cur)
        if cur == goal:
            break
        for nc, nr, cost in get_neighbors(cur[0], cur[1], grid_blocked, diagonal):
            nb = (nc, nr)
            nd = d + cost
            if nd < g.get(nb, float('inf')):
                g[nb] = nd
                came_from[nb] = cur
                heapq.heappush(pq, (nd, nb))
                frontier_snap.append((len(visited_order), nb))
    return came_from, visited_order, frontier_snap


def run_astar(grid_blocked, start, goal, diagonal):
    def h(a, b):
        dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
        return (1.4142*min(dx,dy) + abs(dx-dy)) if diagonal else (dx+dy)

    g = {start: 0.0}
    came_from = {start: None}
    visited_order, frontier_snap = [], []
    pq = [(h(start, goal), 0.0, start)]
    visited_set = set()
    while pq:
        _, d, cur = heapq.heappop(pq)
        if cur in visited_set:
            continue
        visited_set.add(cur)
        visited_order.append(cur)
        if cur == goal:
            break
        for nc, nr, cost in get_neighbors(cur[0], cur[1], grid_blocked, diagonal):
            nb = (nc, nr)
            ng = d + cost
            if ng < g.get(nb, float('inf')):
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(pq, (ng + h(nb, goal), ng, nb))
                frontier_snap.append((len(visited_order), nb))
    return came_from, visited_order, frontier_snap


def reconstruct_path(came_from, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    return path


# ═════════════════════════════════════════════════════════════════════════════
# ─── Aplicația principală
# ═════════════════════════════════════════════════════════════════════════════

class PathfindingApp:
    def __init__(self):
        # ── Stare ──────────────────────────────────────────────────────────
        self.obstacles   = []
        self.current_pts = []
        self.mode        = 'Draw'
        self.algo        = 'A*'
        self.diagonal    = True
        self.show_qt     = False   # afișare quad-tree
        self.qt_depth    = 4
        self.cell_size   = CELL

        self.start = (2, 2)
        self.goal  = (GRID_W - 3, GRID_H - 3)

        self.grid_blocked = [[False]*GRID_W for _ in range(GRID_H)]

        # Drag / edit
        self._drag_idx    = None
        self._drag_offset = None
        self._edit_idx    = None
        self._edit_vtx    = None
        self._hover_idx   = None

        # Vizualizare
        self._shown_visited  = []
        self._shown_frontier = []
        self._shown_path     = []
        self.visited_order   = []
        self.frontier_snap   = []
        self.path_cells      = []
        self.came_from       = {}
        self._anim_step      = 0
        self._auto_running   = False
        self._qt_root        = None
        self._qt_display_depth = -1

        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.titlesize': 9,
            'axes.labelsize': 8,
        })

        # ── Figura ─────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(15, 9), facecolor='#f5f5f5')
        self.fig.patch.set_facecolor('#f0f0f0')
        try:
            self.fig.canvas.manager.set_window_title('Pathfinding Visualizer')
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.01, 0.06, 0.71, 0.92])
        self._build_widgets()

        self.fig.canvas.mpl_connect('button_press_event',   self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)

        self._redraw()

    # ──────────────────────────────────────────────────────────────────────
    # Widget-uri
    # ──────────────────────────────────────────────────────────────────────
    def _build_widgets(self):
        px, pw = 0.745, 0.245

        # Modes
        ax_m = self.fig.add_axes([px, 0.70, pw, 0.28])
        self.radio_mode = RadioButtons(ax_m,
            ['Draw', 'Move', 'Edit vertex', 'Delete', 'Start', 'Goal'],
            active=0)
        self.radio_mode.on_clicked(self._on_mode)
        ax_m.set_title('Mode', fontsize=9, fontweight='bold', pad=2)

        # Algorithms
        ax_a = self.fig.add_axes([px, 0.50, pw, 0.18])
        self.radio_algo = RadioButtons(ax_a,
            ['BFS', 'Dijkstra', 'A*', 'Quad-Tree'], active=2)
        self.radio_algo.on_clicked(self._on_algo)
        ax_a.set_title('Algorithm / Method', fontsize=9, fontweight='bold', pad=2)

        # Options
        ax_o = self.fig.add_axes([px, 0.42, pw, 0.07])
        self.chk_opts = CheckButtons(
            ax_o, ['Grid diagonals', 'QT view'], [True, False])
        self.chk_opts.on_clicked(self._on_opts)

        ax_resolution = self.fig.add_axes(
            [px + 0.01, 0.38, pw - 0.02, 0.025])
        self.cell_size_slider = Slider(
            ax_resolution, 'Cell size', 0.5, 3.0,
            valinit=self.cell_size, valstep=0.5)
        self.cell_size_slider.on_changed(self._on_cell_size)
        ax_resolution.set_visible(True)

        ax_depth = self.fig.add_axes(
            [px + 0.01, 0.345, pw - 0.02, 0.025])
        self.qt_depth_slider = Slider(
            ax_depth, 'Iterations', 0, 6, valinit=self.qt_depth, valstep=1)
        self.qt_depth_slider.on_changed(self._on_qt_depth)
        ax_depth.set_visible(False)

        # Butoane acțiuni
        self.action_buttons = []
        for i, (label, col, handler, row_y) in enumerate([
            ('Run Full',    '#c8f0c8', self._run_full,    0.305),
            ('Step',        '#c8d8f8', self._step_once,   0.250),
            ('Auto',        '#f8e8c8', self._toggle_auto, 0.195),
            ('Reset',       '#f0f0f0', self._reset_viz,   0.140),
            ('Clear All',   '#f8c8c8', self._clear_all,   0.085),
        ]):
            ax_b = self.fig.add_axes([px, row_y, pw, 0.045])
            btn = Button(ax_b, label, color=col,
                         hovercolor=col.replace('c8','90').replace('f0','d0'))
            btn.on_clicked(handler)
            self.action_buttons.append(btn)

        # Info box
        ax_i = self.fig.add_axes([px, 0.005, pw, 0.075])
        ax_i.set_facecolor('#eeeeee')
        ax_i.axis('off')
        self._info_text = ax_i.text(
            0.04, 0.97, 'Ready.',
            transform=ax_i.transAxes, va='top', ha='left',
            fontsize=7.5, family='monospace')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks widget-uri
    # ──────────────────────────────────────────────────────────────────────
    def _on_mode(self, label):
        m = {'Draw':'Draw','Move':'Move','Edit vertex':'Edit',
             'Delete':'Delete','Start':'SetStart','Goal':'SetGoal'}
        self.mode = m.get(label, 'Draw')
        self.current_pts = []
        self._drag_idx = self._edit_idx = None
        self._redraw()

    def _on_algo(self, label):
        self.algo = label
        self.qt_depth_slider.ax.set_visible(label == 'Quad-Tree')
        self.cell_size_slider.ax.set_visible(label != 'Quad-Tree')
        if label == 'Quad-Tree':
            self.action_buttons[0].label.set_text('Build + Run')
            self.action_buttons[1].label.set_text('Split / Step')
            self.action_buttons[2].label.set_text('Split / Auto')
        else:
            self.action_buttons[0].label.set_text('Run Full')
            self.action_buttons[1].label.set_text('Step')
            self.action_buttons[2].label.set_text('Auto')
        self._reset_viz()

    def _on_cell_size(self, value):
        old_size = self.cell_size
        self.cell_size = float(value)
        if self.cell_size == old_size:
            return

        def remap_cell(cell):
            world_x = (cell[0] + 0.5) * old_size
            world_y = (cell[1] + 0.5) * old_size
            return (
                min(self._grid_columns - 1, int(world_x / self.cell_size)),
                min(self._grid_rows - 1, int(world_y / self.cell_size)),
            )

        self.start = remap_cell(self.start)
        self.goal = remap_cell(self.goal)
        self._rebuild_grid()
        self._reset_viz()
        self._set_info(f'Cell size: {self.cell_size:g}')

    def _on_qt_depth(self, value):
        self.qt_depth = int(value)
        if self.algo == 'Quad-Tree':
            self._reset_viz()

    def _on_opts(self, label):
        status = self.chk_opts.get_status()
        self.diagonal  = status[0]
        self.show_qt   = status[1]
        self._reset_viz()

    @property
    def _grid_columns(self):
        return int(np.ceil(WORLD_W / self.cell_size))

    @property
    def _grid_rows(self):
        return int(np.ceil(WORLD_H / self.cell_size))

    # ──────────────────────────────────────────────────────────────────────
    # Grila de celule blocate (robustă)
    # ──────────────────────────────────────────────────────────────────────
    def _rebuild_grid(self):
        self.grid_blocked = [
            [False] * self._grid_columns for _ in range(self._grid_rows)]
        for obs in self.obstacles:
            pts = obs
            min_c = max(0, int(np.floor(pts[:, 0].min() / self.cell_size)))
            max_c = min(
                self._grid_columns - 1,
                int(np.floor(pts[:, 0].max() / self.cell_size)))
            min_r = max(0, int(np.floor(pts[:, 1].min() / self.cell_size)))
            max_r = min(
                self._grid_rows - 1,
                int(np.floor(pts[:, 1].max() / self.cell_size)))
            for r in range(min_r, max_r+1):
                for c in range(min_c, max_c+1):
                    if poly_blocks_cell(pts, c, r, self.cell_size):
                        self.grid_blocked[r][c] = True

    # ──────────────────────────────────────────────────────────────────────
    # Quad-Tree
    # ──────────────────────────────────────────────────────────────────────
    def _rebuild_quadtree(self):
        self._qt_root = build_quadtree(
            self.obstacles, 0, 0, GRID_W*CELL, GRID_H*CELL,
            max_depth=self.qt_depth)

    # ──────────────────────────────────────────────────────────────────────
    # Pathfinding
    # ──────────────────────────────────────────────────────────────────────
    def _prepare_search(self):
        s, g = self.start, self.goal

        if self.algo == 'Quad-Tree':
            self._rebuild_quadtree()
            cf, vo, fs = run_quadtree(
                self._qt_root, s, g, self.cell_size)
            goal_leaf = find_qt_leaf(
                self._qt_root,
                (g[0] + 0.5) * self.cell_size,
                (g[1] + 0.5) * self.cell_size)
            self.came_from = cf
            self.visited_order = vo
            self.frontier_snap = fs
            self.path_cells = (
                reconstruct_path(cf, goal_leaf)
                if vo and vo[-1] is goal_leaf else [])
            if not vo:
                self._set_info('Start or goal is blocked at this tree depth.')
                return False
            self._anim_step = 0
            self._shown_visited = []
            self._shown_frontier = []
            self._shown_path = []
            return True

        self._rebuild_grid()
        def valid(col, row):
            return (0 <= col < self._grid_columns
                    and 0 <= row < self._grid_rows
                    and not self.grid_blocked[row][col])

        if not valid(*s):
            self._set_info('Start is blocked!'); return False
        if not valid(*g):
            self._set_info('Goal is blocked!'); return False

        if self.algo == 'BFS':
            cf, vo, fs = run_bfs(self.grid_blocked, s, g, self.diagonal)
        elif self.algo == 'Dijkstra':
            cf, vo, fs = run_dijkstra(self.grid_blocked, s, g, self.diagonal)
        elif self.algo == 'A*':
            cf, vo, fs = run_astar(self.grid_blocked, s, g, self.diagonal)
        else:
            cf, vo, fs = run_astar(self.grid_blocked, s, g, self.diagonal)

        self.came_from     = cf
        self.visited_order = vo
        self.frontier_snap = fs
        self.path_cells    = reconstruct_path(cf, g) if g in cf else []
        self._anim_step    = 0
        self._shown_visited  = []
        self._shown_frontier = []
        self._shown_path     = []
        return True

    def _run_full(self, event=None):
        self._auto_running = False
        if self.algo == 'Quad-Tree':
            self._qt_display_depth = self.qt_depth
        if not self._prepare_search():
            if self.algo == 'Quad-Tree':
                self._redraw()
            return
        self._shown_visited  = list(self.visited_order)
        self._shown_frontier = []
        self._shown_path     = list(self.path_cells)
        self._anim_step      = len(self.visited_order) + 1
        self._redraw()
        found = bool(self.path_cells)
        path_unit = 'leaves' if self.algo == 'Quad-Tree' else 'steps'
        movement = (
            'Movement: shared edges' if self.algo == 'Quad-Tree'
            else f'Grid diagonals: {"On" if self.diagonal else "Off"}')
        self._set_info(
            f'Algorithm: {self.algo}\n'
            f'Visited: {len(self._shown_visited)}\n'
            f'Path length: {len(self._shown_path)} {path_unit}\n'
            f'{movement}\n'
            f'{"Path found!" if found else "No path found."}')

    def _step_once(self, event=None):
        if self.algo == 'Quad-Tree' and self._qt_display_depth < self.qt_depth:
            if self._qt_root is None:
                self._rebuild_quadtree()
            self._qt_display_depth += 1
            self._shown_visited = []
            self._shown_frontier = []
            self._shown_path = []
            self._set_info(
                f'Subdivision level {self._qt_display_depth} / {self.qt_depth}'
                f'\nShowing {len(collect_qt_nodes_to_depth(self._qt_root, self._qt_display_depth))} cells')
            self._redraw()
            return

        if not self.visited_order:
            if not self._prepare_search():
                return

        step  = self._anim_step
        total = len(self.visited_order)

        if step < total:
            self._shown_visited = self.visited_order[:step+1]
            vset = set(self._shown_visited)
            # frontiera = celulele adăugate în coadă până la acest pas
            self._shown_frontier = [
                nb for (added_at, nb) in self.frontier_snap
                if added_at <= step and nb not in vset
            ]
            self._shown_path = []
            self._anim_step += 1
            self._set_info(
                f'Step {step+1}/{total}\n'
                f'Algorithm: {self.algo}\n'
                f'Frontier: {len(self._shown_frontier)}')
        else:
            self._shown_visited  = list(self.visited_order)
            self._shown_frontier = []
            self._shown_path     = list(self.path_cells)
            self._set_info(
                f'Completed. Visited: {total}\n'
                f'Path length: {len(self.path_cells)}\n'
                f'{"Path found!" if self.path_cells else "No path found."}')
        self._redraw()

    def _toggle_auto(self, event=None):
        if self._auto_running:
            self._auto_running = False
            return
        showing_tree = (
            self.algo == 'Quad-Tree'
            and self._qt_display_depth < self.qt_depth
        )
        if not showing_tree and not self.visited_order:
            if not self._prepare_search():
                return
        self._auto_running = True
        self._fire_auto_timer()

    def _fire_auto_timer(self):
        if not self._auto_running:
            return
        if (self.algo == 'Quad-Tree'
                and self._qt_display_depth < self.qt_depth):
            self._qt_display_depth += 1
            self._redraw()
            if self._qt_display_depth >= self.qt_depth:
                self._auto_running = False
                self._set_info(
                    f'Decomposition complete: level {self.qt_depth}. '
                    'Press Auto again to animate the search.')
                return
            self._set_info(
                f'Subdivision level {self._qt_display_depth} / {self.qt_depth}')
            t = self.fig.canvas.new_timer(interval=700)
            t.single_shot = True
            t.add_callback(self._fire_auto_timer)
            t.start()
            return
        if self._shown_path:
            self._auto_running = False
            return
        if self._anim_step >= len(self.visited_order):
            self._step_once()
            self._auto_running = False
            return
        self._step_once()
        t = self.fig.canvas.new_timer(interval=55)
        t.single_shot = True
        t.add_callback(self._fire_auto_timer)
        t.start()

    def _reset_viz(self, event=None):
        self._auto_running   = False
        if self.algo == 'Quad-Tree':
            self._qt_display_depth = -1
        if self.algo == 'Quad-Tree' or self.show_qt:
            self._rebuild_quadtree()
        self.visited_order   = []
        self.frontier_snap   = []
        self.path_cells      = []
        self.came_from       = {}
        self._anim_step      = 0
        self._shown_visited  = []
        self._shown_frontier = []
        self._shown_path     = []
        self._set_info('Reset.')
        self._redraw()

    def _clear_all(self, event=None):
        self.obstacles   = []
        self.current_pts = []
        self._qt_root    = None
        self._reset_viz()

    # ──────────────────────────────────────────────────────────────────────
    # Mouse events
    # ──────────────────────────────────────────────────────────────────────
    def _xy(self, ev):
        return (ev.xdata, ev.ydata) if ev.inaxes == self.ax else (None, None)

    def _grid_cell(self, x, y):
        c = max(0, min(self._grid_columns - 1, int(np.floor(x/self.cell_size))))
        r = max(0, min(self._grid_rows - 1, int(np.floor(y/self.cell_size))))
        return c, r

    def _obs_at(self, x, y):
        for i, obs in enumerate(self.obstacles):
            if point_in_polygon(x, y, obs):
                return i
        return None

    def _on_press(self, event):
        x, y = self._xy(event)
        if x is None:
            return

        if self.mode == 'Draw':
            if event.button == 1:
                self.current_pts.append([x, y])
                self._redraw()
            elif event.button == 3:
                if len(self.current_pts) >= 3:
                    self.obstacles.append(np.array(self.current_pts, dtype=float))
                    self.current_pts = []
                    self._rebuild_grid()
                    self._reset_viz()
                else:
                    self._set_info('Minimum 3 vertices required for a polygon!')

        elif self.mode == 'Move' and event.button == 1:
            idx = self._obs_at(x, y)
            if idx is not None:
                ctr = self.obstacles[idx].mean(axis=0)
                self._drag_idx    = idx
                self._drag_offset = np.array([x - ctr[0], y - ctr[1]])

        elif self.mode == 'Edit' and event.button == 1:
            best_d, best_i, best_v = float('inf'), None, None
            for i, obs in enumerate(self.obstacles):
                d, vi = dist_to_vertices(x, y, obs)
                if d < 1.5 and d < best_d:
                    best_d, best_i, best_v = d, i, vi
            if best_i is not None:
                self._edit_idx = best_i
                self._edit_vtx = best_v

        elif self.mode == 'Delete' and event.button == 1:
            idx = self._obs_at(x, y)
            if idx is not None:
                del self.obstacles[idx]
                self._rebuild_grid()
                self._reset_viz()
                self._set_info(f'Obstacle #{idx} removed.')

        elif self.mode == 'SetStart' and event.button == 1:
            c, r = self._grid_cell(x, y)
            self.start = (c, r)
            self._reset_viz()
            self._set_info(f'Start: ({c}, {r})')

        elif self.mode == 'SetGoal' and event.button == 1:
            c, r = self._grid_cell(x, y)
            self.goal = (c, r)
            self._reset_viz()
            self._set_info(f'Goal: ({c}, {r})')

    def _on_release(self, event):
        if self._drag_idx is not None or self._edit_idx is not None:
            self._rebuild_grid()
            self._reset_viz()
        self._drag_idx = self._edit_idx = self._edit_vtx = None
        self._drag_offset = None

    def _on_motion(self, event):
        x, y = self._xy(event)
        if x is None:
            return
        changed = False

        if self.mode == 'Move' and self._drag_idx is not None:
            ctr   = self.obstacles[self._drag_idx].mean(axis=0)
            delta = np.array([x, y]) - self._drag_offset - ctr
            self.obstacles[self._drag_idx] += delta
            changed = True

        elif self.mode == 'Edit' and self._edit_idx is not None:
            self.obstacles[self._edit_idx][self._edit_vtx] = [x, y]
            changed = True

        elif self.mode in ('Move', 'Edit', 'Delete'):
            new_h = self._obs_at(x, y)
            if new_h != self._hover_idx:
                self._hover_idx = new_h
                changed = True

        if changed:
            self._redraw()

    # ──────────────────────────────────────────────────────────────────────
    # Desenare
    # ──────────────────────────────────────────────────────────────────────
    def _redraw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, WORLD_W)
        ax.set_ylim(0, WORLD_H)
        ax.set_aspect('equal')
        ax.set_facecolor(COL_FREE)
        tree_mode = self.algo == 'Quad-Tree'
        vset = set(self._shown_visited)
        fset = set(self._shown_frontier)
        pset = set(self._shown_path)

        if not tree_mode:
            for c in range(self._grid_columns + 1):
                x = min(c * self.cell_size, WORLD_W)
                ax.axvline(x, color=COL_GRID, lw=0.4, zorder=1)
            for r in range(self._grid_rows + 1):
                y = min(r * self.cell_size, WORLD_H)
                ax.axhline(y, color=COL_GRID, lw=0.4, zorder=1)

        if (tree_mode or self.show_qt) and self._qt_root:
            if tree_mode:
                tree_nodes = (
                    collect_qt_nodes_to_depth(
                        self._qt_root, self._qt_display_depth)
                    if self._qt_display_depth >= 0 else [])
            else:
                tree_nodes = collect_qt_leaves(self._qt_root)
            for leaf in tree_nodes:
                if leaf.state == QuadNode.FREE:
                    fc, ec = QT_FREE,  '#aaddaa'
                elif leaf.state == QuadNode.FULL:
                    fc, ec = QT_FULL,  '#dd8888'
                else:
                    fc, ec = QT_MIXED, '#ddcc44'
                if tree_mode:
                    if leaf in pset:
                        fc, ec = COL_PATH, '#008833'
                    elif leaf in vset:
                        fc, ec = COL_VISITED, '#ccaa66'
                    elif leaf in fset:
                        fc, ec = COL_FRONTIER, '#cc9900'
                w = leaf.x1 - leaf.x0
                h = leaf.y1 - leaf.y0
                rect = plt.Rectangle(
                    (leaf.x0, leaf.y0), w, h,
                    facecolor=fc, edgecolor=ec, lw=1.5,
                    alpha=0.55, zorder=6 if tree_mode else 2)
                ax.add_patch(rect)

        if not tree_mode:
            for r in range(self._grid_rows):
                for c in range(self._grid_columns):
                    cell = (c, r)
                    if self.grid_blocked[r][c]:
                        color = COL_BLOCKED
                    elif cell in pset:
                        color = COL_PATH
                    elif cell in vset:
                        color = COL_VISITED
                    elif cell in fset:
                        color = COL_FRONTIER
                    else:
                        continue
                    x0, y0 = c * self.cell_size, r * self.cell_size
                    ax.add_patch(plt.Rectangle(
                        (x0, y0),
                        min(self.cell_size, WORLD_W - x0),
                        min(self.cell_size, WORLD_H - y0),
                        facecolor=color, edgecolor='none',
                        alpha=0.78, zorder=3))

        # ── Obstacole poligonale ────────────────────────────────────────
        for i, obs in enumerate(self.obstacles):
            hover = (i == self._hover_idx and
                     self.mode in ('Move', 'Edit', 'Delete'))
            poly = MplPolygon(
                obs, closed=True,
                facecolor=COL_HOVER if hover else COL_OBS,
                edgecolor=COL_OBS_EDGE,
                lw=2.0 if hover else 1.2,
                alpha=0.88, zorder=5)
            ax.add_patch(poly)
            if self.mode == 'Edit':
                ax.scatter(obs[:,0], obs[:,1],
                           c='yellow', edgecolors='black', s=55, zorder=9)

        # ── Poligon în curs de desenare ─────────────────────────────────
        if self.current_pts:
            pts = np.array(self.current_pts)
            ax.plot(pts[:,0], pts[:,1], 'o--',
                    color=COL_DRAW, lw=1.4, ms=5, zorder=8)
            if len(pts) >= 3:
                cl = np.vstack([pts, pts[0]])
                ax.fill(cl[:,0], cl[:,1],
                        facecolor=COL_DRAW, alpha=0.18, zorder=7)

        # ── Drum final cu săgeți și numere ──────────────────────────────
        sp = self._shown_path
        if tree_mode and sp:
            px_ = [(self.start[0] + 0.5) * self.cell_size]
            py_ = [(self.start[1] + 0.5) * self.cell_size]
            px_.extend(leaf.cx for leaf in sp)
            py_.extend(leaf.cy for leaf in sp)
            px_.append((self.goal[0] + 0.5) * self.cell_size)
            py_.append((self.goal[1] + 0.5) * self.cell_size)
            ax.plot(
                px_, py_, '-', color=COL_PATH, lw=3.5, marker='o',
                markersize=4, markeredgecolor='white', markeredgewidth=1.2,
                zorder=12,
                path_effects=[
                    path_effects.Stroke(linewidth=6, foreground='white'),
                    path_effects.Normal(),
                ])
            for i in range(len(px_) - 1):
                ax.annotate(
                    '', xy=(px_[i + 1], py_[i + 1]),
                    xytext=(px_[i], py_[i]),
                    arrowprops=dict(
                        arrowstyle='->', color=COL_PATH, lw=1.8,
                        path_effects=[
                            path_effects.Stroke(linewidth=4, foreground='white'),
                            path_effects.Normal(),
                        ]),
                    zorder=13)
        elif len(sp) >= 2:
            px_ = [(c + 0.5) * self.cell_size for c, r in sp]
            py_ = [(r + 0.5) * self.cell_size for c, r in sp]
            ax.plot(
                px_, py_, '-', color=COL_PATH, lw=3.5, marker='o',
                markersize=4, markeredgecolor='white', markeredgewidth=1.2,
                zorder=12,
                path_effects=[
                    path_effects.Stroke(linewidth=6, foreground='white'),
                    path_effects.Normal(),
                ])
            # săgeți
            step = max(1, len(sp)//25)   # nu desena prea multe săgeți
            for i in range(0, len(sp)-1, step):
                ax.annotate('',
                    xy=(px_[i+1], py_[i+1]),
                    xytext=(px_[i], py_[i]),
                    arrowprops=dict(
                        arrowstyle='->', color=COL_PATH, lw=1.8,
                        path_effects=[
                            path_effects.Stroke(linewidth=4, foreground='white'),
                            path_effects.Normal(),
                        ]),
                    zorder=13)

        # ── Start & Goal ────────────────────────────────────────────────
        for (c,r), col, lbl in [
            (self.start, COL_START, 'S'),
            (self.goal,  COL_GOAL,  'G'),
        ]:
            cx, cy = (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size
            ax.add_patch(Circle((cx, cy), self.cell_size*0.38,
                                color=col, zorder=13))
            ax.text(cx, cy, lbl, ha='center', va='center',
                    color='white', fontsize=7,
                    fontweight='bold', zorder=14)

        # ── Titlu ───────────────────────────────────────────────────────
        titles = {
            'Draw':     'Draw  [left click = vertex | right click = close polygon]',
            'Move':     'Move obstacle  [drag]',
            'Edit':     'Edit vertex  [drag yellow vertex]',
            'Delete':   'Delete obstacle  [click]',
            'SetStart': 'Place Start  [click grid]',
            'SetGoal':  'Place Goal   [click grid]',
        }
        title = titles.get(self.mode, '')
        if tree_mode:
            if self._qt_display_depth < 0:
                title = 'Quad-Tree decomposition  [ready to split]'
            else:
                title = (
                    f'Quad-Tree decomposition  '
                    f'[level {self._qt_display_depth}/{self.qt_depth}]')
        else:
            title += f'  [cell size={self.cell_size:g}]'
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks(range(0, GRID_W+1, 5))
        ax.set_yticks(range(0, GRID_H+1, 5))
        ax.tick_params(labelsize=7)

        # ── Legendă ─────────────────────────────────────────────────────
        if tree_mode:
            legend_items = [
                mpatches.Patch(facecolor=QT_FREE, label='Free leaf'),
                mpatches.Patch(facecolor=QT_FULL, label='Occupied leaf'),
                mpatches.Patch(facecolor=QT_MIXED, label='Mixed (blocked)'),
                mpatches.Patch(facecolor=COL_VISITED, label='Visited'),
                mpatches.Patch(facecolor=COL_PATH, label='Path'),
            ]
        else:
            legend_items = [
                mpatches.Patch(facecolor=COL_BLOCKED,  label='Blocked'),
                mpatches.Patch(facecolor=COL_VISITED,  label='Visited'),
                mpatches.Patch(facecolor=COL_FRONTIER, label='Frontier'),
                mpatches.Patch(facecolor=COL_PATH,     label='Optimal path'),
            ]
            if self.show_qt:
                legend_items += [
                    mpatches.Patch(facecolor=QT_FREE,  label='QT: free'),
                    mpatches.Patch(facecolor=QT_MIXED, label='QT: mixed'),
                    mpatches.Patch(facecolor=QT_FULL,  label='QT: full'),
                ]
        ax.legend(handles=legend_items, loc='lower right',
                  fontsize=7, framealpha=0.85)

        self.fig.canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────────
    def _set_info(self, text):
        if hasattr(self, '_info_text'):
            self._info_text.set_text(text)
            self.fig.canvas.draw_idle()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = PathfindingApp()
    plt.show()
