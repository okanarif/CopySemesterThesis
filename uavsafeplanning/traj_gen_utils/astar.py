# astar_3d.py
from __future__ import annotations
import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np
import pyvista as pv

# ==== HARD-CODED SETTINGS=======================================
TUBE_RADIUS = 0.2          # visual thickness of path
# ===============================================================

def split_shell_inner(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (shell_mask, inner_mask) for a boolean occupancy grid.
    shell: occupied voxels that touch any free voxel in 6-neighborhood
    inner: occupied voxels fully surrounded by occupied voxels in 6-neighborhood
    """
    g = grid
    p = np.pad(g, 1, constant_values=False)
    n_all = (
        p[:-2, 1:-1, 1:-1] &
        p[2:,  1:-1, 1:-1] &
        p[1:-1, :-2,  1:-1] &
        p[1:-1, 2:,   1:-1] &
        p[1:-1, 1:-1, :-2 ] &
        p[1:-1, 1:-1, 2:  ]
    )
    inner = g & n_all
    shell = g & (~n_all)
    return shell, inner


# ---------- neighbors for A* ----------
def neighbor_steps(connectivity: int) -> List[Tuple[int, int, int]]:
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")
    steps = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    if connectivity >= 18:
        steps += [
            (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
            (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
            (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1),
        ]
    if connectivity == 26:
        steps += [
            (a,b,c)
            for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)
            if not (a==0 and b==0 and c==0) and (a,b,c) not in steps
        ]
    return steps


def heuristic(a: Tuple[int,int,int], b: Tuple[int,int,int], connectivity: int) -> float:
    if connectivity == 6:
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    dx, dy, dz = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    return (dx*dx + dy*dy + dz*dz) ** 0.5


def astar_3d(grid: np.ndarray,
             start_idx: Tuple[int,int,int],
             goal_idx: Tuple[int,int,int],
             connectivity: int = 6) -> Optional[List[Tuple[int,int,int]]]:
    nx, ny, nz = grid.shape
    in_bounds = lambda n: (0 <= n[0] < nx and 0 <= n[1] < ny and 0 <= n[2] < nz)
    is_free = lambda n: in_bounds(n) and (not grid[n])

    if not is_free(start_idx) or not is_free(goal_idx):
        return None

    steps = neighbor_steps(connectivity)

    def step_cost(d):
        ax = abs(d[0]) + abs(d[1]) + abs(d[2])
        return 1.0 if ax == 1 else (2**0.5 if ax == 2 else 3**0.5)

    def traversable(cur, d):
        n = (cur[0]+d[0], cur[1]+d[1], cur[2]+d[2])
        if not is_free(n):
            return False
        ax = abs(d[0]) + abs(d[1]) + abs(d[2])
        if ax >= 2:
            checks = []
            if d[0] != 0: checks.append((cur[0]+d[0], cur[1], cur[2]))
            if d[1] != 0: checks.append((cur[0], cur[1]+d[1], cur[2]))
            if d[2] != 0: checks.append((cur[0], cur[1], cur[2]+d[2]))
            for c in checks:
                if not is_free(c):
                    return False
        return True

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_idx))
    g_score = {start_idx: 0.0}
    came_from = {}
    f_score = {start_idx: heuristic(start_idx, goal_idx, connectivity)}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        gi = g_score[current]
        for d in steps:
            if not traversable(current, d):
                continue
            n = (current[0]+d[0], current[1]+d[1], current[2]+d[2])
            cand = gi + step_cost(d)
            if cand < g_score.get(n, float("inf")):
                came_from[n] = current
                g_score[n] = cand
                f = cand + heuristic(n, goal_idx, connectivity)
                if f < f_score.get(n, float("inf")):
                    f_score[n] = f
                    heapq.heappush(open_heap, (f, n))
    return None

def line_of_sight_free(grid: np.ndarray,
                       a_idx: tuple[int,int,int],
                       b_idx: tuple[int,int,int],
                       clearance: int = 0) -> bool:
    nx, ny, nz = grid.shape

    if clearance > 0:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = clearance
        g = np.pad(grid, pad, mode="edge")
        win = sliding_window_view(g, (2*pad+1, 2*pad+1, 2*pad+1))
        grid = (win.any(axis=(3,4,5)))

    def inb(i,j,k): return (0<=i<nx and 0<=j<ny and 0<=k<nz)

    ax, ay, az = map(int, a_idx)
    bx, by, bz = map(int, b_idx)
    if not (inb(ax,ay,az) and inb(bx,by,bz)):
        return False
    if grid[ax,ay,az] or grid[bx,by,bz]:
        return False

    p0 = np.array([ax+0.5, ay+0.5, az+0.5], dtype=float)
    p1 = np.array([bx+0.5, by+0.5, bz+0.5], dtype=float)
    d  = p1 - p0

    if np.allclose(d, 0.0):
        return True

    vx, vy, vz = ax, ay, az

    step = np.sign(d).astype(int)
    step[abs(d) < 1e-15] = 0

    tMax = np.zeros(3, dtype=float)
    tDelta = np.empty(3, dtype=float)
    for i, (pi, di, vi, si) in enumerate(zip(p0, d, (vx,vy,vz), step)):
        if si > 0:
            next_boundary = vi + 1.0
            tMax[i] = (next_boundary - pi) / di
            tDelta[i] = 1.0 / di
        elif si < 0:
            next_boundary = vi * 1.0
            tMax[i] = (next_boundary - pi) / di
            tDelta[i] = -1.0 / di
        else:
            tMax[i] = np.inf
            tDelta[i] = np.inf

    while (vx,vy,vz) != (bx,by,bz):
        axis = int(np.argmin(tMax))
        if not np.isfinite(tMax[axis]):
            return False
        if axis == 0:
            vx += step[0]
        elif axis == 1:
            vy += step[1]
        else:
            vz += step[2]
        tMax[axis] += tDelta[axis]

        if not inb(vx,vy,vz):
            return False
        if grid[vx,vy,vz]:
            return False

    return True

def reduce_turns_by_los(turns_idx: np.ndarray,
                        grid: np.ndarray,
                        clearance: int = 0) -> np.ndarray:
    T = np.asarray(turns_idx, dtype=int)
    if T.ndim != 2 or T.shape[1] != 3 or T.shape[0] <= 2:
        return T

    kept = [T[0]]
    i = 0
    N = T.shape[0]

    while True:
        j = i + 1
        last_good = i + 1
        while j < N and line_of_sight_free(grid, tuple(T[i]), tuple(T[j]), clearance=clearance):
            last_good = j
            j += 1
        kept.append(T[last_good])
        if last_good == N - 1:
            break
        i = last_good

    return np.asarray(kept, dtype=int)



# ---------- Visualization with PyVista ----------
def make_surface_from_mask(mask: np.ndarray,
                           origin: Tuple[int,int,int],
                           spacing=(1,1,1)) -> pv.PolyData:
    nx, ny, nz = mask.shape
    ox, oy, oz = origin
    img = pv.ImageData(dimensions=(nx+1, ny+1, nz+1), spacing=spacing, origin=(ox, oy, oz))
    img.cell_data["occ"] = mask.astype(np.uint8).ravel(order="F")
    vol = img.threshold(0.5, scalars="occ")
    return vol.extract_surface()


def visualize(vg, path_coords: Optional[List[Tuple[int,int,int]]]):
    shell, inner = split_shell_inner(vg.grid)

    surf_shell = make_surface_from_mask(shell, vg.info.origin)
    surf_inner = make_surface_from_mask(inner, vg.info.origin) if inner.any() else None

    p = pv.Plotter()
    # Grid shells
    p.add_mesh(surf_shell, color="blue", opacity=0.25, show_edges=False)
    if surf_inner is not None:
        p.add_mesh(surf_inner, color=(0.0, 0.0, 0.6), opacity=0.95, show_edges=False)

    # Path overlay
    if path_coords and len(path_coords) >= 2:
        centers = np.array(path_coords, dtype=float) + 0.5
        centers_world = np.array([vg.info.to_coord(tuple(map(int, c))) for c in centers])
        centers_world = centers_world + 0.5

        line = pv.Spline(centers_world, n_points=len(centers_world))
        tube = line.tube(radius=TUBE_RADIUS, n_sides=16)
        p.add_mesh(tube, color="red", opacity=1.0)

        start_c = np.array(vg.info.to_coord(path_coords[0])) + 0.5
        goal_c  = np.array(vg.info.to_coord(path_coords[-1])) + 0.5
        p.add_mesh(pv.Sphere(radius=TUBE_RADIUS*1.6, center=start_c), color="green")
        p.add_mesh(pv.Sphere(radius=TUBE_RADIUS*1.6, center=goal_c),  color="orange")
    else:
        print("No path found (or too short). Showing grid only.")

    p.show_axes()
    p.show()
