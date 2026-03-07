# voxel_grid.py
import numpy as np
import pyvista as pv
from dataclasses import dataclass
from typing import List, Tuple


Coord = Tuple[int, int, int]
Index = Tuple[int, int, int]


# ---------- Load points ----------
def _read_points_txt(path: str) -> List[Coord]:
    pts: List[Coord] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for sep in [",", ";", "/"]:
                line = line.replace(sep, " ")
            parts = [p for p in line.split() if p]
            if len(parts) != 3:
                continue
            x, y, z = (int(round(float(p))) for p in parts)
            pts.append((x, y, z))
    return pts


# ---------- Metadata ----------
@dataclass(frozen=True)
class GridInfo:
    origin: Coord
    shape: Tuple[int, int, int]

    def to_index(self, coord: Coord) -> Index:
        ox, oy, oz = self.origin
        x, y, z = coord
        return (x - ox, y - oy, z - oz)

    def to_coord(self, index: Index) -> Coord:
        ox, oy, oz = self.origin
        i, j, k = index
        return (ox + i, oy + j, oz + k)

    def in_bounds(self, index: Index) -> bool:
        nx, ny, nz = self.shape
        i, j, k = index
        return (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz)


# ---------- Helper: shell vs inner ----------
def _split_shell_inner(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


# ---------- Main grid class ----------
class VoxelGrid:
    def __init__(self, grid: np.ndarray, info: GridInfo):
        self.grid = grid
        self.info = info

    def plot_pyvista(self):
        if self.grid.sum() == 0:
            print("Nothing to plot.")
            return

        shell, inner = _split_shell_inner(self.grid)
        nx, ny, nz = self.info.shape
        ox, oy, oz = self.info.origin

        def _surface_from_mask(mask: np.ndarray) -> "pv.PolyData":
            grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1),
                                spacing=(1, 1, 1),
                                origin=(ox, oy, oz))
            grid.cell_data["occ"] = mask.astype(np.uint8).ravel(order="F")
            vol = grid.threshold(0.5, scalars="occ")
            return vol.extract_surface()

        # Create surfaces
        surf_shell = _surface_from_mask(shell)
        surf_inner = _surface_from_mask(inner) if inner.any() else None

        # Plot
        p = pv.Plotter()
        p.add_mesh(surf_shell, color="blue", opacity=0.25, show_edges=False)
        if surf_inner is not None:
            p.add_mesh(surf_inner, color=(0.0, 0.0, 0.6), opacity=0.95, show_edges=False)
        p.show_axes()
        p.show()


# ---------- Loader ----------
def load_voxel_grid(path: str, padding: int = 0) -> VoxelGrid:
    pts = _read_points_txt(path)
    arr = np.array(pts, dtype=int)
    mins = arr.min(axis=0) - padding
    maxs = arr.max(axis=0) + padding
    shape = tuple((maxs - mins + 1).tolist())
    grid = np.zeros(shape, dtype=bool)
    info = GridInfo(origin=tuple(mins.tolist()), shape=shape)
    for p in pts:
        idx = info.to_index(p)
        grid[idx] = True
    return VoxelGrid(grid, info)


# ---------- Hardcoded main ----------
def main():
    path = "data/forest.txt"

    vg = load_voxel_grid(path, padding=0)
    print(f"Origin   : {vg.info.origin}")
    print(f"Shape    : {vg.info.shape}")
    print(f"Occupied : {vg.grid.sum()} cells")

    vg.plot_pyvista()


if __name__ == "__main__":
    main()
