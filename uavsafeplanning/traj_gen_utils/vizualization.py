import numpy as np
import pyvista as pv

def pv_surface_from_grid(mask3d: np.ndarray, origin=(0,0,0)):
        nx, ny, nz = mask3d.shape
        ox, oy, oz = origin
        img = pv.ImageData(dimensions=(nx+1, ny+1, nz+1),
                        spacing=(1,1,1),
                        origin=(ox, oy, oz))
        
        img.cell_data["occ"] = mask3d.astype(np.uint8).ravel(order="F")
        vol = img.threshold(0.5, scalars="occ")
        return vol.extract_surface()

def visualize_voxelgrid_with_path(
        vg,
        path_idx=None,
        path_xyz=None,
        plotPath=True,
        tube_radius=0.2,
        grid_opacity=0.1,
        grid_color="blue",
        turns_idx=None,
        mark_start_end=True,
        mark_turns=True,
        turn_color="orange",
        turn_scale=1.6,
        A_list=None,
        b_list=None,
        show_decomposition=True,
        decomp_color="yellow",
        decomp_opacity=0.3,
        smooth_shading=True,
        proj_segments=None,
        plot_projected=True,
        proj_color="cyan",
        proj_tube_radius=None,
    ):
        surf = pv_surface_from_grid(vg.grid, origin=vg.info.origin)

        p = pv.Plotter()
        p.add_mesh(surf, color=grid_color, opacity=grid_opacity, show_edges=False)

        if path_idx is None and path_xyz is None:
            p.show_axes(); p.show(); return

        centers = None
        if path_xyz is not None:
            centers = np.asarray(path_xyz, float)
        elif path_idx is not None:
            path_idx = np.asarray(path_idx)
            if path_idx.shape[0] >= 2:
                centers = np.array([np.array(vg.info.to_coord(idx)) + 0.5
                                    for idx in path_idx], float)

        if centers is not None and centers.shape[0] >= 2:
            line = pv.Spline(centers, n_points=len(centers))
            if plotPath:
                p.add_mesh(line.tube(radius=tube_radius, n_sides=16), color="red")

            if mark_start_end:
                p.add_mesh(pv.Sphere(radius=tube_radius*turn_scale, center=centers[0]),  color="green")
                p.add_mesh(pv.Sphere(radius=tube_radius*turn_scale, center=centers[-1]), color="orange")
    

            if mark_turns and turns_idx is not None:
                T = np.asarray(turns_idx)
                if T.ndim == 2 and T.shape[1] == 3 and T.shape[0] > 0:
                    T_centers = np.array([np.array(vg.info.to_coord(idx)) + 0.5 for idx in T], float)
                    pts = pv.PolyData(T_centers)
                    glyph_geom = pv.Sphere(radius=tube_radius*turn_scale)
                    glyphs = pts.glyph(scale=False, geom=glyph_geom)
                    p.add_mesh(glyphs, color=turn_color)

            print("[viz] surf points/cells:", getattr(surf, "n_points", None), getattr(surf, "n_cells", None))
            print("[viz] surf bounds:", getattr(surf, "bounds", None))
            print("[viz] centers min/max:", centers.min(axis=0), centers.max(axis=0))
            if A_list is not None:
                print("[viz] segments:", len(A_list))


            if show_decomposition and A_list is not None and b_list is not None:
                try:
                    import cdd
                    from scipy.spatial import ConvexHull
                    have_cdd = True
                except Exception:
                    have_cdd = False

                if not have_cdd:
                    try:
                        from scipy.spatial import HalfspaceIntersection, ConvexHull
                        have_hi = True
                    except Exception:
                        have_hi = False

                nseg = min(len(A_list), len(b_list), len(centers) - 1)
                for i in range(nseg):
                    A = np.asarray(A_list[i], dtype=float)
                    b = np.asarray(b_list[i], dtype=float).reshape(-1)

                    if A.ndim != 2 or A.shape[1] != 3 or b.ndim != 1 or b.size != A.shape[0]:
                        continue

                    active = ~np.all(np.isclose(A, 0.0, atol=1e-12), axis=1)
                    A = A[active]
                    b = b[active]
                    if A.shape[0] < 4:
                        continue

                    if have_cdd:
                        try:
                            mat = cdd.Matrix(np.hstack([b[:, None], -A]), number_type='float')
                            mat.rep_type = cdd.RepType.INEQUALITY
                            poly = cdd.Polyhedron(mat)
                            gens = poly.get_generators()
                            G = np.array(gens, dtype=float)
                            if G.ndim != 2 or G.shape[1] < 4:
                                continue
                            is_point = np.isclose(G[:, 0], 1.0)
                            verts = G[is_point, 1:4]
                            if verts.shape[0] < 4:
                                continue

                            hull = ConvexHull(verts)
                            faces = []
                            for tri in hull.simplices:
                                faces.extend([3, int(tri[0]), int(tri[1]), int(tri[2])])

                            mesh = pv.PolyData(verts, faces)
                            p.add_mesh(
                                mesh,
                                color=decomp_color,
                                opacity=decomp_opacity,
                                smooth_shading=smooth_shading,
                            )

                        except Exception as e:
                            print(e)
                            continue

                    elif have_hi:
                        hs = np.hstack([A, (-b)[:, None]])
                        pt_inside = 0.5 * (centers[i] + centers[i + 1])
                        try:
                            hs_int = HalfspaceIntersection(hs, interior_point=pt_inside)
                            verts = np.asarray(hs_int.intersections)
                            if verts.shape[0] >= 4:
                                hull = ConvexHull(verts)
                                faces = []
                                for tri in hull.simplices:
                                    faces.extend([3, *tri.tolist()])
                                mesh = pv.PolyData(verts, faces)
                                p.add_mesh(mesh, color=decomp_color, opacity=decomp_opacity,
                                        smooth_shading=smooth_shading)
                        except Exception:
                            continue
                    else:
                        pass

        p.show_axes()
        p.show()
