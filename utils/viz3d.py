"""
Functions for simulating in 3d, at paraview.
"""

import numpy as np

def generate_spheroid_mesh(axes, center, R, nU=20, nV=40):
    """
    Create a triangular mesh for a spheroid with semi-axes (a,a,c).
    center: 3D coords of center
    R:      3x3 rotation matrix
    nU, nV: resolution in polar/azimuth directions
    Returns (points, triangles).
    """
    a,b,c = axes
    center = np.asarray(center, dtype=float)
    R = np.asarray(R, dtype=float)
    
    # Param range
    u_vals = np.linspace(0, np.pi, nU+1)
    v_vals = np.linspace(0, 2*np.pi, nV, endpoint=False)

    # Store local coords: shape (nU+1, nV, 3)
    points_local = np.zeros((nU+1, nV, 3), dtype=float)
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            x_loc = a * np.sin(u) * np.cos(v)
            y_loc = b * np.sin(u) * np.sin(v)
            z_loc = c * np.cos(u)
            points_local[i, j] = [x_loc, y_loc, z_loc]

    # Flatten into ( (nU+1)*nV, 3 )
    points_local_flat = points_local.reshape(-1, 3)
    # Rotate + translate => world
    points_world_flat = (points_local_flat @ R.T) + center

    # Triangulate
    # We'll produce 2 triangles per quad cell in the u,v grid
    triangles = []
    for i in range(nU):
        for j in range(nV):
            i0 = i
            j0 = j
            i1 = i + 1
            j1 = (j + 1) % nV

            idx00 = i0*nV + j0
            idx10 = i1*nV + j0
            idx01 = i0*nV + j1
            idx11 = i1*nV + j1

            # two triangles
            triangles.append([idx00, idx10, idx11])
            triangles.append([idx00, idx11, idx01])

    triangles = np.array(triangles, dtype=np.int32)
    return points_world_flat, triangles


# -----------------------------
# Export to VTP (modern ParaView)
# -----------------------------
# TODO: Optimize this whole vtk export process
def export_vtp_polydata(filename, points, triangles):
    """
    Write a triangular surface mesh to a .vtp XML file
    that ParaView can open.
    """
    N = points.shape[0]
    M = triangles.shape[0]

    # Build connectivity (just all triangle indices in sequence)
    connectivity_str = []
    offsets_str = []
    offset_acc = 0
    for tri in triangles:
        i0, i1, i2 = tri
        connectivity_str.append(f"{i0} {i1} {i2}")
        offset_acc += 3
        offsets_str.append(str(offset_acc))

    connectivity_data = "\n".join(connectivity_str)
    offsets_data = "\n".join(offsets_str)

    # Points ASCII
    points_str_list = [f"{p[0]} {p[1]} {p[2]}" for p in points]
    points_data = "\n".join(points_str_list)

    xml_content = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{N}" NumberOfPolys="{M}">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
{points_data}
        </DataArray>
      </Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="ascii">
{connectivity_data}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
{offsets_data}
        </DataArray>
      </Polys>
    </Piece>
  </PolyData>
</VTKFile>
'''
    with open(filename, 'w') as f:
        f.write(xml_content)


def save_points_legacy_vtk(filename, boundary_points, source_points):
    """
    Write boundary & source points into a Legacy VTK (ASCII) PolyData file
    with a 'type' scalar to distinguish them (0=boundary, 1=source).
    
    Parameters
    ----------
    filename : str
        Path to the output .vtk file.
    boundary_points : ndarray of shape (Nb, 3)
    source_points   : ndarray of shape (Ns, 3)
    """
    # Combine boundary+source into one array
    all_points = np.vstack([boundary_points, source_points])
    
    # Create type array: 0 for boundary, 1 for source
    boundary_type = np.zeros(len(boundary_points), dtype=int)
    source_type   = np.ones(len(source_points),   dtype=int)
    all_types     = np.concatenate([boundary_type, source_type])
    
    # Number of total points
    n_all = all_points.shape[0]
    
    # Write the Legacy VTK file
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Ellipsoid Points\n")  # Title/description
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write the points
        f.write(f"POINTS {n_all} float\n")
        for i in range(n_all):
            x, y, z = all_points[i]
            f.write(f"{x} {y} {z}\n")
        
        # Now the point data
        f.write(f"\nPOINT_DATA {n_all}\n")
        f.write("SCALARS type int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in all_types:
            f.write(f"{val}\n")



def save_multiple_ellipsoids_legacy_vtk(filename, list_of_boundary_arrays, list_of_source_arrays):
    """
    Write multiple ellipsoids (boundary & source points) into a single Legacy VTK (ASCII) PolyData file
    with a 'type' scalar to distinguish them (0=boundary, 1=source).
    
    Parameters
    ----------
    filename : str
        Path to the output .vtk file.
    list_of_boundary_arrays : list of ndarrays
        Each element is an (Nb_i, 3) array of boundary points for the i-th ellipsoid.
    list_of_source_arrays : list of ndarrays
        Each element is an (Ns_i, 3) array of source points for the i-th ellipsoid.
        The i-th source array corresponds to the i-th boundary array.
    """
    # Combine all boundary + source points and track their types
    all_points = []
    all_types = []
    
    for boundary_points, source_points in zip(list_of_boundary_arrays, list_of_source_arrays):
        # 0 for boundary, 1 for source
        boundary_type = np.zeros(len(boundary_points), dtype=int)
        source_type   = np.ones(len(source_points),   dtype=int)
        
        # Stack them
        all_points.append(boundary_points)
        all_points.append(source_points)
        
        all_types.append(boundary_type)
        all_types.append(source_type)
    
    # Convert to single arrays
    all_points = np.vstack(all_points)
    all_types  = np.concatenate(all_types)
    
    # Number of total points
    n_all = all_points.shape[0]
    
    # Write the Legacy VTK file
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Multiple Ellipsoids\n")  # Title/description
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write the points
        f.write(f"POINTS {n_all} float\n")
        for i in range(n_all):
            x, y, z = all_points[i]
            f.write(f"{x} {y} {z}\n")
        
        # Now the point data
        f.write(f"\nPOINT_DATA {n_all}\n")
        f.write("SCALARS type int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in all_types:
            f.write(f"{val}\n")



