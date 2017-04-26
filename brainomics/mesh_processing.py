import numpy as  np
from nibabel.gifti import read, write, GiftiDataArray, GiftiImage
#import nipy.algorithms.graph.graph as fg


#############################################################
#  ancillary stuff
#############################################################

def vectp(a,b):
    """
    vect product of two vectors in 3D
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     - a[0] * b[2] + a[2] * b[0],
                     a[0] * b[1] - a[1] * b[0]])

def area (a,b):
    """
    area spanned by the vectors(a,b) in 3D
    """
    c = vectp(a, b)
    return np.sqrt((c ** 2).sum())

def cartesian_to_spherical(cartesian):
    """
    """
    x, y, z = cartesian.T
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos( z / r )
    phi = np.arctan( x/y)
    spherical = np.vstack((r, theta, phi)).T
    return spherical

def spherical_to_cartesian(spherical):
    """
    """
    r, theta, phi = spherical.T
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.vstack((x, y, z)).T

###############################################################
# mesh io
###############################################################


def mesh_arrays(mesh, nibabel=True):
    """ Returns the arrays associated with a mesh
    if len(output)==2, the arrays are coordiantes and triangle lists
    if len(output)==3, the arrays are  coordiantes, triangle lists
    and outer normal
    fixme: use intent !
    """
    if isinstance(mesh, str):
        mesh_ = read(mesh)
    else:
        mesh_ = mesh
    cor = mesh_.getArraysFromIntent("NIFTI_INTENT_POINTSET")[0].data
    tri = mesh_.getArraysFromIntent("NIFTI_INTENT_TRIANGLE")[0].data
    return cor, tri


def mesh_from_arrays(coord, triangles, path=None):
    """ Create a mesh object from two arrays

    fixme:  intent should be set !
    """
    carray = GiftiDataArray().from_array(coord.astype(np.float32),
        "NIFTI_INTENT_POINTSET",
        encoding='B64BIN')
        #endian="LittleEndian")
    tarray = GiftiDataArray().from_array(triangles.astype(np.int32),
        "NIFTI_INTENT_TRIANGLE",
        encoding='B64BIN')
        #endian="LittleEndian")
    img = GiftiImage(darrays=[carray, tarray])
    if path is not None:
        try:
            from soma import aims
            mesh = aims.AimsTimeSurface(3)
            mesh.vertex().assign([aims.Point3df(x)for x in coord])
            mesh.polygon().assign([aims.AimsVector_U32_3(x)for x in triangles])
            aims.write(mesh, path)
        except:
            print("soma writing failed")
            write(img, path)
    return img

#def mesh_from_arrays(coord, triangles, path=None):
#    """ Create a mesh object from two arrays
#
#    fixme:  intent should be set !
#    """
#    carray = GiftiDataArray().from_array(coord.astype(np.float32),
#                                         "NIFTI_INTENT_POINTSET")
#    tarray = GiftiDataArray().from_array(triangles, "NIFTI_INTENT_TRIANGLE")
#    img = GiftiImage(darrays=[carray, tarray])
#    if path is not None:
#        write(img, path)
#    return img

def load_texture(path):
    """Return an array from texture data stored in a gifti file

Parameters
----------
path string or list of strings
path of the texture files

Returns
-------
data array of shape (nnode) or (nnode, len(path))
the corresponding data
"""
    from nibabel.gifti import read

    # use alternative libraries than nibabel if necessary
    if hasattr(path, '__iter__'):
        tex_data = []
        for f in path:
            ftex = read(f).getArraysFromIntent('NIFTI_INTENT_TIME_SERIES')
            tex = np.array([f.data for f in ftex])
            tex_data.append(tex)
        tex_data = np.array(tex_data)
        if len(tex_data.shape) > 2:
            tex_data = np.squeeze(tex_data)
    else:
        tex_ = read(path)
        tex_data = np.array([darray.data for darray in tex_.darrays])
    return tex_data


#def save_texture(path, data, intent='NIFTI_INTENT_NONE', verbose=False):
#    """
#volume saving utility for textures
#Parameters
#----------
#path, string, output image path
#data, array of shape (nnode)
#data to be put in the volume
#intent: string, optional
#intent
#
#Fixme
#-----
#Missing checks
#Handle the case where data is multi-dimensional ?
#"""
#    from nibabel.gifti import write, GiftiDataArray, GiftiImage
#    if verbose:
#        print 'Warning: assuming a float32 gifti file'
#    darray = GiftiDataArray().from_array(data.astype(np.float32), intent)
#    img = GiftiImage(darrays=[darray])
#    write(img, path)

def save_texture(filename, data):
    from soma import aims
    tex = aims.TimeTexture('FLOAT')
    tex[0].assign(data)
    aims.write(tex, filename)


def spherical_coordinates(mesh):
    """ Return the spherical coordinates of the nodes of a certain mesh
    """
    cartesian, _ = mesh_arrays(mesh)
    return cartesian_to_spherical(cartesian)


def mesh_to_graph(mesh_path, nibabel=True):
    """
    Builds a graph from a patha to a mesh
    """
    cor, tri = mesh_arrays(mesh_path, nibabel)
    return poly_to_graph(cor.shape[0], tri, cor)


def poly_to_graph(nnodes, poly, coord=None):
    """ convert a set of polygones into a graph

    Parameters
    ----------
    nnodes: int
    poly: list of triangles
    """
    E = poly.shape[0]
    edges = np.zeros((3 * E, 2), np.int)
    weights = np.ones(3 * E)

    for i in range(E):
        sa = poly[i][0]
        sb = poly[i][1]
        sc = poly[i][2]
        edges[3 * i] = np.array([sa, sb])
        edges[3 * i + 1] = np.array([sa, sc])
        edges[3 * i + 2] = np.array([sb, sc])

    G = fg.WeightedGraph(nnodes, edges, weights)

    # symmeterize the graph
    G.symmeterize()

    # remove redundant edges
    G.cut_redundancies()

    if coord is not None:
        # make it a metric graph
        G.set_euclidian(coord)

    return G


def cut_mesh(mesh_path, vertex_mask, output_path=None):
    """
    create a reduced medh by keeping only the vertices inside vertex mask
    """
    vertex_mask = vertex_mask.ravel()
    coord, triangles = mesh_arrays(mesh_path)

    if vertex_mask.size != coord.shape[0]:
        raise ValueError('incorrect mask provided')

    reduced_coord = coord[vertex_mask]
    reduced_triangles = []
    for t in triangles:
        if np.prod(vertex_mask[np.array(t)]) != 0:
            reduced_triangles.append(t)

    mapping = np.cumsum(vertex_mask) - 1
    reduced_triangles = mapping[np.array(reduced_triangles)]

    return mesh_from_arrays(reduced_coord, reduced_triangles, output_path)


def node_area(mesh):
    """
    returns a vector of are values, one for each mesh,
    which is the averge area of the triangles around it
    """
    vertices, poly = mesh_arrays(mesh)
    E = poly.shape[0]

    narea = np.zeros(len(vertices))
    for i in range(E):
        sa, sb, sc = poly[i]
        a = vertices[sa] - vertices[sc]
        b = vertices[sb] - vertices[sc]
        ar = area(a, b)
        narea[sa] += ar
        narea[sb] += ar
        narea[sc] += ar

    narea /= 6
    # because division by 2 has been 'forgotten' in area computation
    # the area of a triangle is divided into the 3 vertices
    return narea


def mesh_area(mesh):
    """
    This function computes the input mesh area
    """
    vertices, poly = mesh_arrays(mesh)
    marea = 0
    E = poly.size

    for i in range(E):
        sa = poly[i][0]
        sb = poly[i][1]
        sc = poly[i][2]
        a = vertices[sa] - vertices[sc]
        b = vertices[sb] - vertices[sc]
        marea += area(a, b)
    return marea / 2


def mesh_integrate(mesh, tex, coord=None):
    """
    Compute the integral of the texture on the mesh
    - coord is an additional set of coordinates to define the vertex position
    by default, mesh.vertex() is used
     """
    if coord is None:
        vertices = np.array(mesh.vertex())
    else:
        vertices = coord
    poly = mesh.polygon()
    integral = 0
    coord = np.zeros((3, 3))
    E = poly.size()
    data = np.array(tex.data())

    for i in range(E):
        sa = poly[i][0]
        sb = poly[i][1]
        sc = poly[i][2]
        a = vertices[sa] - vertices[sc]
        b = vertices[sb] - vertices[sc]
        mval = (data[sa] + data[sb] + data[sc]) / 3
        integral += mval * area(a, b) / 2

    return integral

#def flatten(mesh):
#    """
#    This function returns a 2-dimensional coordinate system
#    that represents the mesh points
#    """
#    from parietal.python.eda.dimension_reduction import isomap_dev
#    G = mesh_to_graph(mesh)
#    chart = isomap_dev(G, dim=2, p=300, verbose = 0)
#    return chart


def isomap_patch(mesh, mask, show=False):
    """return low-dimensional coordinates for the the patch

    Parameters
    ==========
    mesh: string or nibabel mesh,
          the input mesh to be cherted
    mask: string or array of shape (n_vertices)
          a mask for the region of interest on the mesh
    show: boolean, optional,
          if yes, make an image of the coordinates
    """
    from sklearn.manifold.isomap import Isomap
    # Read the data
    coord, tri = mesh_arrays(mesh)
    if isinstance(mask, basestring):
        mask = read(mask).darrays[0].data > 0
    else:
        mask = mask.astype(np.bool)
    coord, tri = coord[mask], tri[mask[tri].all(1)]
    tri = np.hstack((0, np.cumsum(mask)[:-1]))[tri]

    # perform the dimension reduction
    xy = Isomap().fit_transform(coord)

    # try to make the sign invariant and repect the orientations
    xy *= np.sign((xy ** 3).sum(0))
    a, b, c = tri[0]
    xy[:, 1] *= np.sign(np.linalg.det(np.vstack((xy[b] - xy[a], xy[c] - xy[a]))))

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.tripcolor(xy.T[0], xy.T[1], tri, (xy ** 2).sum(1),
                        shading='faceted')
        plt.show()
    return xy


#def smooth_texture(mesh, input_texture, output_texture=None, sigma=1,
#                   lsigma=1., mask=None):
#   """ Smooth a texture along some mesh
#
#   parameters
#   ----------
#   mesh: string,
#         path to gii mesh
#   input_texture: string,
#                  texture path
#   ouput_texture: string,
#                  smooth texture path
#   sigma: float,
#          desired amount of smoothing
#   lsigma: float,
#           approximate smoothing in one iteration
#   mask: string,
#         path of a mask texture
#   """
#   import nipy.algorithms.graph.field as ff
#
#   G = mesh_to_graph(mesh)
#   if mask is not None:
#       mask = read(mask).darrays[0].data > 0
#       G = G.subgraph(mask)
#   add_edges = np.vstack((np.arange(G.V), np.arange(G.V))).T
#   edges = np.vstack((G.edges, add_edges))
#   weights = np.concatenate((G.weights, np.zeros(G.V)))
#   weights = np.maximum(np.exp(- weights ** 2 / ( 2 * lsigma ** 2)), 1.e-15)
#
#   f = ff.Field(G.V, edges, weights)
#   # need to re-order the edges
#   order = np.argsort(f.edges.T[0] * f.V + f.edges.T[1])
#   f.edges, f.weights = f.edges[order], f.weights[order]
#   f.normalize(0)
#   niter = (sigma * 1. / lsigma) ** 2
#
#   if input_texture[-4:] == '.tex':
#       import tio as tio
#       data = tio.Texture("").read(input_texture).data
#   else:
#       data = read(input_texture).darrays[0].data
#   if mask is not None:
#       data = data[mask]
#   dtype = data.dtype
#   data[np.isnan(data)] = 0
#   f.set_field(data.T)
#   f.diffusion(niter)
#   data = f.get_field().astype(dtype)
#
#   if output_texture is not None:
#       if output_texture[-4:] == '.tex':
#           import tio as tio
#           tio.Texture("", data=data.T).write(output_texture)
#           print 'tex'
#       else:
#           intent = 0
#           wdata = data
#           if mask is not None:
#               wdata = mask.astype(np.float)
#               wdata[mask > 0] = data
#           darray = GiftiDataArray().from_array(wdata.astype(np.float32),
#                                                intent)
#           img = GiftiImage(darrays=[darray])
#           write(img, output_texture)
#   return data


def compute_normal_vertex(mesh, mask=None):
    """ Compute the normal vector at each vertex of the mesh

    Parameters
    ---------
    mesh: string, a path to a mesh file

    Returns
    -------
    normals: array of shape (mesh.n_vertices, 3)
             the normal vector at each mesh node
     mask: string, optional, a path to mask texture for that mesh/subject,
           so that only part of the vertices are conssidered.
    fixme
    -----
    Put in mesh_processing
    """
    # get coordinates and triangles
    coord, triangles = mesh_arrays(mesh)
    if mask is None:
        mask = np.ones(coord.shape[0]).astype(np.bool)
    else:
        mask = read(mask).darrays[0].data.astype(np.bool)

    # compute the normal for each triangle
    norms = np.zeros((mask.size, 3))
    for triangle in triangles:
        if mask[triangle].any():
            sa, sb, sc = triangle
            a = coord[sb] - coord[sa]
            b = coord[sc] - coord[sa]
            norm = vectp(a, b)
            norms[sa] += norm
            norms[sb] += norm
            norms[sc] += norm

    # normalize the normal at each vertex
    eps = 1.e-15
    norms = (norms.T / np.sqrt(eps + np.sum(norms ** 2, 1))).T
    return norms[mask]


def texture_gradient(mesh, texture, mask=None):
    """ Compute the gradient of a given texture at each point of the mesh

    Parameters
    ---------
    mesh: string, a path to a mesh file
    texture: string, a path to a texture file
    mask: string, optional, a path to mask texture for that mesh/subject,
          so that only part of the vertices are conssidered.
    Returns
    -------
    gradient: array of shape (mesh.n_vertices, 3)
    the gradient vector at each mesh node.

    fixme
    -----
    Put in mesh_processing

    Note
    ----
    the gradient is expressedn in 3D space, note on surface coordinates
    """
    # get coordinates and triangles
    coord, triangles = mesh_arrays(mesh)
    if mask is None:
        mask = np.ones(coord.shape[0]).astype(np.bool)
    else:
        mask = read(mask).darrays[0].data.astype(np.bool)

    # compute the neighborhood system
    neighb = mesh_to_graph(mesh).to_coo_matrix().tolil().rows

    # read the texture
    y = read(texture).darrays[0].data

    # compute the gradient
    gradient = []
    for i in np.where(mask)[0]:
        yi = y[neighb[i]]
        yi[mask[neighb[i]] is False] = y[i]
        grad = np.linalg.lstsq(coord[neighb[i]], yi)[0]
        gradient.append(grad)
    return np.array(gradient)


#####################################################################
# mesh resampling
#####################################################################

def remesh(input_path, src, target, output_path=None):
    """ Main function to project a given mesh onto a template

    Parameters
    ==========
    input_path: string, path of the mesh to be resampled
    src: string, path of the equivalent mesh, with coordinates in target space
    target: string, path of the target mesh
    output_path: string, optional,
                 if not None, path where to write the output to

    Returns
    =======
    output_mesh: the output mesh object
    """
    isin = remesh_step1(src, target)
    input_coord, input_triangles = mesh_arrays(input_path)
    ref_coord, ref_triangles = mesh_arrays(target)

    output_coord = ref_coord.copy()
    for v in range(len(ref_coord)):
        origv = input_coord[input_triangles[isin[v][0]]]
        lam, gam = isin[v][1:3]
        output_coord[v] = lam * origv[1] + gam * origv[2] +\
                             (1. - lam - gam) * origv[0]

    output_mesh = mesh_from_arrays(output_coord, ref_triangles, output_path)
    return output_mesh


def remesh_step1(src, target):
    """Computation of mesh resampling parameters

    Parameters
    ==========
    src: mesh to be resampled or path such a mesh
    target: reference mesh or path such a mesh

    Returns
    =======
    is_in: array of shape (W, 3)
           each row yields i) the triangle in [1..E] in which node w is
           ii) the barycentric coordinates of the node inside the triangle

    """
    svertex, _ = mesh_arrays(target)
    uvertex, ufaces = mesh_arrays(src)
    return _remesh(uvertex, ufaces, svertex)


def _remesh(uvertex, ufaces, svertex):
    """Computation of mesh resampling parameters - low level function

    Parameters
    ==========
    uvertex: array of shape (V, 3),
             position of the nodes of the src mesh
    ufaces: array of shape (E, 3),
            triangles of the src mesh
    svertex: array of shape (W, 3),
             position of the nodes of the target mesh

    Returns
    =======
    is_in: array of shape (W, 3)
           each row yields i) the triangle in [1..E] in which node w is
           ii) the barycentric coordinates of the node inside the triangle

    NOTE
    ====
    Work in progress -- not to be considered as reliable code
    """
    # roughly same center and same bounding box
    if (not np.allclose(svertex.mean(0), uvertex.mean(0), atol=1e-2)) or \
           (not np.allclose(svertex.max(0), uvertex.max(0), atol=1e-2)):
        print("\n WARNING : different mean or max(0) \n")
    polycenters = uvertex[ufaces].mean(1)
    maxs = np.sum((uvertex[ufaces] - polycenters[:, np.newaxis]) ** 2, 2).max(1)

    # precompute some stuffs
    dmax = np.sqrt(maxs.max())
    smin, smax = svertex.min(0) - 2 * dmax, svertex.max(0) + 2 * dmax
    grid_xyz = [np.r_[smin[i]:smax[i]:dmax] for i in (0, 1, 2)]
    S = [np.searchsorted(g, s) for g, s in zip((grid_xyz), svertex.T)]
    d = [[set() for _ in grid_xyz[0]] for x in (0, 1, 2)] # parcel list
    for dxyz, Sxyz in zip(d, S):
        for v, n in enumerate(Sxyz):
            dxyz[n - 1].add(v)
            dxyz[n].add(v)
            dxyz[n + 1].add(v)
    # the isinN and isin lists will eventually, for each of the structured mesh
    # vertex, store the associated unstructured triangle index,
    #and the 3 weights
    isinN, isin = [-1] * len(svertex), [(-1, -1)] * len(svertex)

    # will iterate on the unstructured-mesh triangles (their centers, actually)
    triangle = np.empty((3, 3))
    a, bc = triangle[0], triangle[1:] # setup some views
    for i, p in enumerate(polycenters):
        if i % 30000 == 0:
            print(i, "/", len(polycenters))
        # find closest points of p : first a rough filter, then an exact filter
        binxyz = [np.searchsorted(grid_xyz[x], p[x]) for x in (0, 1, 2)]
        W = d[0][binxyz[0]].intersection(d[1][binxyz[1]]).intersection(
            d[2][binxyz[2]])
        W = np.fromiter(W, int, len(W))
        W = W[np.sum((svertex[W] - p) ** 2  ,1) <= maxs[i]]
        if len(W) == 0:
            continue
        triangle[:] = uvertex[ufaces[i]]
        bc -= a
        proj = np.inner(bc, bc)
        # (precompute part of the weight computation for the current triangle)
        m00, m01, m11 = float(proj[0,0]), float(proj[0,1]), float(proj[1,1])
        detA = m00 * m11 - m01 * m01
        m00 /= detA
        m01 /= detA
        m11 /= detA
        # For each (triangle-edges projected) candidate vertex
        # of the structured mesh,
        # computes the 3 weights, and store the results if it belongs
        # to the triangle.
        for v0v1, n in zip(np.dot(svertex[W] - a, bc.T), W):
            v0, v1 = float(v0v1[0]), float(v0v1[1])
            lam, gam = m11 * v0 - m01 * v1, - m01 * v0 + m00 * v1
            if (lam >= -0.001) & (gam >= - 0.001) & (lam + gam <= 1.001):
                isinN[n] = i
                isin[n] = lam, gam
            elif isinN[n] < 0: # debug
                isinN[n] -= 1
                isin[n] = lam, gam
    return np.vstack((np.array(isinN), np.array(isin).T)).T



#####################################################################
# Deprecated stuff
#####################################################################


def write_aims_Mesh(vertex, polygon, fileName):
    """
    Given a set of vertices, polygons and a filename,
    write the corresponding aims mesh
    the aims mesh is returned

    Caveat
    ======
    depends on aims
    """
    from soma import aims
    vv = aims.vector_POINT3DF()
    vp = aims.vector_AimsVector_U32_3()
    for x in vertex: vv.append(x)
    for x in polygon: vp.append(x)
    m = aims.AimsTimeSurface_3()
    m.vertex().assign( vv )
    m.polygon().assign( vp )
    m.updateNormals()
    W = aims.Writer()
    W.write(m, fileName)
    return m


def brainvisa_to_gifti(bv_mesh, gii_mesh):
    """ Conversion of brainvisa mesh to a gifti mesh

    Caveat
    ======
    requires aims
    """
    from gifti import GiftiEncoding, GiftiImage_fromarray, GiftiIntentCode
    from soma import aims
    mesh = aims.Reader().read(bv_mesh)
    vertices = np.array(mesh.vertex())
    poly  = np.array([np.array(p) for  p in mesh.polygon()])

    def buildMeshImage(vertices, faces,
                       encoding=GiftiEncoding.GIFTI_ENCODING_ASCII):
        """ build a GiftiImage from arrays of vertices and faces

        NOTE : this is doing the same as the gifti.GiftiImage_fromTriangles
        function and is only redefined here for demonstration purpose"""

        k = GiftiImage_fromarray(vertices)
        k.arrays[0].intentString = "NIFTI_INTENT_POINTSET"
        k.addDataArray_fromarray(faces, GiftiIntentCode.NIFTI_INTENT_TRIANGLE)
        for a in k.arrays:
            a.encoding = encoding
        return k
    buildMeshImage(vertices, poly).save(gii_mesh)


