import numpy as np
from skimage import measure

def save_off(filename, VPos, ITris):
    """
    Save OFF file given vertex and triangle buffers
    Parameters
    ----------
    filename: string
        Path to which to save file
    VPos: ndarray(M, 3)
        Coordinates of the M vertices
    ITris: ndarray(N, 3)
        Vertex indices of the N triangles
    """
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    if nV == 0:
        print("ERROR: The mesh you're trying to save has zero vertices, so the volume is either all negative or all positive")
        return
    fout = open(filename, "w")
    fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g\n"%tuple(VPos[i, :]))
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()

def get_ellipsoid_gaussian(x, y, z, c, v1, sigma1, v2, sigma2, v3, sigma3):
    """
    Given a grid of points, create a multivariate Gaussian function
    with ellipsoidal cross sections
    Parameters
    ----------
    x: ndarray(M, N, W)
        A grid of x locations
    y: ndarray(M, N, W)
        A grid of y locations
    z: ndarray(M, N, W)
        A grid of z locations
    c: list(x, y, z)
        Location of the center of the Gaussian
    v1: list(x, y, z)
        Direction of the first principal axis
    sigma1: float
        Standard deviation along the first principal axis
    v2: list(x, y, z)
        Direction of the second principal axis
    sigma2: float
        Standard deviation along the second principal axis
    v3: list(x, y, z)
        Direction of the second principal axis
    sigma3: float
        Standard deviation along the third principal axis
    Returns
    -------
    f: ndarray(M, N, W)
        The corresponding scalar function of the Gaussian evaluated
        at all points on the grid
    """
    c = np.array(c, dtype=float).flatten()
    v1 = np.array(v1, dtype=float).flatten()
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = np.array(v2, dtype=float).flatten()
    v2 /= np.sqrt(np.sum(v2**2))
    v3 = np.array(v3, dtype=float).flatten()
    v3 /= np.sqrt(np.sum(v3**2))
    X = np.array([x, y, z]).T
    print(X.shape)
    X -= c[None, :]
    d1 = (X.dot(v1))**2/(sigma1**2)
    d2 = (X.dot(v2))**2/(sigma2**2)
    d3 = (X.dot(v3))**2/(sigma3**2)
    return np.exp(-d1)*np.exp(-d2)*np.exp(-d3)

def get_sphere(N):
    """
    Parameters
    ----------
    N: int
        How many coordinates to sample along each axis
    """
    pix = np.linspace(-1, 1, N)
    x, y, z = np.meshgrid(pix, pix, pix)
    sigma = 0.2
    u = np.exp(-((x-0.2)**2 + (y-0.2)**2 + (z-0.2)**2)/(sigma*sigma))
    VPos, ITris, _, _ = measure.marching_cubes(u, 0.2, gradient_direction="ascent")
    save_off("sphere.off", VPos, ITris)

def get_bowling_pin(N):
    """
    Parameters
    ----------
    N: int
        How many coordinates to sample along each axis
    """
    pix = np.linspace(-1, 1, N)
    x, y, z = np.meshgrid(pix, pix, pix)
    sigma1 = 0.2
    u = np.exp(-((x-0.2)**2 + (y-0.2)**2 + (z-0.2)**2)/(sigma1*sigma1))
    sigma2 = 0.1
    u += np.exp(-(x**2 + y**2 + z**2)/(sigma2*sigma2))
    VPos, ITris, _, _ = measure.marching_cubes(u, 0.2, gradient_direction="ascent")
    save_off("bowlingpin.off", VPos, ITris)

def get_sphere_with_divet(N):
    """
    Parameters
    ----------
    N: int
        How many coordinates to sample along each axis
    """
    pix = np.linspace(-1, 1, N)
    x, y, z = np.meshgrid(pix, pix, pix)
    sigma1 = 0.2
    u = np.exp(-(x**2 + y**2 + z**2)/(sigma1*sigma1))
    sigma2 = 0.1
    u -= np.exp(-((x-0.2)**2 + (y-0.2)**2 + (z-0.2)**2)/(sigma2*sigma2))
    VPos, ITris, _, _ = measure.marching_cubes(u, 0.01, gradient_direction="ascent")
    save_off("spherecut.off", VPos, ITris)

def get_my_surface(N):
    """
    Parameters
    ----------
    N: int
        How many coordinates to sample along each axis
    """
    pix = np.linspace(0, 1, 100)
    x, y, z = np.meshgrid(pix, pix, pix)
    ## TODO: Fill this in
    u = 0*x ## TODO: This is a dummy value
    
    VPos, ITris, _, _ = measure.marching_cubes(u, 0.1, gradient_direction="ascent")
    save_off("mysurface.off", VPos, ITris)


if __name__ == '__main__':
    get_sphere(100)
    get_bowling_pin(100)
    get_sphere_with_divet(100)
    get_my_surface(100)
