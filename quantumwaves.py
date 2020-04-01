from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import Vector
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.exporters
import numpy as np
import sys, os

from PIL import Image
from pathlib import Path
from math import sqrt

import matplotlib.pyplot as plt

from numba import njit, prange
from schrodinger import schrodinger

from time import time

import numpy as np

from scipy import interpolate

do_parallel = False
do_collapse = False
sim_size = 125

index = 0

sec = 30
fps = 30

collapse_interval = (fps*sec)/2#/3 #2 * fps

record = False
#record = True

#do_smoothing = False
do_smoothing = True

layer = 'imag'
layer = 'real'
layer = 'surf'
layer = ''

if len(sys.argv) >= 2:
    layer = sys.argv[1]

name = 'decay_movement9_follow'
name = name + ('_' + layer) if layer != '' else name

folder = Path('C:/frames') / name

res = np.array((1920, 1080))
#res = np.array((3840, 2160))
res = res if record else res // 1.25

sim = schrodinger.Simulate(sim_size, collapse=do_collapse)
frames = fps*sec

schrodinger.util.do_parallel = do_parallel

if record:
    do_smoothing = True

pg.setConfigOptions(antialias=True)

## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle(f'{os.path.basename(__file__)} ... {sim_size}x{sim_size} ... {f"RENDERING {name}" if record else "PREVIEW"}')

# WF COLLAPSE
#w.setCameraPosition(distance=60)

# WF COLLAPSE 2
#w.setCameraPosition(distance=80)

# WF COLLAPSE 3
#w.setCameraPosition(distance=133)

# WF COLLAPSE 4
#w.setCameraPosition(distance=100)

# WF COLLISION
#w.setCameraPosition(distance=100)

# WF ENTANGLEMENT
#w.setCameraPosition(distance=200)

# WF FOLLOW MOVEMENT
w.setCameraPosition(distance=80)

w.resize(res[0],res[1])
w.setFixedSize(res[0],res[1])

 # move window in OS
# this is overkill, I know
w.move( app.desktop().screenGeometry().width()  / 2 - res[0] / 2,
        app.desktop().screenGeometry().height() / 2 - res[1] / 2)

@njit(cache=True)
def cubic_interp1d(x0, x, y):
    """
    Interpolate a 1-D function using cubic splines.
      x0 : a float or an 1d-array
      x : (N,) array_like
          A 1-D array of real/complex values.
      y : (N,) array_like
          A 1-D array of real values. The length of y along the
          interpolation axis must be equal to the length of x.

    Implement a trick to generate at first step the cholesky matrice L of
    the tridiagonal matrice A (thus L is a bidiagonal matrice that
    can be solved in two distinct loops).

    additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
    """
    #x = np.asfarray(x)
    #y = np.asfarray(y)
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    # remove non finite values
    # indexes = np.isfinite(x)
    # x = x[indexes]
    # y = y[indexes]

    # check if sorted
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # allocate buffer matrices
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 # natural boundary
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # solve [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # find index
    index = np.searchsorted(x, x0)
    #np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0

@njit(cache=True)
def make_gridlines(X, Y, axis=0, stride=4, extend=3):

    # AXIS IS ONLY USED FOR PLACING POINTS, NOT SIZING, THUS MUST BE SQUARE
    if X != Y:
        raise(Exception('X must equal Y'))


    # X = maximum x value
    # Y = maximum y value

    P = 4 # points per line
    N = (2*X) // stride # number of lines
    M = N*P # total number of points

    points = np.zeros((M,3))
    colors = np.ones((M, 4))

    for n in prange(N):

        i0 = n*(P)
        i1 = n*(P) + n*(P) - 1

        a = int(axis)
        b = int(not axis)

        if n < N//2:
            # first half
            points[i0 : (i1+1), a] = 0 - n * stride
            fading = (1 - n/(N//2))
        else:
            # second half
            points[i0 : (i1+1), a] = 0 + n * stride
            fading = (1 - (n//2)/(N//2))

        points[i0,   b] = (- Y)
        points[i0+1, b] = 0
        points[i0+2, b] = (+ Y)
        points[i0+3, b] = (Y * 2)

        colors[i0,   3] = 0
        colors[i0+1, 3] = .25 * fading
        colors[i0+2, 3] = .25 * fading
        colors[i0+3, 3] = 0

        if n == 0 or n == (N-1):
            # avoid overlaying with previous wavelines
            colors[i0+1, 3] = 0
            colors[i0+2, 3] = 0

    return points, colors # (N, 3), (N, 4)

@njit(cache=True, parallel=do_parallel)
def make_wavelines(wavedata,
                        P = 5000, # points per spline at each line
                        axis=0,
                        stride=4,
                        smoothing = True,
                    ):


    X = wavedata.shape[0] # maximum x value
    Y = wavedata.shape[1] # maximum y value
    N = X // stride # number of lines

    if not smoothing:
        P = int(Y)

    M = N * (P + 2) # total number of points

    points = np.zeros((M,3))
    colors = np.ones((M,4))

    y0 = np.arange(0, X)

    if smoothing:
        # interpolation parameter for splines
        y = np.linspace(0, X-1, P)
    else:
        y = np.linspace(0, X-1, X)


    for x in prange(N):

        w = x * stride # index in the wave data for this row

        # indices in the destination arrays for this ith line's points
        i0 = x*(P+2) # start index
        i1 = x*(P+2) + (P+2) -1 # final index (inclusive)

        points[i0 : (i1+1), 0] = w  # all x values are i

        # start at horizon, alpha=0
        points[i0, 1] = - Y
        points[i0, 2] = 0
        colors[i0, 3] = 0

        # (ovs, 3)
        if axis == 0:
            z0 = wavedata[w, :]
        else:
            z0 = wavedata[:, w]

        if smoothing:
            # interpolate wave lines into splines for smoother lines
            z = cubic_interp1d(y, y0, z0)
        else:
            z = z0

        # put interpolated lines into destsination arrays
        points[i0 + 1: i1, 1] = y
        points[i0 + 1: i1, 2] = z
        colors[i0 + 1: i1, 3] = .33#points[i0 + 1: i1, 2] + .25

        # end at horizon, alpha=0
        points[i1, 1] = (2 * Y)
        points[i1, 2] = 0
        colors[i1, 3] = 0

    return points, colors # (N, 3), (N, 4)

def surf_smoothing(surf_data, smoothing=2):

    X = surf_data.shape[0]
    Y = surf_data.shape[1]

    x = np.arange(X)
    y = np.arange(Y)
    f = interpolate.interp2d(x, y, surf_data, kind='cubic')

    xnew = np.linspace(0, X, X*smoothing)
    ynew = np.linspace(0, Y, Y*smoothing)

    return f(xnew, ynew)

# background sphere
ds = 100
md = gl.MeshData.sphere(rows=ds, cols=ds)

sphere_colors = np.zeros((md.faceCount(), 4), dtype=float)
#colors[:,0] = np.linspace(1, 0, colors.shape[0])
#colors[:,1] = np.linspace(1, 0, colors.shape[0])
#colors[:,2] = np.linspace(1, 0.25, colors.shape[0])
#colors *= .2

sphere = gl.GLMeshItem(meshdata=md, smooth=True)
sphere.translate(5, -5, 0)
sphere.scale(1000,1000,1000)

## since this does not require normal vectors to render (thus we
## can set computeNormals=False to save time when the mesh updates)

# COLORMAP
cmap = plt.get_cmap('viridis')

# set bg to minimum of colormap for continuity
sphere_colors[:] = cmap(0)
md.setFaceColors(sphere_colors)

real = gl.GLLinePlotItem(antialias=True) # aa doesn't actually toggle here
imag = gl.GLLinePlotItem(antialias=True) # aa doesn't actually toggle here

rhzn = gl.GLLinePlotItem(antialias=True) # aa doesn't actually toggle here
ihzn = gl.GLLinePlotItem(antialias=True) # aa doesn't actually toggle here

real.setDepthValue(0)
imag.setDepthValue(0)

rhzn.setDepthValue(0)
ihzn.setDepthValue(0)

surf = gl.GLSurfacePlotItem(computeNormals=False, smooth=True)
surf.setGLOptions('translucent')
surf.setDepthValue(10)

d = sim.simulate_frame(debug=0)

X, Y = d.real.shape
rhzn_points, rhzn_colors = make_gridlines(X, Y, axis=0)
ihzn_points, ihzn_colors = make_gridlines(X, Y, axis=1)
rhzn.setData(pos=rhzn_points, color=rhzn_colors)
ihzn.setData(pos=ihzn_points, color=ihzn_colors)

rcol_bias = 0.7 #.8
icol_bias = 0.3 #.5

rhzn_colors[:,0:3:2] *= rcol_bias # bias more red
ihzn_colors[:,0] *= icol_bias # bias more blue

rescale = 200

if sim_size > 350:
    surf_smooth = 1
else:
    surf_smooth = 8 if do_smoothing else 1
zscale =  3

if layer == '':
    w.addItem(sphere)

for n, elem in enumerate([real, imag, rhzn, ihzn]):
    elem.scale(1/d.shape[0], 1/d.shape[1], 1/d.shape[0])
    elem.translate(-.5,-.5,0)
    elem.scale(*(rescale,)*3)
    elem.translate(-rescale/2,-rescale/2,0)
    elem.scale(1,1,zscale)

    if  (layer == '' or
        (layer == 'real' and n==0) or
        (layer == 'imag' and n==1)):

        w.addItem(elem)

    if (layer == '' and n in [2,3]):
        w.addItem(elem)

for elem in [surf]:

    elem.scale( 1/(d.shape[0]*surf_smooth),
                1/(d.shape[1]*surf_smooth),
                1/(d.shape[0]*surf_smooth))

    elem.translate(-.5,-.5,0)
    elem.scale(*(rescale,)*3)
    elem.translate(-rescale/2,-rescale/2,0)
    elem.scale(1,1,zscale*surf_smooth)

    if (layer == '' or layer == 'surf'):
        w.addItem(elem)

# prep the particle mesh
ds = 100
particle_mesh = gl.MeshData.sphere(rows=ds, cols=ds)
particle_colors = np.ones((particle_mesh.faceCount(), 4), dtype=float)
particle_colors[:] = cmap(255)
particle_colors[:,3] = .25
particle_mesh.setFaceColors(particle_colors)

imag.rotate(90,0,0,1)
imag.translate(-2,0,0)

#w.orbit(+45, -10)
w.orbit(+45, -20)
#w.orbit(0, 0)

last_time = time()

if not folder.exists() and record:
    folder.mkdir()

prev,i = 0,0

def follow(pdf):
    global prev, i

    x, y = np.where(pdf == np.amax(pdf))
    x = x[0]
    y = y[0]

    #print('>>> FOLLOWING', x,y)

    x = x/pdf.shape[0]
    y = y/pdf.shape[1]

    x *= rescale
    y *= rescale

    x -= rescale/2
    y -= rescale/2

    xy = np.array([x,y])
    cx = np.array([w.opts['center'].x(), w.opts['center'].y()])
    dx = xy - cx

    #print('>>> FOLLOWING', xy)

    #ki = .05 # fast follow
    ki = .01 # medium follow
    #ki = .005 # slow follow
    i += ki*(dx)

    new_xy = i

    #d  = .005*((dx) - prev)
    #new_xy += d

    prev = (new_xy).copy()

    w.opts['center'] = Vector(new_xy[0], new_xy[1], 0)

def update():
    global surf, index, folder, last_time, start_time, record, fps, timer

    t = index

    if index >= frames and record:
        app.quit()

    if record:

        # (time / frame) * frames remaining
        ETA = (time()-last_time) * (frames-index)
        ETA = (ETA / 60) # sec to min ... / 60 # seconds to hours
        ETA = np.modf(ETA)
        ETA = int(ETA[1]), int(round(ETA[0]*60))
        ETA = str(ETA[0]) + ":" + str(ETA[1]).zfill(2)
        last_time = time()

        print(index, 'ETA', ETA)

        w.grabFrameBuffer().save(str(folder / f'{name}_{index}.png'))
    else:
        print(index, 'TIME', time()-last_time, 's', 'ELAPSED', index/fps, 's')
        last_time = time()

    ##print('>>>', time()-lt)
    lt = time()

    d = sim.simulate_frame(debug=0)
    #global d

    #print('>>>', time()-lt)
    lt = time()

    zdata = d
    zdata = np.abs(zdata)**2 # complex square: amplitude -> density

    if do_smoothing:
        zdata = surf_smoothing(zdata, smoothing=surf_smooth)

    #print('>>>', time()-lt)
    lt = time()

    cremap = lambda x: np.interp(x, [0,4], [0,1])
    zcol = cmap(cremap(zdata))
    zcol[:,:,3] = zdata + .1
    zcol[:,:,3] *= 0.75

    #print('>>>', time()-lt)
    lt = time()

    dreal, dimag = d.real, np.flipud(d.imag)
    rpoints, realcolors = make_wavelines(dreal, axis=0, smoothing=do_smoothing)
    ipoints, imagcolors = make_wavelines(dimag, axis=1, smoothing=do_smoothing)

    realcolors[:,0:3:2] *= rcol_bias # bias more red
    imagcolors[:,0] *= icol_bias # bias more blue

    #print('>>>', time()-lt)
    lt = time()

    if do_collapse:
        surf.setData(z=zdata/zdata.max(), colors=zcol)
        surf.setData(z=3*zdata/zdata.max(), colors=zcol)
        #surf.setData(z=zdata, colors=zcol)
    else:
        surf.setData(z=zdata, colors=zcol)

    real.setData(pos=rpoints, color=realcolors)
    imag.setData(pos=ipoints, color=imagcolors)

    #print('>>>', time()-lt)
    lt = time()

    dazim = -.09#+.05#0.25
    delev = +.005#1#+.05/10#1*2*.5
    ddist = 0#-.08#0#+.05
    #dazim = +.05#0.25
    #delev = +.05/10#1*2*.5
    #ddist = -.08#0#+.05

    #dazim = 0#+.05#0.25
    #delev = 0#+.005#1#+.05/10#1*2*.5
    #ddist = 0#-.08#0#+.05

    if 'follow' in name:
        follow(zdata)

    w.orbit(dazim, delev)
    w.setCameraPosition(distance=w.opts['distance']+ddist)

    #print('>>>', time()-lt); print()
    lt = time()

    index += 1

    if (sim.collapse and index % collapse_interval == 0 and index > 0

            and index < collapse_interval*2 # do once only

        ):

        #selection = sim.collapse_wavefunction()
        selection = sim.dual_collapse_wavefunction()

def ffmpeg(folder, name, FPS):

    if not record:
        return

    dest = Path('C:/')
    dest = dest / Path(f'Veritasium/ManyWorlds/{name}.mov')

    convert_cmd = (f'''ffmpeg -f image2 -framerate {FPS}'''
                       f''' -i {str(folder / name)}_%d.png'''
                       f''' -c:v prores_ks -profile:v 3'''
                       f''' "{str(dest)}" ''')

    print('CONVERTING >>>', convert_cmd)
    os.system(convert_cmd)

    if dest.exists():
        print('DELETING >>>', folder)
        filelist = [f for f in os.listdir(str(folder.absolute())) if f.endswith(".png") ]

        for f in filelist:
            os.remove(os.path.join(folder, f))

        folder.rmdir()

timer = QtCore.QTimer()
timer.timeout.connect(update)

start_time = time()
timer.start(0)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


ffmpeg(folder, name, fps)
