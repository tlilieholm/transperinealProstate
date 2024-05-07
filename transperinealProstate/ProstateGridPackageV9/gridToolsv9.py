import numpy as np
"""
Thomas Lilieholm, UW Madison Dept. of Medical Physics, 2024

Assorted supplementary tools for the protate intervention grid process- Python version, with no RTHawk integration

Note that pointSim is reliant on predetermined measurements of the physical dimensions of the grid (interfiducial spacings)
Change these values if the dimensions of the grid change.
"""

def vMag(vector):
  """
  Use to calculate magnitude of vectors- only 3D
  I wrote this myself because this is meant to be ported in RTHawk, which does not have numpy support
  Accordingly, many functions are written using as few libraries as possible

  Args:
    vector (array): the vector to calculate the magnitude of
    
  Returns:
    magnitude (float): the magnitude of the vector
  """
  magnitude  = (vector[0]**2+vector[1]**2+vector[2]**2)**(1/2)
  return magnitude

def makeVert(vector):
  """
  Use to make a 'horizontal' vector 'vertical'
  That is, go from 1x3 to 3x1. This is to simplify matrix math later
  Again, coded myself instead of just using .T for RTH integration

  Args:
    vector (array): the vector to transpose

  Returns:
    vertical (array): the vector, having been transposed
  """
  vertical = np.array([[vector[0]],[vector[1]],[vector[2]]])
  return vertical

def makeHoriz(vector):
  """
  Inverse of the makeVert function- go from 3x1 to 1x3
  Again, because RTH and matrix math

  Args:
    vector (array): the vector to transpose

  Returns:
    horizontal (array): the vector, having been transposed
  """
  horizontal = np.array([vector[0][0], vector[1][0], vector[2][0]])
  return horizontal


def makeX(point, colorKey):
  """
  For use in plot visualization- makes a small 'X' on the graph to denote points rather than vectors

  Args:
    point (1x3 array): the point in 3-space on which to center the X
    colorKey: the color to make the X, subject to a dictionary in the main function: {0:"Blue",1:"Red",2:"Green",3:"Black", 4:"Purple"}

  Returns:
    An array of vectors describing the X
    
  """
  vec1 = np.array([point[0]-5, point[1]-5, point[2], 10, 10, 0, colorKey])
  vec2 = np.array([point[0]-5, point[1]+5, point[2], 10, -10, 0, colorKey])
  return np.array([vec1,vec2])

###
def rigidRegistration3p(mat1, mat2):
  """
  The rigid registration algorithm, based on:
    K. S. Arun, T. S. Huang and S. D. Blostein, "Least-Squares Fitting of Two 3-D Point Sets,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-9, no. 5,
    pp. 698-700, Sept. 1987, doi: 10.1109/TPAMI.1987.4767965.
  Used to fit user-reported inputs to an idealized port orientation

  This could be expanded to take more than 3 points, but at this point, the grid itself only has 3 fiducials
  Future design iterations may change this

  Args:
    mat1 (3x3 array): [[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]] matrix of 3 points in 3D space, this one is the observed point set
    mat2 (3x3 array): matrix of 3 points in 3D space, this is the idealized pointset based on known measurements

  Returns:
    rotMat (3x3 array): rotation matrix for registration
    transMat (3x1 array): translation matrix for registration
  """
  assert mat1.shape==mat2.shape, f'Registration input mismatch {mat1.shape}!={mat2.shape}'

  #calculate centroids
  centroid1 = np.mean(mat1, axis=1)
  centroid2 = np.mean(mat2, axis=1)

  centroid1 = centroid1.reshape(-1, 1)
  centroid2 = centroid2.reshape(-1, 1)

  #shift to origin
  mat1M = mat1 - centroid1
  mat2M = mat2 - centroid2
  H = mat1M @ np.transpose(mat2M)

  #perform singular value decomposition
  U, S, Vt = np.linalg.svd(H)
  rotMat = Vt.T @ U.T

  #catch reflection case
  if np.linalg.det(rotMat) < 0:
      Vt[2,:] *= -1
      rotMat = Vt.T @ U.T

  #calculate transMat
  transMat = centroid2 - (rotMat @ centroid1)
    
  return rotMat, transMat
###

def pointSim(gridL=[0,0,0], roll=0, pitch=0, yaw=0):
  """
  Use to simulate a rotated/translated grid coordinate set for testing the registration without having to take scans
  For true assurances of accuracy, test later with actual scanned values

  Args:
    gridL (1x3 array): location to put the lowerleft fiducial- assumed [0,0,0], this translates the whole set to match
    roll (float): degree amount to simulate roll- roll is about z-axis 
    pitch (float): degree amount for pitch, rotation about x-axis
    yaw (float): degree for yaw about y-axiz

  Returns:
    portHole (1x3 array): returns where the hole fiducial/point would be, given rotations
    gridR (1x3 array): returns where cross2 fiducial would be, given rotations
    gridARM (1x3 array): returns where the arc1 fiducial would be, given rotations
  """
#THESE VALUES ARE HARDCODED AND BASED ON PHYSICAL GRID MEASUREMENTS
  d = 101.02 #mm length of the diagonal (101mm)
  theta = 8*np.pi/180 #degrees above perpendicular for the grid arm (22deg measured- 7 degrees in experiments)
  aL = 81.05 #mm length of the arm (80mm)
  phi = 50.1*np.pi/180 #angle from LL to UR- it is NOT 45 degree, the area defined by them is slightly rectangular
  fidSpaceHoriz = 2.4 #length horizontally from column A to LL fiducial (or M to UR fid), in mm
  fidSpaceVert = 8.75 #distance vertically from row 1 to LL or row 13 to UR, in mm
#CHANGE THESE VALUES IF THE GRID CHANGES
  
#Predicts the other two points based on one of the input points and the expected vectors
  LLtoARM = np.array([d*np.cos(phi)/2, (d*np.sin(phi))+(aL*np.sin(theta)) , -aL*np.cos(theta)])
  LLtoUR = np.array([d*np.cos(phi), d*np.sin(phi), 0])
  gridR = np.array([LLtoUR[0], LLtoUR[1], 0])
  gridARM = np.array([LLtoARM[0], LLtoARM[1], LLtoARM[2]])

  b = -roll*np.pi/180
  a = pitch*np.pi/180
  c = yaw*np.pi/180
  
  yawMat = np.array([[np.cos(b),-np.sin(b),0],
            [np.sin(b), np.cos(b),0],
            [0, 0, 1]])
            
  pitchMat = np.array([[np.cos(c),0,np.sin(c)],
              [0,1,0],
              [-np.sin(c), 0, np.cos(c)]])
              
  rollMat = np.array([[1,0,0],
              [0,np.cos(a),-np.sin(a)],
              [0,np.sin(a), np.cos(a)]])


  rotMat = (yawMat@pitchMat@rollMat)
  print(f'Simulation rotMat: \n{rotMat}')
  
  gridR = makeVert(gridR)
  gridARM = makeVert(gridARM)
  
  gridR = makeHoriz((np.matmul(rotMat,gridR)))
  gridARM = makeHoriz((np.matmul(rotMat,gridARM)))
#Rotates the UR and ARM points about axes relative to the LL point
  
  gridR = [gridL[0]+gridR[0], gridL[1]+gridR[1], gridL[2]+gridR[2]]
  gridARM = [gridL[0]+gridARM[0], gridL[1]+gridARM[1], gridL[2]+gridARM[2]]
#If LL isn't treated as origin, this shift the roated UR and ARM points as needed

  return gridR, gridARM

def trigCheck(val):
  """
    Use to correct an annoyingly persistent rounding issue that occasionally occurs in matrix math
    Values for arcsin must stay between -1 and 1, sometimes the values gets set to 1.0000000000000002- this corrects that

    Args:
        val (float): number to be rounded within -1 to 1
    Returns:
        output (int): the value, truncated to 1 or -1
  """
  if val > 1:
    print(f'Rounding error: {val}>1')
    output = 1
  elif val < -1:
    output = -1
    print(f'Rounding error: {val}<-1')
  else:
    output = val
  return output

def findAngles(rotMat, verbose = False):
  """
    Use to convert rotMat into a comprehensible form, listing the yaw, pitch, and roll in degrees
    Note that some redundancies/degeneracies can occur at specific values (exactly 90 degrees, see Gimbal Lock)
    These angular values do not occur in practice, where angular discrepancies seldom exceed 10 degrees

    Args:
        rotMat (array): rotation matrix between gridSpace and magnetSpace, as output by rigidRegistration3p
        verbose (bool): flag to output additional information
    Return:
        [pitch, roll, yaw]: the rotations of the grid, in degrees
  """
  rotMat = rotMat.T
  
  yaw=np.arcsin(trigCheck(-rotMat[2][0]))
  roll=np.arccos(trigCheck(rotMat[0][0]/np.cos(yaw)))
  pitch=np.arcsin(trigCheck(rotMat[2][1]/np.cos(yaw)))

  pitch=round(pitch*180/np.pi,2)
  roll=round(roll*180/np.pi,2)
  yaw=round(yaw*180/np.pi,2)
  if verbose:  
    print(f"Pitch: {pitch}, Roll: {roll}, Yaw: {yaw}")
  return [pitch, roll, yaw]

def optimizeARMZ(LLARM, URARM, gridL, gridR, gridZ, verbose=False):
  """
    Optional, simple, optimization to be applied to point registration to account for reduced resolution along z-axis (nonisotropic scan, thicker slices, etc.)
    If you can, run a scan with finer resolution on the SI axis. Otherwise, use this to try and improve the fit
    It shifts the ARM fiducial along the z-axis to find a better match for prior-known interfiducial measurements

    Args:
        LLARM (float): distance from LowerLeft fiducial to ARM fiducial
        URARM (float): distance from UpperRight fiducial to ARM fiducial
        gridL (float): position of LowerLeft fiducial
        gridR (float): position of UpperRight fiducial
        gridZ (float): position of ARM fiducial
        verbose (bool): flag to print details of the refitting

    Returns:
        prevShift (int): newly-shifted ARM fiducial location
  """
  base_LLARMfactor = LLARM/vMag(np.array([gridZ[0]-gridL[0], gridZ[1]-gridL[1], gridZ[2]-gridL[2]]))
  base_URARMfactor = URARM/vMag(np.array([gridZ[0]-gridR[0], gridZ[1]-gridR[1], gridZ[2]-gridR[2]]))
  base_factor=(base_URARMfactor+base_LLARMfactor)/2
  if verbose:
    print(f"Original ARMZ = {gridZ[2]}, base_factor: {base_factor}")

  #simple optimization- account for 5mm resolution on SI axis
  optimize=True
  zshift, prevshift, prev_factor = 1, 0, base_factor

  while optimize:
    testZ=gridZ[2]+zshift
    cur_LLARMfactor = LLARM/vMag(np.array([gridZ[0]-gridL[0], gridZ[1]-gridL[1], testZ-gridL[2]]))
    cur_URARMfactor = URARM/vMag(np.array([gridZ[0]-gridR[0], gridZ[1]-gridR[1], testZ-gridR[2]]))
    cur_factor=(cur_URARMfactor+cur_LLARMfactor)/2
    if verbose:
      print(f"Current ARMZ: {testZ}, Zshift: {zshift}, cur_factor: {cur_factor}")
    #try new z value, calculate fit
    
    if (abs(1-cur_factor)>abs(1-prev_factor)) and (zshift==1): #first guess is worse, must reverse direction
      prevshift=zshift
      zshift=-1

    elif (abs(1-cur_factor)>abs(1-prev_factor)): #worse, but not first guess- optimization done
      if verbose:
        print('Optimization Done')
      optimize=False

    elif (abs(1-cur_factor)<abs(1-prev_factor)): #better, keep going
      prevshift=zshift
      zshift=(abs(zshift)+1)*np.sign(zshift)
    
    else:
      optimize=False
      print('Exception Occured')
    prev_factor=cur_factor
  if verbose:
    print(f"Optimal zshift to ARM is {prevshift}")
  return prevshift
