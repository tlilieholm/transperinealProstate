import numpy as np
import matplotlib.pyplot as plt #for debugging
from mpl_toolkits.mplot3d import Axes3D
from gridToolsv9 import vMag, makeVert, makeHoriz, makeX, rigidRegistration3p, findAngles, optimizeARMZ
"""
Thomas Lilieholm, UW Madison Dept. of Medical Physics, 2024

Main function for registration and target calculation with transperineal guidance grid for prostate interventions
This is the Python version- it has since been adapted into JavaScript for integration with RTHawk scanner interfacing software

Note that the function is reliant on predetermined measaurements of the physical dimensions of the grid (interfiducial spacings)
Change these values if the design of the grid is modified in the future.
"""


def battleshipAim(gridL, gridR, gridZ, targets, show=False, imX=256, imY=256, imZ=256, optimizeZ=False, verbose=False):
  """
  Main function for registration and target calculation
  Note that some scanner interfaces may report axes differently (inverted y-axis, swapped y- and z-axes etc.)
  Check beforehand to see how scanner coordinates may compare against the coordinate system used here
  Which, btw, is: [LR, AP, SI] where patient L, A, S are positive directions and patient R, P, I are negative

  Args:
    gridL (list): [LR, AP, SI] coordinates of the LowerLeft fiducial
    gridR (list): [LR, AP, SI] coordinates of the UpperRight fiducial
    gridZ (list): [LR, AP, SI] coordinates of the Arm fiducial (further along the Z-axis, hence gridZ)
    targets (list): List of lists in the above format for targeted points on the prostate, taken from the same scan
    show (Bool): flag to display wireframe visualization of the grid in magnetSpace and gridSpace reference frames. Useful for checking inputs are resonable
    imX (int): X-axis dimension for visualization, irrelevant if show=False, likely not worth changing otherwise
    imY (int): Y-axis dimension for visualization, irrelevant if show=False, likely not worth changing otherwise
    imZ (int): Z-axis dimension for visualization, irrelevant if show=False, likely not worth changing otherwise
    optimizeZ (Bool): Often, the z-axis resolution of the scan is lesser, this shifts the gridZ input slightly to account for resolution loss. Usually doesn't make much difference
    verbose (Bool): Flag to print more in-depth outputs

  Returns:
    None: prints information to console
  """

#THESE VALUES ARE HARDCODED AND BASED ON PHYSICAL GRID MEASUREMENTS- USE TO DFINE GRIDSPACE
  d = 101.02 #mm length of the diagonal
  theta = 8*np.pi/180 #degrees above perpendicular for the grid arm
  aL = 81.05 #mm length of the arm
  phi = 50.1*np.pi/180 #angle from LL to UR- it is NOT 45 degree, the area defined by them is slightly rectangular
  fidSpaceHoriz = 2.4 #length horizontally from column A to LL fiducial (or M to UR fid), in mm
  fidSpaceVert = 8.75 #distance vertically from row 1 to LL or row 13 to UR, in mm
  
#Values updated following precise measurements from Leif in the machine shop! V6
#The original design used a mixture of the metric and imperial systems, hence getting weird decimals
  
#####CALCULATIONS TO DEFINE RELATIVE SPACES
#Calculating relative vectors in the grid frame of reference
  LLtoARM = np.array([d*np.cos(phi)/2, (d*np.sin(phi))+(aL*np.sin(theta)), -aL*np.cos(theta)])
  LLARM=vMag(LLtoARM)
  URtoARM = np.array([-d*np.cos(phi)/2, aL*np.sin(theta), -aL*np.cos(theta)])
  URARM=vMag(URtoARM)
  LLtoUR = np.array([d*np.cos(phi), d*np.sin(phi), 0])
  LLtoOrigin = np.array([LLtoUR[0]/2, LLtoUR[1]/2, LLtoUR[2]/2])

  heightFactor = LLtoUR[1]/LLtoUR[0]
  
#Defining Points in GridSpace around the Origin
  gridSpaceLL = np.array([-LLtoOrigin[0], -LLtoOrigin[1], -LLtoOrigin[2]])
  gridSpaceARM = np.array([gridSpaceLL[0]+LLtoARM[0], gridSpaceLL[1]+LLtoARM[1], gridSpaceLL[2]+LLtoARM[2]])
  gridSpaceUR = np.array([gridSpaceLL[0]+LLtoUR[0], gridSpaceLL[1]+LLtoUR[1], gridSpaceLL[2]+LLtoUR[2]])

#V6 MODIFICATIONS
#########################################################################################################################
#Caclulating lengths in MagSpace (as reported by user input)
  if optimizeZ:
    gridZ[2] += optimizeARMZ(LLARM, URARM, gridL, gridR, gridZ)
    print(f"\nOptimization shifted ARMZ to {gridZ[2]}")

  diagLine = np.array([gridR[0]-gridL[0],gridR[1]-gridL[1],gridR[2]-gridL[2]])
  diagMag = vMag(diagLine)
  inputURtoARM = np.array([gridZ[0]-gridR[0], gridZ[1]-gridR[1], gridZ[2]-gridR[2]])
  inputLLtoARM = np.array([gridZ[0]-gridL[0], gridZ[1]-gridL[1], gridZ[2]-gridL[2]])
  
  planeOrigin = [(((max(gridR[0],gridL[0])-min(gridL[0],gridR[0]))/2)+min(gridL[0],gridR[0])),
  (((max(gridR[1],gridL[1])-min(gridL[1],gridR[1]))/2)+min(gridL[1],gridR[1])),
  (((max(gridR[2],gridL[2])-min(gridL[2],gridR[2]))/2)+min(gridL[2],gridR[2]))]
  
#Calculating discrepancies in dimensions from pixel to mm
  dFactor = d/vMag(np.array([gridR[0]-gridL[0],gridR[1]-gridL[1],gridR[2]-gridL[2]]))
  LLARMfactor = LLARM/vMag(np.array([gridZ[0]-gridL[0], gridZ[1]-gridL[1], gridZ[2]-gridL[2]]))
  URARMfactor = URARM/vMag(np.array([gridZ[0]-gridR[0], gridZ[1]-gridR[1], gridZ[2]-gridR[2]]))
#In V6+, tweak input along Z-axis to try and improve these factors###
  if verbose:
    print("\nRelative inter-point distances are:")
    print(f"LL to UR: {dFactor}")
    print(f"LL to ARM: {LLARMfactor}")
    print(f"UR to ARM: {URARMfactor}")
    print("If the above 3 factors are not roughly equal, reselect points")
#With 'perfect' user input, these factors  should all be the same. If 1 mm = 1 pixel, they'd be 1
#In practice, limitations of scanner resolution stymie this ideal
###########################################################################################################################

#Produces transformation matrix to bring user-reported magnet space points into grid space
  gridSpace = np.array([[gridSpaceLL[0],gridSpaceUR[0],gridSpaceARM[0]],[gridSpaceLL[1],gridSpaceUR[1],gridSpaceARM[1]],[gridSpaceLL[2],gridSpaceUR[2],gridSpaceARM[2]]])
  magnetSpace = np.array([[gridL[0],gridR[0],gridZ[0]],[gridL[1],gridR[1],gridZ[1]],[gridL[2],gridR[2],gridZ[2]]])
  rotMat, tMat = rigidRegistration3p(magnetSpace,gridSpace)
  if verbose:
    print(f'\nCalculated RotMat: \n{rotMat}')

#FINDS THE ANGLES OF ROTATION BETWEEN THE GRIDSPACE AND THE MAGNETSPACE  
  findAngles(rotMat, verbose = True)
  
  targetPoints=[] #V6
  projList=[]
  gridProjList=[]
  gridHoleList=[]
  gridTarList=[]
  magOriginToTarList=[]
  gridOriginToTarList=[]
  tarDistList=[]
  attainList=[] #V6
  j=0
  
  for i in targets:
     j+=1
     if verbose:
       print(f"\nTarget Point {j}")
     target=i
     magTar = makeVert(target)
     gridTar = makeHoriz(np.matmul(rotMat,magTar)+tMat)
     gridTarList.append(gridTar)
   
   #####PROJECTION CALCULATIONS
     originToTarget = np.array([target[0]-planeOrigin[0],target[1]-planeOrigin[1],target[2]-planeOrigin[2]])
     magOriginToTarList.append(originToTarget)
     
     gridSpaceOrigintoTar = np.array([gridTar[0], gridTar[1], gridTar[2]])
     gridOriginToTarList.append(gridSpaceOrigintoTar)
     
     gridNorm = np.array([0,0,1])
     distFromGrid = np.dot(gridSpaceOrigintoTar,gridNorm)
     tarDistList.append(distFromGrid)
     
     if verbose:
       print(f"Target depth: {distFromGrid+10} units")
     projectedPoint = gridTar - distFromGrid*gridNorm
     gridProjList.append(projectedPoint)
     
     if verbose:
       print(f"Target point in magnet coordinates: {target}")
       print(f"Target point in grid coordinates: {gridTar}")
       print(f"Projected point in grid coordinates: {projectedPoint}")
     targetPoints.append(target) #V6
   
   
   #####TRANSFORM PROJECTED POINT BACK TO MAGNET SPACE
     unTransform = makeVert(projectedPoint)-tMat
     unRotMat = np.linalg.inv(rotMat)
     magProjectedPoint = makeHoriz(np.matmul(unRotMat,unTransform))
     if verbose:
       print(f"Projected point in magnet coordinates: {magProjectedPoint}")
     projList.append(magProjectedPoint)
     
   #####NAMING GRID HOLES- SUBJECT TO CHANGE PENDING OFFICIAL GRID MEASUREMENTS#####
     gridDict = {-6:"A",-5:"B",-4:"C",-3:"D",-2:"E",-1:"F",0:"G",1:"H",2:"I",3:"J",4:"K",5:"L",6:"M"}
     width = LLtoUR[0]-(2*fidSpaceHoriz) #V6
     height = LLtoUR[1]-(2*fidSpaceVert) #V6
     xGridPos = projectedPoint[0]/(width/2)
     yGridPos = projectedPoint[1]/(height/2)
     xHole = round(xGridPos*6)
     yHole = round(yGridPos*6)+7
     if xHole in gridDict and (0<yHole<14):
       if verbose:
         print("Grid hole is: "+gridDict[xHole],yHole)
       gridHoleList.append(str(str(gridDict[xHole])+str(int(yHole))))
     else:
       if verbose:
         print("Target point does not fall within grid area.")
       gridHoleList.append("X")

    #######CHECKING BINNING DISPLACEMENT V6
     holeProj = np.array([xHole*5, (yHole-7)*5, distFromGrid])
     unTransform = makeVert(holeProj)-tMat
     unRotMat = np.linalg.inv(rotMat)
     magHoleProjection = makeHoriz(np.matmul(unRotMat,unTransform))
     print(f"{gridDict[xHole],yHole} projected in magnet coordinates: {magHoleProjection}")
     attainList.append(magHoleProjection)
    #######################################
   
   #####VERIFYING MAGSPACE GRID ORIENTATION
     magGridNorm = target-magProjectedPoint
     magGridNorm = magGridNorm/vMag(magGridNorm)
     if verbose:
       print("This value should be close to zero: "+str(np.dot(magGridNorm,diagLine/vMag(diagLine))))

  for i in range(len(targets)):
    print("\nTarget Point "+str(i+1)+":")
    print("Hole "+str(gridHoleList[i])+": "+str(np.round(tarDistList[i]+10,3))+" units deep.") #V7 Rounding
    lateralDisplacement = np.round((attainList[i]-targetPoints[i])[0],2) #V7
    verticalDisplacement = np.round((attainList[i]-targetPoints[i])[1],2) #V7
    
    #print(f"Placement will be {lateralDisplacement}mm Patient Left and {verticalDisplacement}mm Anterior due to hole binning") #V6, replaced by V7 for clarity
    
    #V7 modifications for greater clarity of binning displacement
    if (lateralDisplacement<0 and list(gridDict.values()).index(gridHoleList[i][0])==12) or (lateralDisplacement>0 and list(gridDict.values()).index(gridHoleList[i][0])==0):
      print(f"Due to grid size constraints, Lateral Trajectory may not reach the point- grid adjustment may be necessary.")
    elif lateralDisplacement>0:
      print(f"Due to binning, actual target is {lateralDisplacement}mm Patient Right of trajectory (towards column {gridDict[list(gridDict.values()).index(gridHoleList[i][0])-7]})")
    else:
      print(f"Due to binning, actual target is {-lateralDisplacement}mm Patient Left of trajectory (towards column {gridDict[list(gridDict.values()).index(gridHoleList[i][0])-5]})")

    if (verticalDisplacement<0 and int(gridHoleList[i][1:])==13) or (verticalDisplacement>0 and int(gridHoleList[i][1:])==1):
      print(f"Due to grid size constraints, Vertical Trajectory may not reach the point- changing grid legs may be necessary.")
    elif verticalDisplacement>0:
      print(f"Due to binning, actual target is {verticalDisplacement}mm Patient Posterior of trajectory (towards row {int(gridHoleList[i][1:])-1})")
    else:
      print(f"Due to binning, actual target is {-verticalDisplacement}mm Patient Anterior of trajectory (towards row {int(gridHoleList[i][1:])+1})")
    #V7 modifications for greater clarity of binning displacement

    print("==========================================")
    #print("MagnetSpace Coordinate is "+str(projList[i]))
    
#####IMAGE DISPLAY FOR DEBUGGING/VERIFICATION
  if show:
    #Defines the vectors and points of GridSpace Display
    magDiag2 = np.cross(magGridNorm,diagLine)
    magDiag2 = magDiag2/vMag(magDiag2)
    magDiag3 = np.cross(magGridNorm,diagLine)
    magUL = [planeOrigin[0]+magDiag2[0]*(diagMag/2)/heightFactor,planeOrigin[1]+magDiag2[1]*(diagMag/2)*heightFactor,planeOrigin[2]+magDiag2[2]*(diagMag/2)]
    magLR = [planeOrigin[0]-magDiag2[0]*(diagMag/2)/heightFactor,planeOrigin[1]-magDiag2[1]*(diagMag/2)*heightFactor,planeOrigin[2]-magDiag2[2]*(diagMag/2)]
    
    colorDict = {0:"Blue",1:"Red",2:"Green",3:"Black", 4:"Purple", 5:"Orange"}
    vectors=np.array([
    [gridSpaceLL[0],gridSpaceLL[1],gridSpaceLL[2],LLtoUR[0],LLtoUR[1],LLtoUR[2],2],
    [gridSpaceLL[0],gridSpaceLL[1],gridSpaceLL[2],LLtoUR[0],0,0,0],
    [gridSpaceLL[0],gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,0],
    [gridSpaceUR[0],gridSpaceUR[1],gridSpaceUR[2],0,-LLtoUR[1],0,0],
    [gridSpaceUR[0],gridSpaceUR[1],gridSpaceUR[2],-LLtoUR[0],0,0,0],
    [gridSpaceLL[0],gridSpaceLL[1],gridSpaceLL[2],LLtoARM[0],LLtoARM[1],LLtoARM[2],2],
    [gridSpaceUR[0],gridSpaceUR[1],gridSpaceUR[2],URtoARM[0],URtoARM[1],URtoARM[2],2],
    [0,0,0,gridNorm[0]*max(tarDistList),gridNorm[1]*max(tarDistList),gridNorm[2]*max(tarDistList),1],
    ])

    scale = d/100
    vectorsThatch=np.array([
    [gridSpaceLL[0]+2*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+7*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+12*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+17*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+22*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+27*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+32*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+37*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+42*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+47*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+52*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+57*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    [gridSpaceLL[0]+62*scale,gridSpaceLL[1],gridSpaceLL[2],0,LLtoUR[1],0,3],
    
    [gridSpaceLL[0],gridSpaceLL[1]+9*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+14*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+19*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+24*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+29*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],    
    [gridSpaceLL[0],gridSpaceLL[1]+34*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+39*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+44*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+49*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+54*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+59*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+64*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    [gridSpaceLL[0],gridSpaceLL[1]+69*scale,gridSpaceLL[2],LLtoUR[0],0,0,3],
    ])
    #thatching to represent grid hole pattern
  
  
    vectors=np.concatenate((vectors,makeX(gridSpaceLL,1)),axis=0)
    vectors=np.concatenate((vectors,makeX(gridSpaceUR,1)),axis=0)
    vectors=np.concatenate((vectors,makeX(gridSpaceARM,1)),axis=0)
    #X's to mark the fiducials
    
    for i in range(len(projList)):
      #vectors=np.concatenate((vectors,[[0,0,0,gridOriginToTarList[i][0],gridOriginToTarList[i][1],gridOriginToTarList[i][2],4]]),axis=0)
      vectors=np.concatenate((vectors,[[gridProjList[i][0],gridProjList[i][1],gridProjList[i][2],gridNorm[0]*tarDistList[i],gridNorm[1]*tarDistList[i],gridNorm[2]*tarDistList[i],1]]),axis=0)
      vectors=np.concatenate((vectors,makeX(gridTarList[i],4)),axis=0)
      vectors=np.concatenate((vectors,makeX(gridProjList[i],5)),axis=0)
  
    fig = plt.figure(num="Grid Space")
    ax = fig.add_subplot(111, projection='3d')
    for vector in vectors:
      v = np.array([vector[3],vector[4],vector[5]])
      ax.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
              pivot='tail', color=colorDict[vector[6]], arrow_length_ratio=0.01)
##
    for vector in vectorsThatch:
      v = np.array([vector[3],vector[4],vector[5]])
      ax.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
              pivot='tail', color=colorDict[vector[6]], arrow_length_ratio=0.01, linewidth=0.2)
##  
    ax.set_xlim([-imX/2,imX/2])
    ax.set_ylim([-imY/2,imY/2])
    ax.set_zlim([imZ/2,-imZ/2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
  
    #Defines the vectors and points of MagSpace Display
    vectors2=np.array([
    [gridL[0],gridL[1],gridL[2],diagLine[0],diagLine[1],diagLine[2],2],
    [gridL[0],gridL[1],gridL[2],gridZ[0]-gridL[0], gridZ[1]-gridL[1], gridZ[2]-gridL[2],2],
    [gridR[0],gridR[1],gridR[2],gridZ[0]-gridR[0], gridZ[1]-gridR[1], gridZ[2]-gridR[2],2],
    [planeOrigin[0],planeOrigin[1],planeOrigin[2],magGridNorm[0]*max(tarDistList),magGridNorm[1]*max(tarDistList),magGridNorm[2]*max(tarDistList),1],
    [gridR[0],gridR[1],gridR[2],magLR[0]-gridR[0], magLR[1]-gridR[1], magLR[2]-gridR[2],0],
    [gridR[0],gridR[1],gridR[2],magUL[0]-gridR[0], magUL[1]-gridR[1], magUL[2]-gridR[2],0],
    [gridL[0],gridL[1],gridL[2],magLR[0]-gridL[0], magLR[1]-gridL[1], magLR[2]-gridL[2],0],
    [gridL[0],gridL[1],gridL[2],magUL[0]-gridL[0], magUL[1]-gridL[1], magUL[2]-gridL[2],0],
    ])
  
    vectors2=np.concatenate((vectors2,makeX(target,4)),axis=0)
    vectors2=np.concatenate((vectors2,makeX(gridL,3)),axis=0)
    vectors2=np.concatenate((vectors2,makeX(gridR,3)),axis=0)
    vectors2=np.concatenate((vectors2,makeX(gridZ,3)),axis=0)
    vectors2=np.concatenate((vectors2,makeX(magProjectedPoint,5)),axis=0)
    
    for i in range(len(projList)):
      #vectors2=np.concatenate((vectors2,[[planeOrigin[0],planeOrigin[1],planeOrigin[2],magOriginToTarList[i][0],magOriginToTarList[i][1],magOriginToTarList[i][2],4]]),axis=0)
      vectors2=np.concatenate((vectors2,[[projList[i][0],projList[i][1],projList[i][2],magGridNorm[0]*tarDistList[i],magGridNorm[1]*tarDistList[i],magGridNorm[2]*tarDistList[i],1]]),axis=0)
      vectors2=np.concatenate((vectors2,makeX(targets[i],4)),axis=0)
      vectors2=np.concatenate((vectors2,makeX(projList[i],5)),axis=0)
    
    fig2 = plt.figure(num="Magnet Space")
    ax2 = fig2.add_subplot(111, projection='3d')
    for vector in vectors2:
      v = np.array([vector[3],vector[4],vector[5]])
      ax2.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
              pivot='tail', color=colorDict[vector[6]], arrow_length_ratio=0.01)
    ax2.set_xlim([-imX/2,imX/2])
    ax2.set_ylim([-imY/2,imY/2])
    ax2.set_zlim([imZ/2,-imZ/2])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    plt.show()
#display images for debugging
