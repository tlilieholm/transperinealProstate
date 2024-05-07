from gridV9 import battleshipAim

def main():
  """
  Thomas Lilieholm, UW Madison Dept. of Medical Physics, 2024

  Main functions for prostate grid applications- use this file when actually performing a procedure
  After acquiring a scan covering all 3 fiducials and the prostate, use scanner tools to determine relevant coordinate points
  Then input those below, where indicated

  A reasonable set of dummy coordinates and targets are already in place for one registration and 10 targets
  Replace these with actual registration/target coordinates in practice
  """
  gridLL = [-26.57, -37.68, -112.74] #(Lower Left fiducial)
  gridUR = [36.20, 41.54, -117.74] #(Upper Right fiducial)
  gridARM = [-0.94, 52.12, -197.74] #(ARM (noncoplanar) fiducial)
  #Input the coordinate locations of the three fiducials as: [LR, AP, SI]
  #Note: patient left, anterior, and superior are the positive directions
  #Use units of mm
  
  targets = [[-11.52, -6.88, -32.74],
             [5.87, -7.82, -37.74],
             [-15.05, -2.89, -47.74],
             [8.46, -2.42, -42.74],
             [-18.79, -14.03, -32.74],
             [11.96, -15.48, -32.74],
             [-17.40, -25.22, -32.74],
             [12.69, -25.22, -32.74],
             [-3.53, -28.27, -32.74],
             [7.99, -13.00, -32.74]]
  #Input coordinates of target points on prostate
  #As many as desired, in the form: [LR, AP, SI]
  #mm units again

###################################################################
  if (gridLL[0]>gridUR[0]) and (gridARM[2]>gridLL[2]):
    print("Radiologic View Detected")
    gridLL[0], gridLL[2] = -gridLL[0], -gridLL[2]
    gridUR[0], gridUR[2] = -gridUR[0], -gridUR[2]
    gridARM[0], gridARM[2] = -gridARM[0], -gridARM[2]
    for i in targets:
      i[0], i[2] = -i[0], -i[2]
  #Correction for Radiologic View (flipped LR and SI axes)      
  
  print(f"\nLL coordinates: {gridLL}")
  print(f"UR coordinates: {gridUR}")
  print(f"ARM coordinates: {gridARM}")
  j = 1
  for i in targets:
    print(f"TAR{j} coordinates: {i}")
    j+=1
  #Printing values for verification prior to calculation
  
  battleshipAim(gridLL, gridUR, gridARM, targets, show=True, optimizeZ=True, verbose=True)

main()
