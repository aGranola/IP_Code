import openvsp as vsp
import numpy as np
from cst_modeling.section import cst_foil, cst_curve, cst_foil_fit, foil_bump_modify, foil_increment
from matplotlib import pyplot as plt

#CREATE SOME TEST GEOMETRIES
print("--> Generating Geometries")
print("")

#add components
pod_id = vsp.AddGeom("POD", "")
wing_id = vsp.AddGeom("WING", "")

#set paramters
vsp.SetParmVal(wing_id, "X_Rel_Location", "XForm", 2.5)
vsp.SetParmVal(wing_id, "TotalArea", "WingGeom", 25)

subsurf_id = vsp.AddSubSurf(wing_id, vsp.SS_CONTROL, 0)

vsp.Update()

#SETUP EXPORT FILENAMES#
fname_vspaerotests = "TestVSPAero.vsp3"

#SAVE VEHICLE TO FILE
#check progress and operation
print("--> Saving vehicle file to: ", False)
print(fname_vspaerotests, True)
print("")
vsp.WriteVSPFile(fname_vspaerotests, vsp.SET_ALL)
print("COMPLETE\n")
vsp.Update()
