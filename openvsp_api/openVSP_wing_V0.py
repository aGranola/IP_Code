from __future__ import print_function
import openvsp as vsp

# This is something that has to do with errors
stdout = vsp.cvar.cstdout
errorMgr = vsp.ErrorMgrSingleton.getInstance()

# Setup Check
vsp.VSPCheckSetup()
errorMgr.PopErrorAndPrint(stdout)

# Create Wing Instance
wing_id = vsp.AddGeom("WING")

# Split wing into 4 Sections (3 splits)
# This produces 5 XSecs (cross sectional areas that define the wing)
for i in [1, 2, 3]:
    vsp.InsertXSec(wing_id, i, vsp.XS_SIX_SERIES)
    vsp.CopyXSec(wing_id , i)


# Wing Section 1 "XSec_1" Body
vsp.SetParmValUpdate(wing_id, "Span", "XSec_1", 2.25)
vsp.SetParmValUpdate(wing_id, "Root_Chord", "XSec_1", 7)
vsp.SetParmValUpdate(wing_id, "Tip_Chord", "XSec_1", 3.5)
vsp.SetParmValUpdate(wing_id, "Sweep", "XSec_1", 37)
vsp.SetParmValUpdate(wing_id, "Sweep_Location", "XSec_1", 0)
vsp.SetParmValUpdate(wing_id, "Twist", "XSec_1", 0)
vsp.SetParmValUpdate(wing_id, "Twist_Location", "XSec_1", 0.25)
vsp.SetParmValUpdate(wing_id, "Dihedral", "XSec_1", 0)

# Wing Section 2 "XSec_2" Main wing
vsp.SetParmValUpdate(wing_id, "Span", "XSec_2", 9)
vsp.SetParmValUpdate(wing_id, "Root_Chord", "XSec_2", 3.5)
vsp.SetParmValUpdate(wing_id, "Tip_Chord", "XSec_2", 2)
vsp.SetParmValUpdate(wing_id, "Sweep", "XSec_2", 27)
vsp.SetParmValUpdate(wing_id, "Sweep_Location", "XSec_2", 0)
vsp.SetParmValUpdate(wing_id, "Twist", "XSec_2", 0)
vsp.SetParmValUpdate(wing_id, "Twist_Location", "XSec_2", 0.25)
vsp.SetParmValUpdate(wing_id, "Dihedral", "XSec_2", 2.5)

# Wing Section 3 "XSec_3" Winglet Transition
vsp.SetParmValUpdate(wing_id, "Span", "XSec_3", 0.5)
vsp.SetParmValUpdate(wing_id, "Root_Chord", "XSec_3", 2)
vsp.SetParmValUpdate(wing_id, "Tip_Chord", "XSec_3", 1.75)
vsp.SetParmValUpdate(wing_id, "Sweep", "XSec_3", 35)
vsp.SetParmValUpdate(wing_id, "Sweep_Location", "XSec_3", 0)
vsp.SetParmValUpdate(wing_id, "Twist", "XSec_3", 0)
vsp.SetParmValUpdate(wing_id, "Twist_Location", "XSec_3", 0.25)
vsp.SetParmValUpdate(wing_id, "Dihedral", "XSec_3", 40)

# Wing Section 4 "XSec_4" Winglet
vsp.SetParmValUpdate(wing_id, "Span", "XSec_4", 2.25)
vsp.SetParmValUpdate(wing_id, "Root_Chord", "XSec_4", 1.75)
vsp.SetParmValUpdate(wing_id, "Tip_Chord", "XSec_4", 0.75)
vsp.SetParmValUpdate(wing_id, "Sweep", "XSec_4", 40) 
vsp.SetParmValUpdate(wing_id, "Sweep_Location", "XSec_4", 0)
vsp.SetParmValUpdate(wing_id, "Twist", "XSec_4", 0)
vsp.SetParmValUpdate(wing_id, "Twist_Location", "XSec_4", 0.25)
vsp.SetParmValUpdate(wing_id, "Dihedral", "XSec_4", 80)


# Blending

# XSec_0 - main root chord

# Leading Edge
#OutLESweep = vsp.GetParm(wing_id, "OutLESweep", "LeadingEdge")

vsp.SetParmValUpdate(wing_id, "OutLESweep", "XSec_0", 10)
vsp.SetParmValUpdate(wing_id, "OutLEStrength", "XSec_0", 0.75)
vsp.SetParmValUpdate(wing_id, "OutLEDihedral", "XSec_0", -0.00)

# Trailing Edge
vsp.SetParmValUpdate(wing_id, "OutTESweep", "XSec_0", -15)
vsp.SetParmValUpdate(wing_id, "OutTEStrength", "XSec_0", 0.75)





# Writing file
fname = "openVSP_wing_V0_test.vsp3"
vsp.WriteVSPFile(fname)

# A final error check
geoms = vsp.FindGeoms()
errorMgr.PopErrorAndPrint(stdout)

