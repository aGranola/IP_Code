#%%
import openvsp as vsp
vsp.VSPCheckSetup()

#%%
# Set the root chord length to 5 meters
root_chord = 5.0

# Set the tip chord length to 1 meter
tip_chord = 1.0

# Set the wing span to 10 meters
span = 10.0


# Create a new wing
wing_id = vsp.AddGeom( "WING", "" )

# Get the wing geometry object
wing = vsp.FindGeom( wing_id, 1 )

# Set the wing origin to (0, 0, 0)
wing.SetOrigin( 0.0, 0.0, 0.0 )

# Set the wing semi-span to half of the span
wing.SetSpan( span / 2.0 )

# Add a root airfoil section
root_section = wing.AddXSec( 0.0, vsp.XS_FOUR_SERIES )
root_section.SetChord( root_chord )

# Add a tip airfoil section
tip_section = wing.AddXSec( span, vsp.XS_FOUR_SERIES )
tip_section.SetChord( tip_chord )

# Set the wing twist to 2 degrees
wing.SetTwist( 2.0 )

# Set the wing dihedral to 5 degrees
wing.SetDihedral( 5.0 )

# Set the wing sweep to 10 degrees
wing.SetSweep( 10.0 )

# Update the wing
wing.Update()

# Write the wing to a file
vsp.WriteVSPFile( "wing.vsp3" )



# %%
