# Quick look only: unlike vizconfig.py, this does not unwrap positions across
# the periodic x/y boundary, so a vortex line anchored near x=0/Lx or y=0/Ly
# can show as a long spurious diagonal segment. Use vizconfig.py for a
# publication-quality flux-line plot.
set title "3D Vortex Line Lattice"
set xlabel "X"
set ylabel "Y"
set zlabel "Z Layer"
set ticslevel 0
set view 65, 35
set grid

# 1. Skip header (skip 3)
# 2. Sort by the persistent in-plane label ($7 = global_id % Nxy, constant
#    across layers), then Z ($3). $6 (global id) differs per layer for the
#    same physical vortex line, so grouping on it would never connect points
#    across planes.
# 3. Use AWK to insert a blank line whenever the label ($7) changes
data_cmd = "< awk 'NR>3 && NF>=3 {print $0}' config_step_0.dat | sort -k7,7n -k3,3n | awk '{if (NR>1 && $7!=prev) print \"\"; prev=$7; print $0}'"

splot data_cmd using 1:2:3:7 with linespoints pt 7 ps 0.8 lc variable title "Vortex Lines"