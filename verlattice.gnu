set title "3D Vortex Line Lattice"
set xlabel "X"
set ylabel "Y"
set zlabel "Z Layer"
set ticslevel 0
set view 65, 35
set grid

# 1. Skip header (skip 3)
# 2. Sort by Vortex ID ($6), then Z ($3)
# 3. Use AWK to insert a blank line whenever Vortex ID ($6) changes
data_cmd = "< awk 'NR>3 && NF>=3 {print $0}' config_step_0.dat | sort -k6,6n -k3,3n | awk '{if (NR>1 && $6!=prev) print \"\"; prev=$6; print $0}'"

splot data_cmd using 1:2:3:6 with linespoints pt 7 ps 0.8 lc variable title "Vortex Lines"