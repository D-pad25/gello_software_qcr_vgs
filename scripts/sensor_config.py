import json
import numpa as np

# Define the default sensor positions
# WAS 0.008 INSTEAD OF 0.a?
y0 = 0.002
a = 0.00407

x = 0
sensor_positions = {
    "ref_pos": [0, 0, 0],  # Reference position (not used in calculations)
    "positions": [
        [x, y0, a*1.5], [x, y0+a, a*1.5], [x, y0+a*2, a*1.5], [x, y0+a*3, a*1.5],
        [x, y0, a*.5], [x, y0+a, a*.5], [x, y0+a*2, -a*.5], [x, y0+a*3, -a*.5],
        [x, y0, -a*.5], [x, y0+a, -a*.5], [x, y0+a*2, -a*.5], [x, y0+a*3, -a*.5],
        [x, y0, -a*1.5], [x, y0+a, -a*1.5], [x, y0+a*2, -a*1.5], [x, y0+a*3, -a*1.5],
        [x, -y0-a*3, a*1.5], [x, -y0-a*2, a*1.5], [x, -y0-a, a*1.5], [x, -y0, a*1.5],
        [x, -y0-a*3, a*.5], [x, -y0-a*2, a*.5], [x, -y0-a, a*.5], [x, -y0, a*.5],
        [x, -y0-a*3, -a*.5], [x, -y0-a*2, -a*.5], [x, -y0-a, -a*.5], [x, -y0, -a*.5],
        [x, -y0-a*3, -a*1.5], [x, -y0-a*2, -a*1.5], [x, -y0-a, -a*1.5], [x, -y0, -a*1.5]
    ]
}

# Save to a JSON file
with open("sensor_positions.json", "w") as f:
    json.dump(sensor_positions, f, indent=4)

print("Sensor positions saved in 'sensor_positions.json'")
