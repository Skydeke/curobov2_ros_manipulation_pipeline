# Single scene-obstacle spec (mirrors AddObject.srv request fields so one
# bulk call carries exactly what N add_object calls would).

# Possible object types (identical values to AddObject.srv)
int8 CUBOID=0
int8 SPHERE=1
int8 CAPSULE=2
int8 CYLINDER=3
int8 MESH=4

int8 type
string name
# Mesh objects: either a file path (absolute) or inline vertex/triangle data.
# When `vertices`/`triangles` are non-empty they take precedence.
string mesh_file_path ""
geometry_msgs/Point[] vertices        # inline mesh: raw vertex buffer
int32[] triangles                     # inline mesh: flat triangle index buffer
geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions      # CUBOID: [dx, dy, dz]; SPHERE: [r, _, _];
                                      # CAPSULE/CYLINDER: [radius, height, _];
                                      # MESH: [scale_x, scale_y, scale_z]
std_msgs/ColorRGBA color
