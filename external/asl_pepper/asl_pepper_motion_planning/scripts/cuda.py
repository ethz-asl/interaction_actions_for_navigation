
import numpy as np
from map2d import Map2D, SimAgent
from timeit import default_timer as timer
from pepper_2d_simulator import compiled_raytrace

## ----------CUDA ---------------
import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule
kernel = """
__global__ void multiply_them(float *angles, float *ranges, float *occupancy, 
            int occupancy_shape_i, int occupancy_shape_j,
            float threshold, float resolution, float o_i, float o_j )
{
  const int k = threadIdx.x;

  ranges[k] = 0;
  float angle = angles[k];
  float i_inc = cos(angle);
  float j_inc = sin(angle);
  // Stretch the ray so that every 1 unit in the ray direction lands on a cell in i or 
  float i_abs_inc = abs(i_inc);
  float j_abs_inc = abs(j_inc);
  float raystretch;
  if ( i_abs_inc >= j_abs_inc ) {
      raystretch = 1. / i_abs_inc;
  } else {
      raystretch = 1. / j_abs_inc;
  }
  i_inc = i_inc * raystretch;
  j_inc = j_inc * raystretch;
  // max amount of increments before crossing the grid border
  int max_inc;
  if (i_inc == 0) {
      if ( j_inc >= 0 ) { 
          max_inc = floor((occupancy_shape_j - 1 - o_j) / j_inc);;
      } else {
          max_inc = floor(o_j / -j_inc);
      }
  } else if ( j_inc == 0 ) {
      if ( i_inc >= 0 ) {
          max_inc = floor((occupancy_shape_i - 1 - o_i) / i_inc) ;
      } else {
          max_inc = floor(o_i / -i_inc);
      }
  } else {
      int max_i_inc;
      int max_j_inc;
      if ( j_inc >= 0 ) {
        max_i_inc = floor((occupancy_shape_j - 1 - o_j) / j_inc) ;
      } else { 
        max_i_inc = floor(o_j / -j_inc);
      }
      if ( i_inc >= 0 ) {
        max_j_inc = floor((occupancy_shape_i - 1 - o_i) / i_inc) ;
      } else {
        max_j_inc = floor(o_i / -i_inc);
      }
      if ( max_i_inc <= max_j_inc ) {
        max_inc = max_i_inc;
      } else {
        max_inc = max_j_inc;
      }
  }
  // Trace a ray
  int n_i = o_i + 0;
  int n_j = o_j + 0;

  for (int n = 1; n < (max_inc - 1); n++) {

if ( threadIdx.x == 0 && blockIdx.x == 0 ) {
    printf("Hello from block %d, thread %d\\n", blockIdx.x, threadIdx.x);
}
        

      // makes the beam 'thicker' by checking intermediate pixels
      float occ3 = occupancy[int(n_i) * occupancy_shape_j +  int(n_j+j_inc)]; 
      n_i += i_inc;
      float occ2 = occupancy[int(n_i) * occupancy_shape_j +  int(n_j)];
      n_j += j_inc;
      float occ  = occupancy[int(n_i) * occupancy_shape_j +  int(n_j)];
      if ( occ >= threshold || occ2 >= threshold || occ3 >= threshold ) {
          float dx = ( n_i - o_i ) * resolution;
          float dy = ( n_j - o_j ) * resolution;
          float r = sqrt(dx*dx + dy*dy);
          ranges[k] = r;
          break;
      }
  }
}
"""
mod = SourceModule(kernel)

multiply_them = mod.get_function("multiply_them")



map_folder = "."
map_name = "empty"
map2d = Map2D(map_folder, map_name)
print("Map '{}' loaded.".format(map_name))

lidar_pos = np.array([0, 0, 0])
kLidarMinAngle = -2.35619449615
kLidarMaxAngle = 2.35572338104
kLidarAngleIncrement = 0.00581718236208
angles = np.arange(kLidarMinAngle, kLidarMaxAngle, kLidarAngleIncrement) + lidar_pos[2]
ranges = np.zeros_like(angles)
occupancy = map2d.occupancy
lidar_pos_ij = map2d.xy_to_ij(lidar_pos[:2])

tic =timer()
compiled_raytrace(angles, lidar_pos_ij, map2d.occupancy(),
        map2d.thresh_occupied(), map2d.resolution_, map2d.origin, ranges)
toc = timer()
print("raytrace: {}s".format(toc-tic))
tic =timer()
aa = drv.In(angles)
rr = drv.Out(ranges)
oo = drv.In(occupancy.flatten())
print("driver ok")
multiply_them(aa, rr, oo,
        np.intp(occupancy.shape[0]),
        np.intp(occupancy.shape[1]),
        np.float32(map2d.thresh_occupied()), 
        np.float32(map2d.resolution()), 
        np.float32(lidar_pos_ij[0]),
        np.float32(lidar_pos_ij[1]),
        block=(400,1,1), grid=(1,1), 
        )
toc = timer()
print("cuda raytrace: {}s".format(toc-tic))
other_agents = []
for i in range(8):
    # populate agents list
    agent = SimAgent()
    agent.pose_2d_in_map_frame = np.array([0,0,0])
    agent.type = "legs"
    agent.state = np.array([0, 0, 0])
    other_agents.append(agent)
tic = timer()
ranges = map2d.render_agents_in_lidar(ranges, angles, other_agents, lidar_pos_ij)
toc = timer()
print("render: {}s".format(toc-tic))
