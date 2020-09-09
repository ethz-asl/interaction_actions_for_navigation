

Header header

# Just one of them, one per iteration. 
int32 AcadoNx
int32 AcadoNod
int32 AcadoNplus
int32 AcadoNu

float64 MpcNodeComputationTime
float64 AcadoCost
float64 AcadoSuccessFlag

float64[] OnlineData
float64[] States
float64[] Outputs
float64[] Weights

# That array rubbish doesn't simplify things at all! 
#std_msgs::Float64MultiArray[] outputs
#std_msgs/Float64MultiArray onlineData
#std_msgs/Float64MultiArray states