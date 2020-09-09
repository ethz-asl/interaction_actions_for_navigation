/*
*    This file is part of ACADO Toolkit.
*
*    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
*    Copyright (C) 2008-2009 by Boris Houska and Hans Joachim Ferreau, K.U.Leuven.
*    Developed within the Optimization in Engineering Center (OPTEC) under
*    supervision of Moritz Diehl. All rights reserved.
*
*    ACADO Toolkit is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 3 of the License, or (at your option) any later version.
*
*    ACADO Toolkit is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with ACADO Toolkit; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*
*/


/**
*    Author David Ariens, Rien Quirynen
*    Date 2009-2013
*    http://www.acadotoolkit.org/matlab 
*/

#include <acado_optimal_control.hpp>
#include <acado_toolkit.hpp>
#include <acado/utils/matlab_acado_utils.hpp>

USING_NAMESPACE_ACADO

#include <mex.h>


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) 
 { 
 
    MatlabConsoleStreamBuf mybuf;
    RedirectStream redirect(std::cout, mybuf);
    clearAllStaticCounters( ); 
 
    mexPrintf("\nACADO Toolkit for Matlab - Developed by David Ariens and Rien Quirynen, 2009-2013 \n"); 
    mexPrintf("Support available at http://www.acadotoolkit.org/matlab \n \n"); 

    if (nrhs != 0){ 
      mexErrMsgTxt("This problem expects 0 right hand side argument(s) since you have defined 0 MexInput(s)");
    } 
 
    TIME autotime;
    DifferentialState x;
    DifferentialState y;
    DifferentialState x_speed;
    DifferentialState y_speed;
    DifferentialState x_turmoil;
    DifferentialState y_turmoil;
    DifferentialState x_turmoil_dot;
    DifferentialState y_turmoil_dot;
    DifferentialState x_agent1;
    DifferentialState x_agent2;
    DifferentialState x_agent3;
    DifferentialState x_agent4;
    DifferentialState x_agent5;
    DifferentialState y_agent1;
    DifferentialState y_agent2;
    DifferentialState y_agent3;
    DifferentialState y_agent4;
    DifferentialState y_agent5;
    DifferentialState r_agent1;
    DifferentialState r_agent2;
    DifferentialState r_agent3;
    DifferentialState r_agent4;
    DifferentialState r_agent5;
    OnlineData x_goal_int; 
    OnlineData y_goal_int; 
    OnlineData x_speed_goal_int; 
    OnlineData y_speed_goal_int; 
    OnlineData obstacles_x1; 
    OnlineData obstacles_x2; 
    OnlineData obstacles_x3; 
    OnlineData obstacles_x4; 
    OnlineData obstacles_x5; 
    OnlineData obstacles_x6; 
    OnlineData obstacles_x7; 
    OnlineData obstacles_x8; 
    OnlineData obstacles_x9; 
    OnlineData obstacles_x10; 
    OnlineData obstacles_y1; 
    OnlineData obstacles_y2; 
    OnlineData obstacles_y3; 
    OnlineData obstacles_y4; 
    OnlineData obstacles_y5; 
    OnlineData obstacles_y6; 
    OnlineData obstacles_y7; 
    OnlineData obstacles_y8; 
    OnlineData obstacles_y9; 
    OnlineData obstacles_y10; 
    OnlineData obstacles_r1; 
    OnlineData obstacles_r2; 
    OnlineData obstacles_r3; 
    OnlineData obstacles_r4; 
    OnlineData obstacles_r5; 
    OnlineData obstacles_r6; 
    OnlineData obstacles_r7; 
    OnlineData obstacles_r8; 
    OnlineData obstacles_r9; 
    OnlineData obstacles_r10; 
    OnlineData x_speed_agent1; 
    OnlineData x_speed_agent2; 
    OnlineData x_speed_agent3; 
    OnlineData x_speed_agent4; 
    OnlineData x_speed_agent5; 
    OnlineData y_speed_agent1; 
    OnlineData y_speed_agent2; 
    OnlineData y_speed_agent3; 
    OnlineData y_speed_agent4; 
    OnlineData y_speed_agent5; 
    OnlineData yaw; 
    Control u_x_speed_dot;
    Control u_y_speed_dot;
    BMatrix acadodata_M1;
    acadodata_M1.read( "mpc_controller_data_acadodata_M1.txt" );
    BMatrix acadodata_M2;
    acadodata_M2.read( "mpc_controller_data_acadodata_M2.txt" );
    Function acadodata_f2;
    acadodata_f2 << x;
    acadodata_f2 << y;
    acadodata_f2 << x_speed;
    acadodata_f2 << y_speed;
    acadodata_f2 << x_turmoil;
    acadodata_f2 << y_turmoil;
    acadodata_f2 << x_turmoil_dot;
    acadodata_f2 << y_turmoil_dot;
    acadodata_f2 << 1.00000000000000000000e+01*sqrt((1.00000000000000004792e-04+1.25000000000000000000e+01*pow(x_turmoil,2.00000000000000000000e+00)+5.00000000000000000000e-01*pow(x_turmoil_dot,2.00000000000000000000e+00)));
    acadodata_f2 << 1.00000000000000000000e+01*sqrt((1.00000000000000004792e-04+1.25000000000000000000e+01*pow(y_turmoil,2.00000000000000000000e+00)+5.00000000000000000000e-01*pow(y_turmoil_dot,2.00000000000000000000e+00)));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r1+sqrt((pow((-obstacles_x1+x),2.00000000000000000000e+00)+pow((-obstacles_y1+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r2+sqrt((pow((-obstacles_x2+x),2.00000000000000000000e+00)+pow((-obstacles_y2+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r3+sqrt((pow((-obstacles_x3+x),2.00000000000000000000e+00)+pow((-obstacles_y3+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r4+sqrt((pow((-obstacles_x4+x),2.00000000000000000000e+00)+pow((-obstacles_y4+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r5+sqrt((pow((-obstacles_x5+x),2.00000000000000000000e+00)+pow((-obstacles_y5+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r6+sqrt((pow((-obstacles_x6+x),2.00000000000000000000e+00)+pow((-obstacles_y6+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r7+sqrt((pow((-obstacles_x7+x),2.00000000000000000000e+00)+pow((-obstacles_y7+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r8+sqrt((pow((-obstacles_x8+x),2.00000000000000000000e+00)+pow((-obstacles_y8+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r9+sqrt((pow((-obstacles_x9+x),2.00000000000000000000e+00)+pow((-obstacles_y9+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-obstacles_r10+sqrt((pow((-obstacles_x10+x),2.00000000000000000000e+00)+pow((-obstacles_y10+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent1+sqrt((pow((x-x_agent1),2.00000000000000000000e+00)+pow((y-y_agent1),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent2+sqrt((pow((x-x_agent2),2.00000000000000000000e+00)+pow((y-y_agent2),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent3+sqrt((pow((x-x_agent3),2.00000000000000000000e+00)+pow((y-y_agent3),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent4+sqrt((pow((x-x_agent4),2.00000000000000000000e+00)+pow((y-y_agent4),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f2 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent5+sqrt((pow((x-x_agent5),2.00000000000000000000e+00)+pow((y-y_agent5),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f2 << u_x_speed_dot;
    acadodata_f2 << u_y_speed_dot;
    Function acadodata_f3;
    acadodata_f3 << x;
    acadodata_f3 << y;
    acadodata_f3 << x_speed;
    acadodata_f3 << y_speed;
    acadodata_f3 << x_turmoil;
    acadodata_f3 << y_turmoil;
    acadodata_f3 << x_turmoil_dot;
    acadodata_f3 << y_turmoil_dot;
    acadodata_f3 << 1.00000000000000000000e+01*sqrt((1.00000000000000004792e-04+1.25000000000000000000e+01*pow(x_turmoil,2.00000000000000000000e+00)+5.00000000000000000000e-01*pow(x_turmoil_dot,2.00000000000000000000e+00)));
    acadodata_f3 << 1.00000000000000000000e+01*sqrt((1.00000000000000004792e-04+1.25000000000000000000e+01*pow(y_turmoil,2.00000000000000000000e+00)+5.00000000000000000000e-01*pow(y_turmoil_dot,2.00000000000000000000e+00)));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r1+sqrt((pow((-obstacles_x1+x),2.00000000000000000000e+00)+pow((-obstacles_y1+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r2+sqrt((pow((-obstacles_x2+x),2.00000000000000000000e+00)+pow((-obstacles_y2+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r3+sqrt((pow((-obstacles_x3+x),2.00000000000000000000e+00)+pow((-obstacles_y3+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r4+sqrt((pow((-obstacles_x4+x),2.00000000000000000000e+00)+pow((-obstacles_y4+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r5+sqrt((pow((-obstacles_x5+x),2.00000000000000000000e+00)+pow((-obstacles_y5+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r6+sqrt((pow((-obstacles_x6+x),2.00000000000000000000e+00)+pow((-obstacles_y6+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r7+sqrt((pow((-obstacles_x7+x),2.00000000000000000000e+00)+pow((-obstacles_y7+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r8+sqrt((pow((-obstacles_x8+x),2.00000000000000000000e+00)+pow((-obstacles_y8+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r9+sqrt((pow((-obstacles_x9+x),2.00000000000000000000e+00)+pow((-obstacles_y9+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-obstacles_r10+sqrt((pow((-obstacles_x10+x),2.00000000000000000000e+00)+pow((-obstacles_y10+y),2.00000000000000000000e+00))))*2.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent1+sqrt((pow((x-x_agent1),2.00000000000000000000e+00)+pow((y-y_agent1),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent2+sqrt((pow((x-x_agent2),2.00000000000000000000e+00)+pow((y-y_agent2),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent3+sqrt((pow((x-x_agent3),2.00000000000000000000e+00)+pow((y-y_agent3),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent4+sqrt((pow((x-x_agent4),2.00000000000000000000e+00)+pow((y-y_agent4),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    acadodata_f3 << 1/(1.00000000000000000000e+00+exp((-5.00000000000000000000e-01-r_agent5+sqrt((pow((x-x_agent5),2.00000000000000000000e+00)+pow((y-y_agent5),2.00000000000000000000e+00))))*3.00000000000000000000e+00));
    DifferentialEquation acadodata_f1;
    acadodata_f1 << dot(x) == x_speed;
    acadodata_f1 << dot(y) == y_speed;
    acadodata_f1 << dot(x_speed) == u_x_speed_dot;
    acadodata_f1 << dot(y_speed) == u_y_speed_dot;
    acadodata_f1 << dot(x_turmoil) == x_turmoil_dot;
    acadodata_f1 << dot(y_turmoil) == y_turmoil_dot;
    acadodata_f1 << dot(x_turmoil_dot) == ((-2.50000000000000000000e+01)*x_turmoil-1.19999999999999995559e+00*x_turmoil_dot+cos((-yaw))*u_x_speed_dot-sin((-yaw))*u_y_speed_dot);
    acadodata_f1 << dot(y_turmoil_dot) == ((-2.50000000000000000000e+01)*y_turmoil-1.19999999999999995559e+00*y_turmoil_dot+cos((-yaw))*u_y_speed_dot+sin((-yaw))*u_x_speed_dot);
    acadodata_f1 << dot(x_agent1) == x_speed_agent1;
    acadodata_f1 << dot(x_agent2) == x_speed_agent2;
    acadodata_f1 << dot(x_agent3) == x_speed_agent3;
    acadodata_f1 << dot(x_agent4) == x_speed_agent4;
    acadodata_f1 << dot(x_agent5) == x_speed_agent5;
    acadodata_f1 << dot(y_agent1) == y_speed_agent1;
    acadodata_f1 << dot(y_agent2) == y_speed_agent2;
    acadodata_f1 << dot(y_agent3) == y_speed_agent3;
    acadodata_f1 << dot(y_agent4) == y_speed_agent4;
    acadodata_f1 << dot(y_agent5) == y_speed_agent5;
    acadodata_f1 << dot(r_agent1) == 1.00000000000000005551e-01;
    acadodata_f1 << dot(r_agent2) == 1.00000000000000005551e-01;
    acadodata_f1 << dot(r_agent3) == 1.00000000000000005551e-01;
    acadodata_f1 << dot(r_agent4) == 1.00000000000000005551e-01;
    acadodata_f1 << dot(r_agent5) == 1.00000000000000005551e-01;

    OCP ocp1(0, 2, 40);
    ocp1.minimizeLSQ(acadodata_M1, acadodata_f2);
    ocp1.minimizeLSQEndTerm(acadodata_M2, acadodata_f3);
    ocp1.subjectTo((-obstacles_r1+sqrt((pow((-obstacles_x1+x),2.00000000000000000000e+00)+pow((-obstacles_y1+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r2+sqrt((pow((-obstacles_x2+x),2.00000000000000000000e+00)+pow((-obstacles_y2+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r3+sqrt((pow((-obstacles_x3+x),2.00000000000000000000e+00)+pow((-obstacles_y3+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r4+sqrt((pow((-obstacles_x4+x),2.00000000000000000000e+00)+pow((-obstacles_y4+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r5+sqrt((pow((-obstacles_x5+x),2.00000000000000000000e+00)+pow((-obstacles_y5+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r6+sqrt((pow((-obstacles_x6+x),2.00000000000000000000e+00)+pow((-obstacles_y6+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r7+sqrt((pow((-obstacles_x7+x),2.00000000000000000000e+00)+pow((-obstacles_y7+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r8+sqrt((pow((-obstacles_x8+x),2.00000000000000000000e+00)+pow((-obstacles_y8+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r9+sqrt((pow((-obstacles_x9+x),2.00000000000000000000e+00)+pow((-obstacles_y9+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-obstacles_r10+sqrt((pow((-obstacles_x10+x),2.00000000000000000000e+00)+pow((-obstacles_y10+y),2.00000000000000000000e+00)))) >= 0.00000000000000000000e+00);
    ocp1.setNOD( 45 );
    ocp1.setModel( acadodata_f1 );


    ocp1.setNU( 2 );
    ocp1.setNP( 0 );
    ocp1.setNOD( 45 );
    OCPexport ExportModule1( ocp1 );
    ExportModule1.set( GENERATE_MATLAB_INTERFACE, 1 );
    uint options_flag;
    options_flag = ExportModule1.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: HESSIAN_APPROXIMATION");
    options_flag = ExportModule1.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: DISCRETIZATION_TYPE");
    options_flag = ExportModule1.set( SPARSE_QP_SOLUTION, FULL_CONDENSING_N2 );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: SPARSE_QP_SOLUTION");
    options_flag = ExportModule1.set( INTEGRATOR_TYPE, INT_IRK_GL6 );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: INTEGRATOR_TYPE");
    options_flag = ExportModule1.set( NUM_INTEGRATOR_STEPS, 40 );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: NUM_INTEGRATOR_STEPS");
    options_flag = ExportModule1.set( QP_SOLVER, QP_QPOASES );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: QP_SOLVER");
    options_flag = ExportModule1.set( HOTSTART_QP, YES );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: HOTSTART_QP");
    options_flag = ExportModule1.set( LEVENBERG_MARQUARDT, 1.000000E-10 );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: LEVENBERG_MARQUARDT");
    options_flag = ExportModule1.set( PRINTLEVEL, DEBUG );
    if(options_flag != 0) mexErrMsgTxt("ACADO export failed when setting the following option: PRINTLEVEL");
    uint export_flag;
    export_flag = ExportModule1.exportCode( "." );
    if(export_flag != 0) mexErrMsgTxt("ACADO export failed because of the above error(s)!");


    clearAllStaticCounters( ); 
 
} 

