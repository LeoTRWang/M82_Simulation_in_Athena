//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file firecracker.cpp
//a simple simulation on supernova feedback in a disk galaxy environment
//included physics
//sukhbold supernova model
//metal and dust chasing
//grackle dust field and cooling
//fix metallicity cooling
//static gravitational potential
//conduction
//fluid particle tracing
//========================================================================================
//WARNING:
//passive scalar and fluid particle tracing is NOT compatible with AMR
//

// C headers
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>


// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <cstdio> 
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <random>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../scalars/scalars.hpp"

// MPI header
//only use in passing particle information
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//Radiation code Grackle header
//including grackle.h in a C++ code requires the extern “C” directive
extern "C"{
	#include "/WORK/sysu_spa_wszhu_2/wangtr/grackle/include/grackle.h"
	//#include <grackle.h>
}
//Froward declaration

void CoolingF(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
int RefinementCondition(MeshBlock *pmb);
Real MyTimeStep(MeshBlock *pmb);
void HydrogenConductioncoeff(
    HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);
Real invert(Real(*func)(Real),Real rl, Real rr, Real ref);
Real lambdafunc(Real temp);
Real nefromt(Real temp);
Real phi_m82(Real x,Real y,Real z);
Real rho_miyamoto(Real x,Real y,Real z,Real a,Real b,Real m);
Real phi_miyamoto(Real x,Real y,Real z,Real a,Real b,Real m);
Real rho_3m(Real x,Real y,Real z,Real e,Real r_hm,Real m);//r_hm means half mass radius, in other comments we will use hm as the abbreviation of half mass
Real phi_3m(Real x,Real y,Real z,Real e,Real r_hm,Real m);
Real phi_king(Real x,Real y,Real z);
Real phi_nfw(Real x,Real y,Real z);
Real phi_m82_with_gas(Real x,Real y,Real z);
Real phi_m82_without_gas(Real x,Real y,Real z);
void srcmask(AthenaArray<Real> &src, int is, int ie, int js, int je, int ks, int ke, const MGCoordinates &coord);

namespace{
	int inject_counter, inject_bo, max_inject,type_inject, cooling, t_dependency, sts_method,
		max_sup_num, turb_flag, nlow, nhigh, turb_cuz, gp_bool, particle_bool, disk_sim, m82_sim, particle_num,
		star_bool, reallocate_count, heating, t_step_delay;
	Real sigma1, sigma2, sigma3, sigma4, p0, rho0, rho_cc, mas_inject, eng_inject, sup_dt, seed1, seed2, gm1, 
		t_amb, substep_t0, substep_t1, substep_dt, dt_floor, threshold, amr_level, k_b, m_h, t_floor,
		dt_inject, t_mainstep, dt_mainstep, gt_limit, expo, eng_turb, f_shear, Z_solar, den_floor, a1, a2, z0,
		particle_output_dt, scale_factor, g, omega_0, m_sb, rho_d_0, rho_h_0, m_gas, e_gas, r_hm_gas,
		rho_0, r_s, a_sd, b_sd, m_sd, stellar_mass_tot, rot_factor, c, mvir, user_cfl_number, c_nfw,
		particle_mass_threshold;
	double initial_redshift;
	code_units my_units;
	chemistry_data *my_grackle_data;
	Real cds[91][12];//cooling rate data set
	Real sds[10000][13];//
	Real sds_t[3000][9];
	Real pds[300][3];
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
	
	
	//==========part 1 loading necessary files==========	
	std::ifstream file_1("../src/coolingrate.dat");
	
	//file>>
	for (int i=0;i<91;i++){
		for (int j=0;j<12;j++){
			file_1>>cds[i][j];
		}
	}
	file_1.close();
	
	std::ifstream file_2("../src/psRealisticSNeData.dat");
	//file>>
	//max supernovae number
	max_sup_num = 10000;
	for (int i=0;i<max_sup_num;i++){
		for (int j=0;j<13;j++){
			file_2>>sds[i][j];
		}
	}
	file_2.close();
	
	
	std::ifstream file_3("../src/ParticleInitLoc.dat");
	//file>>
	particle_num=0;//pin->GetOrAddInteger("problem","particle_num",0);
	for (int i=0;i<particle_num;i++){
		for (int j=0;j<3;j++){
			file_3>>pds[i][j];
		}
	}
	file_3.close();
	
	std::ifstream file_4("../src/TimeSerialSNData.dat");
	//file>>
	for (int i=0;i<3000;i++){
		for (int j=0;j<9;j++){
			file_3>>pds[i][j];
		}
	}
	file_4.close();
	
	//==========part 2 physics constants==========
	k_b=1.380649e-16;
	m_h=1.66053904e-24; 
	Z_solar=0.013;
	g=6.6743e-8;
	c=2.997e10;
	particle_mass_threshold=5.0e4*2.0e33;
	
	//==========part 3 reading parameters==========
	//Reading parameters from athinput & defining parameters
	sigma1      =pin->GetReal("problem","sigma_eng_dis");
	//sigma1      =3.0e21/256.0;
	sigma2      =pin->GetReal("problem","sigma_sup_dis");
	sigma3      =pin->GetReal("problem","sigma_fluc");
	//sigma4      =pin->GetReal("problem","r_cloud");
	//t_amb       =pin->GetReal("problem","t_amb");
	//rho0        =pin->GetReal("problem","rho0")*m_h*(4/(1+3*0.91));
	//rho_cc      =pin->GetReal("problem","rho_cloud_center")*m_h*(4/(1+3*0.91));
	//mas_inject  =pin->GetReal("problem","mass_inject")*1.989e33;
	//eng_inject  =pin->GetReal("problem","eng_inject");
	sup_dt      =pin->GetReal("problem","sup_dt");
	seed1       =pin->GetReal("problem","random_seed1");
	seed2       =pin->GetReal("problem","random_seed2");
	t_floor     =pin->GetReal("problem","t_floor");
	dt_floor    =pin->GetReal("problem","dt_floor");
	//type_inject =pin->GetInteger("problem","injection_type");
	cooling     =pin->GetInteger("problem","cooling");
	//t_dependency=pin->GetInteger("problem","inject_t_dependency");
	gt_limit    =pin->GetOrAddReal("problem","gt_limit",-1);
	//max_inject  =pin->GetOrAddInteger("problem","max_inject",50);
	//sts_method  =pin->GetOrAddInteger("problem","sts_method",0);
	
	//for initial turbulence
	turb_flag   =pin->GetOrAddInteger("problem","turb_flag",0);
	//turb_flag   =0;
	turb_cuz    =pin->GetInteger("problem","turb_cuz");
	//turb_cuz   =0;
	nlow        =pin->GetInteger("turbulence","nlow");
	nhigh       =pin->GetInteger("turbulence","nhigh");
	expo        =pin->GetReal("turbulence","expo");
	f_shear     =pin->GetReal("turbulence","f_shear");
	eng_turb    =pin->GetReal("turbulence","dedt");
	den_floor   =pin->GetReal("hydro","dfloor");
	
	user_cfl_number   =pin->GetReal("time","cfl_number");
	
	//for star/cloudlet particle
	particle_bool=1;
	
	//for heating
	stellar_mass_tot = 0.0;
	
	//for disk environment
	disk_sim   = 0;//pin->GetOrAddInteger("problem","disk_sim",0);
	m82_sim    = 1;//pin->GetOrAddInteger("problem","m82_sim",0);
	gp_bool=0;
	if ((disk_sim==1)||(m82_sim==1)){
		gp_bool=1;
	}
	
	reallocate_count=0;
	
	dt_inject=0;
	/*
	if (t_dependency==1){
		dt_inject=pin->GetReal("problem","dt_inject");
	}
	*/
	inject_counter  =0;
	t_mainstep      =0;
	dt_mainstep     =0;
	substep_t0=-2.0;
	substep_t1=-1.0;
	substep_dt=0.0;
	inject_bo=0;
	t_step_delay=0;
	//rho_amb=mas_inject*9.786805/pow(sigma1,3);
	
	//==========part 4 initialize problem-specific parameters==========
	if (disk_sim==1){
		a1=1.42e-3*1.0e3*3.087e18/SQR(1.0e6*3.157e7);
		a2=5.49e-4/SQR(1.0e6*3.157e7);
		z0=0.18*1.0e3*3.087e18;
		p0=rho0/(m_h*(4/(1+3*0.91)))*k_b*t_amb;
	}
	if (m82_sim==1){
		//stellar fraction
		m_sb=6.0e8*2.0e33*(0.4/1.4);
		omega_0 = 450*3.0e18/1.82;
		m_sd=6.0e8*2.0e33*(1.0/1.4);
		a_sd=650*3.0e18/1.61;
		b_sd=0.3*650*3.0e18/1.61;
		//gas fraction
		e_gas=0.3;
		m_gas=3.5e9*2.0e33;
		r_hm_gas=1000.0*3.0e18;
		rho_d_0 = 80.0*m_h;
		rho_h_0 = 2.0e-3*m_h;
		//DM halo
		mvir=1.7027535489373234e+44;
		Real r_200=53*1000*3.0e18;
		c_nfw=pow(10,(1.071-0.098*(std::log10(mvir/(2.0e33/0.7))-12)));
		r_s=r_200/c_nfw;
		rho_0= mvir/(4*PI*pow(r_s,3)*(std::log(1+c_nfw)-c_nfw/(1+c_nfw)));
		//source mask funciton
		EnrollUserMGGravitySourceMaskFunction(srcmask);
		rot_factor=0.9;
	}
	
	//==========part 5 enroll functions==========
	EnrollUserExplicitSourceFunction(CoolingF);  
	EnrollUserTimeStepFunction(MyTimeStep);
	EnrollConductionCoefficient(HydrogenConductioncoeff);
	if (adaptive) {
		//std::cout<<"AMR enrolled"<<std::endl;
		threshold = pin->GetReal("problem", "thr");
		EnrollUserRefinementCondition(RefinementCondition);
		amr_level = pin->GetReal("mesh", "numlevel");
	}
	//initialize cooling data set
	if ((cooling==2)&&(NSCALARS<=0)){
		std::stringstream nan_msg;
		nan_msg<<"Fatal Error: Grackle can not run without metal(scalar) data"<<std::endl;
		ATHENA_ERROR(nan_msg);
	}
	
	//==========part 6 allocate user mesh data==========
	//mpi output part
	//bool(rank) of each node that determines whether output happens is stored here
	AllocateIntUserMeshDataField(2);
	iuser_mesh_data[0].NewAthenaArray(1);
	iuser_mesh_data[0](0)=0;
	
	iuser_mesh_data[1].NewAthenaArray(1);
	iuser_mesh_data[1](0)=particle_num-1;
	
	//==========part 7 initialize cooling==========
	if (cooling==2){
		//initialize grackle
		//set initial redshift to local universe
		initial_redshift = 0.0;
		
		//enabling output
		grackle_verbose = 1;
		
		//unit conversion
		//these are conversions from code units to cgs.
		
		my_units.comoving_coordinates = 0; // 1 if cosmological sim, 0 if not
		my_units.density_units = 1.0;
		my_units.length_units = 1.0;
		my_units.time_units = 1.0;
		my_units.velocity_units = my_units.length_units / my_units.time_units;
		my_units.a_units = 1.0; // units for the expansion factor
		
		// Set expansion factor to 1 for non-cosmological simulation.
		my_units.a_value = 1. / (1. + initial_redshift) / my_units.a_units;
		
		//create chemistry object 
		my_grackle_data = new chemistry_data;
		if (set_default_chemistry_parameters(my_grackle_data) == 0) {
			std::stringstream nan_msg;
			nan_msg<<"Fatal Error: Error in set_default_chemistry_parameters"<<std::endl;
			ATHENA_ERROR(nan_msg);
		}
		
		//setting grackle chemistry parameters
		//to use the complete setup including self-shielding radiative cooling, self-shielding uvb heating, one
		//must set uvb switch off but leave user-defined radiation field on and set pe heating to 2
		//always keep dust chemistry off for the reason that it cause nan to occur in cooling time
		grackle_data->use_grackle = 1;                                                 // chemistry on
		grackle_data->use_isrf_field = 1;                                              // interstellar radiation field set to user-defined
		grackle_data->photoelectric_heating = 2;                                       // nautural gas heating caused by FUV photon
		grackle_data->with_radiative_cooling = 1;                                      // cooling on
		grackle_data->primordial_chemistry = 0;                                        // tabulated cooling function only
		grackle_data->dust_chemistry = 0;
		grackle_data->metal_cooling = 1;                                               // metal cooling on
		grackle_data->UVbackground = 1;                                                // UV background on
		grackle_data->self_shielding_method = 3;                                       //self-shielding on
		grackle_data->grackle_data_file = "/WORK/sysu_spa_wszhu_2/wangtr/grackle/grackle_data_files/input/CloudyData_UVB=HM2012_shielded.h5";
		
		//initialize the chemistry object
		if (initialize_chemistry_data(&my_units) == 0) {
			std::stringstream nan_msg;
			nan_msg<<"Fatal Error: Error in initialize_chemistry_data"<<std::endl;
			ATHENA_ERROR(nan_msg);
		}
		
		cooling=0;
		heating=0;
	}
	
	SetFourPiG(4*PI*g);
	SetGravityThreshold(0.0);
	
	particle_output_dt=0.0;
	star_bool=0;
	if (particle_bool==1){
		particle_output_dt=pin->GetReal("problem","particle_output_dt");
		particle_bool=0;
	}
	
	//==========part 8 initialize turbulence==========
	
	if (turb_cuz==1){
		Real unscaled_e=0.0;
		/*
		if ((disk_sim==0)&&(m82_sim==0)){
			Real zone_scale=2.0*3.085678e20;
			for (int k=0; k<64; k++) {
				for (int j=0; j<64; j++) {
					for (int i=0; i<=64; i++) {
						Real v=0.0;
						Real theta=2.5238;
						Real delta=5.6241;
						Real x = i*(zone_scale/64.0)-0.5*zone_scale+(0.5*zone_scale/64.0);//cell center coordinate
						Real y = j*(zone_scale/64.0)-0.5*zone_scale+(0.5*zone_scale/64.0);
						Real z = k*(zone_scale/64.0)-0.5*zone_scale+(0.5*zone_scale/64.0);
						Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
						Real den = rho0 + (rho_cc-rho0)*exp(-(rad*rad)/(2*sigma4*sigma4));
						for (int nx=0; nx<=nhigh; nx++){
							
							//theta=fmod((theta*theta+delta),2.0*M_PI);
							//Real phx=theta;
							Real kx=2.0*M_PI*nx/zone_scale;
							
							for (int ny=0; ny<=nhigh; ny++){
								
								//theta=fmod((theta*theta+delta),2.0*M_PI);
								//Real phy=theta;
								Real ky=2.0*M_PI*ny/zone_scale;
								
								for (int nz=0; nz<=nhigh; nz++){
									//theta=fmod((theta*theta+delta),2.0*M_PI);
									//Real phz=theta;
									Real kz=2.0*M_PI*nz/zone_scale;
									
									Real nmag=std::sqrt(SQR(nx)+SQR(ny)+SQR(nz));
									Real kmag=std::sqrt(SQR(kx)+SQR(ky)+SQR(kz));
									if ((nmag>=nlow)&&(nmag<=nhigh)){
										v+=(1.0/pow(kmag,1.0+expo/2.0))*cos(kx*x+ky*y+kz*z+theta);//taking only the real part
									}
								}
							}
						}
						unscaled_e+=3*0.5*den*pow(v,2.0)*pow(zone_scale/64.0,3.0);
					}
				}
			}
		}
		if (disk_sim==1){
			Real zone_scale=4.0*3.085678e20;
			for (int k=0; k<64; k++) {
				for (int j=0; j<64; j++) {
					for (int i=0; i<=128; i++) {
						Real v=0.0;
						Real theta=2.5238;
						Real delta=5.6241;
						Real x = i*(zone_scale/64.0)-0.5*zone_scale+(0.5*zone_scale/64.0);//cell center coordinate
						Real y = j*(zone_scale/64.0)-0.5*zone_scale+(0.5*zone_scale/64.0);
						Real z = k*(zone_scale/64.0)-0.5*zone_scale+(0.5*zone_scale/64.0);
						Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
						Real den=rho0*exp(-a1*rho0*(std::sqrt(SQR(z)+SQR(z0))-z0)/p0-a2*rho0*SQR(z)/(2*p0));
						for (int nx=0; nx<=nhigh; nx++){
							
							Real kx=2.0*M_PI*nx/zone_scale;
							
							for (int ny=0; ny<=nhigh; ny++){
								
								Real ky=2.0*M_PI*ny/zone_scale;
								
								for (int nz=0; nz<=nhigh; nz++){
									Real kz=2.0*M_PI*nz/zone_scale;
									
									Real nmag=std::sqrt(SQR(nx)+SQR(ny)+SQR(nz));
									Real kmag=std::sqrt(SQR(kx)+SQR(ky)+SQR(kz));
									if ((nmag>=nlow)&&(nmag<=nhigh)){
										v+=(1.0/pow(kmag,1.0+expo/2.0))*cos(kx*x+ky*y+kz*z+theta);//taking only the real part
									}
									
									theta+=delta;
								}
							}
						}
						unscaled_e+=3*0.5*den*pow(v,2.0)*pow(zone_scale/64.0,3.0);
					}
				}
			}
		}
		*/
		/*
		if (m82_sim==1){
			//for m82 simulations
			Real t_halo = 6.7e6;
			//Real t_disk = 4.0e5;
			//Real c_s_h = std::sqrt(gamma_cahce*k_b*t_halo/(m_h));
			Real c_s_h = 3.0e7;
			//Real c_s_d = std::sqrt(gamma_cahce*k_b*t_disk/(m_h));
			Real c_s_d = 5.0e6;
			
			Real zone_scale=6.0e21;
			for (int k=0; k<128; k++) {
				for (int j=0; j<128; j++) {
					for (int i=0; i<=128; i++) {
						std::mt19937 mt(20);
						Real v=0.0;
						//Real theta=2.5238;
						//Real delta=5.6241;
						Real x = i*(zone_scale/128.0)-0.5*zone_scale+(0.5*zone_scale/128.0);//cell center coordinate
						Real y = j*(zone_scale/128.0)-0.5*zone_scale+(0.5*zone_scale/128.0);
						Real z = k*(zone_scale/128.0)-0.5*zone_scale+(0.5*zone_scale/128.0);
						Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
						
						//rotational support factor
						Real e_rot = rot_factor*exp(-std::abs(z)/(5.0*1000.0*3.0e18));
						
						Real rho_halo = rho_h_0*exp(-(phi_m82_with_gas(x,y,z)-SQR(e_rot)*phi_m82_with_gas(x,y,0.0)-(1-SQR(e_rot))*phi_m82_with_gas(0.0,0.0,0.0))/SQR(c_s_h));
						
						Real rho_disk = rho_d_0*exp(-(phi_m82_with_gas(x,y,z)-SQR(e_rot)*phi_m82_with_gas(x,y,0.0)-(1-SQR(e_rot))*phi_m82_with_gas(0.0,0.0,0.0))/SQR(c_s_d));
						
						Real den = rho_disk;
						//Real den = rho_3m(x,y,z,e_gas,r_hm_gas,m_gas);
						
						for (int nx=-nhigh/2; nx<=nhigh/2; nx++){
							
							Real kx=4.0*M_PI*nx/zone_scale;
							
							for (int ny=-nhigh/2; ny<=nhigh/2; ny++){
								
								Real ky=4.0*M_PI*ny/zone_scale;
								
								for (int nz=-nhigh/2; nz<=nhigh/2; nz++){
									Real kz=4.0*M_PI*nz/zone_scale;
									
									//theta+=delta;
									Real theta=2*M_PI*(Real)(mt())/(Real)(mt.max());
									
									Real nmag=std::sqrt(SQR(nx)+SQR(ny)+SQR(nz));
									Real kmag=std::sqrt(SQR(kx)+SQR(ky)+SQR(kz));
									if ((nmag>=nlow)&&(nmag<=nhigh)){
										v+=(1.0/pow(kmag,1.0+expo/2.0))*cos(kx*x+ky*y+kz*z+theta);//taking only the real part
									}
								}
							}
						}
						unscaled_e+=3*0.5*den*pow(v,2.0)*pow(zone_scale/128.0,3.0);
					}
				}
			}
		}
		*/
		scale_factor = std::sqrt(eng_turb/unscaled_e);
		
	}
	return;
}

//Useful function defined by user
//Inverse erf and random function are included here, which later decides the coordination
//of each explosion's center

//invert function
//using dichotomy method, with worst precision=abs(rl-rr)*(1/2)^50
Real invert(Real(*func)(Real),Real r_l, Real r_r, Real ref){
	Real rl=r_l;
	Real rr=r_r;
	Real result=(rl+rr)/2;
	//cout<<func(rl)<<" "<<func(rr)<<" "<<ref<<endl; 
	if ((func(rl)-ref)*(func(result)-ref)*(func(rr)-ref)*(func(result)-ref)>0){
		std::cout<<"Invertion error"<<std::endl;
	}
	for (int ii=0;ii<=50;ii++){
		if ((func(rl)-ref)*(func(result)-ref)<0 &&
			(func(rr)-ref)*(func(result)-ref)>0){
			rr=result;
		} else if ((func(rl)-ref)*(func(result)-ref)>0 &&
		    (func(rr)-ref)*(func(result)-ref)<0){
		    rl=result;
		} else {
			//do nothing
		}
	   	result=(rr+rl)/2;
	}
	return result;
}


//==========sedov part==========

//three functions from sedov solution that decided v_r c_s^2 rho
Real xi5fromu(Real u){
	Real gamma_xif =1.666667;
	Real nu_1=-(13*gamma_xif*gamma_xif-7*gamma_xif+12)/((2*gamma_xif+1)*(3*gamma_xif-1));
	Real nu_2=5*(gamma_xif-1)/(2*gamma_xif+1);
	Real result=pow((5-(3*gamma_xif-1)*u)*(gamma_xif+1)/(7-gamma_xif),nu_1)
				*pow((gamma_xif*u-1)*(gamma_xif+1)/(gamma_xif-1),nu_2)
				/pow((gamma_xif+1)*u*0.5,2);
	return result;
}

Real ufunc(Real xi){
	Real gamma_uf  =1.666667;
	Real u_l=1/gamma_uf;
	Real u_r=2/(gamma_uf+1);
	return invert(*xi5fromu,u_l,u_r,pow(xi,5));
}

Real gfunc(Real u){
	Real gamma_gf  =1.666667;
	Real gamma_di=(gamma_gf+1)/(gamma_gf-1);
	Real nu_1=-(13*gamma_gf*gamma_gf-7*gamma_gf+12)/((2*gamma_gf+1)*(3*gamma_gf-1));
	Real nu_3=3/(2*gamma_gf+1);
	Real nu_4=-nu_1/(2-gamma_gf);
	Real nu_5=-2/(2-gamma_gf);
	Real result=gamma_di
				*pow(gamma_di*(gamma_gf*u-1),nu_3)
				*pow((5-(3*gamma_gf-1)*u)*(gamma_gf+1)/(7-gamma_gf),nu_4)
				*pow(gamma_di*(1-u),nu_5);
	if (result<0.0005){
		result = 0.0005;
	}
	return result;
}

Real zfunc(Real u){
	Real gamma_zf  =1.666667;
	Real result=0.5*gamma_zf*(gamma_zf-1)*u*u*(1-u)/(gamma_zf*u-1);
	return result;
}

//==========EOS part==========

Real x_(Real rho, Real T) {
  return 2./(1.+std::sqrt(1.+4.*std::exp(1./T-1.5*std::log(T)+std::log(rho))));
}

Real lambdafunc(Real temp){
	Real result=0;
	if (temp>1.0e4){
		int l=0;
		int r=1;
		Real logt=log10(temp);
		if (logt>cds[90][0]){
			return pow(10,cds[90][5]);
		}
		else if (logt<4){
			return 0.0;
		}	
		for (int i=1;i<90;i++){
			if (logt>cds[i][0]){
				l=i;
				r=i+1;
			}
		}
		result=pow(10,cds[l][5]+(logt-cds[l][0])*(cds[r][5]-cds[l][5])/(cds[r][0]-cds[l][0]));
	}
	else{
		result=1.478977*2.0e-26*(1.0e7*exp(-114800/(temp+1000))+0.01*sqrt(temp)*exp(-92/(temp)));
	}
	return result;
}

Real pfromt(Real temp){
	Real logt=log10(temp);
	
	int l=0;
	int r=1;
	for (int i=1;i<90;i++){
		if (logt>cds[i][0]){
			l=i;
			r=i+1;
		}
	}
	Real ne=cds[l][1]+(logt-cds[l][0])*(cds[r][1]-cds[l][1])/(cds[r][0]-cds[l][0]);
	if (logt>cds[90][0]) ne =cds[90][1];
	if (logt<cds[0][0])  ne =exp(logt)*cds[0][1]/exp(cds[0][0]);
	
	Real xx=ne/cds[90][1];
	Real mu=4/(1+3*0.91+4*0.91*xx);
	return k_b*temp/(mu*m_h);
}

Real nefromt(Real temp){
	Real ne=0;
	Real logt=log10(temp);
				
	int l=0;
	int r=1;
	for (int i=1;i<90;i++){
		if (logt>cds[i][0]){
			l=i;
			r=i+1;
		}
	}
	ne=cds[l][1]+(logt-cds[l][0])*(cds[r][1]-cds[l][1])/(cds[r][0]-cds[l][0]);
	if (logt>cds[90][0]) ne =cds[90][1];
	if (logt<cds[0][0])  ne =exp(logt)*cds[0][1]/exp(cds[0][0]);
	
	return ne;
}

//==========psuedo-random part==========

//psuedo-random function
Real prand(Real counter){
	Real result=fmod(counter*seed1+seed2,2)-1;
	return result;
}

//injection-location function
Real sup_dis(Real xx,Real sigma_input){
	Real coef=1/(sqrt(2)*sigma_input);
	Real a0=-10*sigma_input;
	Real b0=10*sigma_input;
	//Real result=(a0+b0)/2;
	Real result=invert(erf,coef*a0,coef*b0,xx)/coef;
	return result;
}

//==========gravity solver part==========



Real rtof(Real rr,Real dx_cell){
	//inverse-square law could cause numerical instability when particles are too close apart
	//or cause unphysical increase in gas energy
	//a simple smoothing scheme is adopted
	//by simply assuming that mass of each sink particle
	Real result=0.0;
	Real lim=5.0;
	if (rr>lim*dx_cell){
		result=1/(SQR(rr));
	}
	else{
		//result=rr;
		//result=(1/(SQR(lim*dx_cell)))*(rr/lim*dx_cell);
		result=0.0;
	}
	return result;
}

//==========m82 gravity potential part==========
Real rho_miyamoto(Real x,Real y,Real z,Real a,Real b,Real m){
	Real rr=std::sqrt(SQR(x)+SQR(y));
	Real result=(SQR(b)*m/(4*PI))*
			(a*SQR(rr)+(a+3*std::sqrt(SQR(z)+SQR(b)))*SQR(a+std::sqrt(SQR(z)+SQR(b))))/
			(pow(SQR(rr)+SQR(a+std::sqrt(SQR(z)+SQR(b))),2.5)*pow(SQR(z)+SQR(b),1.5));
	return result;
}

Real phi_miyamoto(Real x,Real y,Real z,Real a,Real b,Real m){
	Real result=-g*m     /std::sqrt(SQR(x)+SQR(y)+SQR(a        +std::sqrt(SQR(z)+SQR(b        )))+0.1);
	//          -g*m_disk/std::sqrt(SQR(x)+SQR(y)+SQR(m82para_a+std::sqrt(SQR(z)+SQR(m82para_b)))+0.1);
	return result;
}

Real rho_3m(Real x,Real y,Real z,Real e,Real r_hm,Real m){//r_hm means half mass radius, in other comments we will use hm as the abbreviation of half mass
	if (std::abs(e-0.3)>0.01){
		return -1.0;
	}
	Real r_halfmass=3.909114273641178;//edge-on half mass radius
	Real a_list[3]={2.6712658568789247, 4.8327240447497735, 2.3936222151483677};
	Real m_list[3]={0.07792599000689583, -0.4495029985958817, 1.3715770085889858};
	Real result=0.0;
	for (int i =0;i<3;i++){
		Real b_i=1.0*r_hm/r_halfmass;
		Real a_i=a_list[i]*b_i;
		Real m_i=m_list[i]*m;
		result+=rho_miyamoto(x,y,z,a_i,b_i,m_i);
	}
	return result;
}

Real phi_3m(Real x,Real y,Real z,Real e,Real r_hm,Real m){
	if (std::abs(e-0.3)>0.01){
		return -1.0;
	}
	Real r_halfmass=3.909114273641178;//edge-on half mass radius
	Real a_list[3]={2.6712658568789247, 4.8327240447497735, 2.3936222151483677};
	Real m_list[3]={0.07792599000689583, -0.4495029985958817, 1.3715770085889858};
	Real result=0.0;
	for (int i =0;i<3;i++){
		Real b_i=1.0*r_hm/r_halfmass;
		Real a_i=a_list[i]*b_i;
		Real m_i=m_list[i]*m;
		result+=phi_miyamoto(x,y,z,a_i,b_i,m_i);
	}
	return result;
}

Real phi_king(Real x,Real y,Real z){
	Real rad=std::sqrt(SQR(x)+SQR(y)+SQR(z));
	return (-g*m_sb/omega_0)*(log((rad+1.0e10)/omega_0+std::sqrt(1+SQR(rad+1.0e10)/SQR(omega_0)))/((rad+1.0e10)/omega_0));
}

Real phi_nfw(Real x,Real y,Real z){
	Real rad=std::sqrt(SQR(x)+SQR(y)+SQR(z))+1.0e9;
	//Real result=-4*PI*g*rho_0*(pow(r_s,3)/rad)*std::log(1+rad/r_s);
	
	/*
	r_s = rs
    
    r_c=500.0*3.0e18
    r_p=0.6*r_c
    
    x=r_p/r_s
    m_rc=mvir*gc*(np.log(1+x)-x/(1+x))
    rho_rc=rho0/(x*(1+x)**2)
    #print(rho_rc)
    m_replace=np.pi*(4.0/3.0)*rho_rc*r_p**3
    phi_0=-4*np.pi*g*rho0*(r_s**3/r_p)*np.log(1+r_p/r_s)-(-g*m_rc/r_p)+(-g*m_replace/r_p)-0.5*g*(4/3)*np.pi*rho_rc*r_p**2
    if (rr<r_p):
        return phi_0+0.5*g*(4/3)*np.pi*rho_rc*rr**2
    else:
        return -4*np.pi*g*rho0*(r_s**3/rr)*np.log(1+rr/r_s)-(-g*m_rc/rr)+(-g*m_replace/rr)
	*/
	
	Real r_p = 300.0*3.0e18;
	Real gc=1/(std::log(1+c_nfw)-c_nfw/(1+c_nfw));
	Real xx=r_p/r_s;
	Real rho_rc=rho_0/(xx*SQR(1+xx));
	Real m_rc=mvir*gc*(std::log(1+xx)-xx/(1+xx));
	Real m_replace=PI*(4.0/3.0)*rho_rc*pow(r_p,3);
	Real result=0.0;
	Real phi_0=-4*PI*g*rho_0*(pow(r_s,3)/r_p)*std::log(1+r_p/r_s)-(-g*m_rc/r_p)+(-g*m_replace/r_p)-0.5*g*(4.0/3.0)*PI*rho_rc*SQR(r_p);
	if (rad<r_p){
		result=phi_0+0.5*g*(4.0/3.0)*PI*rho_rc*SQR(rad);
	}
	else{
		result=-4*PI*g*rho_0*(pow(r_s,3)/rad)*std::log(1+rad/r_s)-(-g*m_rc/rad)+(-g*m_replace/rad);
	}
	
	
	/*
	Real result=-4*PI*g*rho_0*(pow(r_s,3)/rad)*std::log(1+rad/r_s);
	if (rad<100.0*3.0e18){
		Real smooth_a=(-4*PI*g*rho_0*(pow(r_s,3)/(101.0*3.0e18))*std::log(1+(101.0*3.0e18)/r_s)+4*PI*g*rho_0*(pow(r_s,3)/(99.0*3.0e18))*std::log(1+(99.0*3.0e18)/r_s))/(2.0*3.0e18*2.0*100.0*3.0e18);
		Real smooth_b=-4*PI*g*rho_0*(pow(r_s,3)/(100.0*3.0e18))*std::log(1+(100.0*3.0e18)/r_s)-smooth_a*SQR(100.0*3.0e18);
		result=SQR(rad)*smooth_a+smooth_b;
	}
	*/
	return result;
}

Real phi_m82_with_gas(Real x,Real y,Real z){
	//          bulge           DM halo        stellar disk                       gas(mainly HI)
	Real result=phi_king(x,y,z)+phi_nfw(x,y,z)+phi_miyamoto(x,y,z,a_sd,b_sd,m_sd)+phi_3m(x,y,z,e_gas,r_hm_gas,m_gas);
	
	//          bulge           DM halo        stellar disk                       gas(mainly HI)
	//Real result=phi_3m(x,y,z,e_gas,r_hm_gas,m_gas);
	return result;
}

Real phi_m82_without_gas(Real x,Real y,Real z){
	//          bulge           DM halo        stellar disk                       gas(mainly HI)(not included)
	Real result=phi_king(x,y,z)+phi_nfw(x,y,z)+phi_miyamoto(x,y,z,a_sd,b_sd,m_sd);
	
	//          bulge           DM halo        stellar disk                       gas(mainly HI)(not included)
	//Real result=0.0;
	return result;
}

Real rtov(Real x,Real y,Real z){
	Real rad=std::sqrt(SQR(x)+SQR(y));
	if (rad==0.0){
		return 0.0;
	}
	Real dphidr=(phi_m82_with_gas(rad*1.01,0.0,z)-phi_m82_with_gas(rad*0.99,0.0,z))/(rad*0.02);
	Real result=std::sqrt(rad*dphidr);
	return result;
}

//==========enrolled functions==========
//time step function
Real MyTimeStep(MeshBlock *pmb){
	Real dt_min=dt_floor;
	
	if (t_step_delay==0){
		dt_min=20*dt_floor;
	}
	
	if (cooling==1){
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					
					Real den=pmb->phydro->u(IDN,k,j,i);
					Real ine=pmb->phydro->u(IEN,k,j,i)-0.5*(pmb->phydro->u(IM1,k,j,i)*pmb->phydro->u(IM1,k,j,i)
						+pmb->phydro->u(IM2,k,j,i)*pmb->phydro->u(IM2,k,j,i)
						+pmb->phydro->u(IM3,k,j,i)*pmb->phydro->u(IM3,k,j,i))/den;
					//calculate local temp(K)
					Real real_temp=(ine/den)*m_h/(1.5*k_b);
					
					//lower limit of temp with which x_3~0
					//mu with 1 M_solar goes as
					Real mu=4/(1+3*0.91);
					
					if ((real_temp>t_floor)&&(den/(mu*m_h)>0.1)){
						
						//Real cooling_coef=(den/1.4)*5e-33*pow(real_temp,1.5)/1.380649e-16*3.397278e14;
						Real cooling_coef=lambdafunc(real_temp)*pow(den/(mu*m_h),2);
						
						if (dt_min>ine*0.1/cooling_coef){
							dt_min=ine*0.1/cooling_coef;
						}
					}
					
				}
			}
		}
		//if (dt_min<0.001*dt_floor){
		//	dt_min=0.001*dt_floor;
	}
	
	if (cooling==2){
		
		//create struct for storing grackle field data
		//grackle_field_data my_fields;
		grackle_field_data my_fields;
		int cellnum = (pmb->ke-pmb->ks+1)*(pmb->je-pmb->js+1)*(pmb->ie-pmb->is+1);
		my_fields.grid_rank = 3;
		my_fields.grid_dimension = new int[3];
		my_fields.grid_start = new int[3];
		my_fields.grid_end = new int[3];
		
		//int grid_dimension[3] = {pmb->ke-pmb->ks+1, pmb->je-pmb->js+1, pmb->ie-pmb->is+1};
		//int grid_start[3] = {0, 0, 0};
		//int grid_end[3] = {pmb->ke-pmb->ks, pmb->je-pmb->js, pmb->ie-pmb->is};
		//int grid_dimension[3] = {pmb->ke-pmb->ks+1, pmb->je-pmb->js+1, pmb->ie-pmb->is+1};
		//int grid_start[3] = {0, 0, 0};
		//int grid_end[3] = {pmb->ke-pmb->ks, pmb->je-pmb->js, pmb->ie-pmb->is};
		my_fields.grid_dimension[0]=pmb->ke-pmb->ks+1;
		my_fields.grid_dimension[1]=pmb->je-pmb->js+1;
		my_fields.grid_dimension[2]=pmb->ie-pmb->is+1;
		my_fields.grid_start[0]=0;
		my_fields.grid_start[1]=0;
		my_fields.grid_start[2]=0;
		my_fields.grid_end[0]=pmb->ke-pmb->ks;
		my_fields.grid_end[1]=pmb->je-pmb->js;
		my_fields.grid_end[2]=pmb->ie-pmb->is;
		
		my_fields.grid_dx = 0.0;
		//creating cell data grips for grackle
		//since Athena++ does not provide direct path to density/internal energy/velocity field to which a pointer can point, we have to create our own arrays
		//but the major drawback of such method is that it is memory-consuming and time-consuming
		my_fields.density         = new gr_float[cellnum];
		my_fields.internal_energy = new gr_float[cellnum];
		my_fields.x_velocity      = new gr_float[cellnum];
		my_fields.y_velocity      = new gr_float[cellnum];
		my_fields.z_velocity      = new gr_float[cellnum];
		my_fields.metal_density   = new gr_float[cellnum];
		
		// for primordial_chemistry >= 1
		my_fields.HI_density      = new gr_float[cellnum];
		my_fields.HII_density     = new gr_float[cellnum];
		my_fields.HeI_density     = new gr_float[cellnum];
		my_fields.HeII_density    = new gr_float[cellnum];
		my_fields.HeIII_density   = new gr_float[cellnum];
		my_fields.e_density       = new gr_float[cellnum];
		// for primordial_chemistry >= 2
		my_fields.HM_density      = new gr_float[cellnum];
		my_fields.H2I_density     = new gr_float[cellnum];
		my_fields.H2II_density    = new gr_float[cellnum];
		// for primordial_chemistry >= 3
		my_fields.DI_density      = new gr_float[cellnum];
		my_fields.DII_density     = new gr_float[cellnum];
		my_fields.HDI_density     = new gr_float[cellnum];
		
		// volumetric heating rate (provide in units [erg s^-1 cm^-3])
		my_fields.volumetric_heating_rate = new gr_float[cellnum];
		// specific heating rate (provide in units [egs s^-1 g^-1]
		my_fields.specific_heating_rate = new gr_float[cellnum];
		
		// radiative transfer ionization / dissociation rate fields (provided in units of [1/s])
		my_fields.RT_HI_ionization_rate = new gr_float[cellnum];
		my_fields.RT_HeI_ionization_rate = new gr_float[cellnum];
		my_fields.RT_HeII_ionization_rate = new gr_float[cellnum];
		my_fields.RT_H2_dissociation_rate = new gr_float[cellnum];
		// radiative transfer heating rate (provide in units [erg s^-1 cm^-3])
		my_fields.RT_heating_rate = new gr_float[cellnum];
		
		// interstellar radiation field strength
		
		my_fields.isrf_habing = new gr_float[cellnum];
		
		//unused parameters
		my_fields.dust_density = new gr_float[cellnum];
		my_fields.H2_self_shielding_length = new gr_float[cellnum];
		
		//writing data into arrays declared above
		int n=0;
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real mu=4/(1+3*0.91);
					Real den=pmb->phydro->u(IDN,k,j,i);
					Real ine=pmb->phydro->u(IEN,k,j,i)-0.5*pmb->phydro->u(IDN,k,j,i)*(pmb->phydro->w(IVX,k,j,i)*pmb->phydro->w(IVX,k,j,i)+
																	pmb->phydro->w(IVY,k,j,i)*pmb->phydro->w(IVY,k,j,i)+
																	pmb->phydro->w(IVZ,k,j,i)*pmb->phydro->w(IVZ,k,j,i));
					Real real_temp=(ine/den)*m_h/(1.5*k_b);
					
					if (pmb->phydro->u(IDN,k,j,i)<=1.0e3*m_h){
						my_fields.density[n]         = pmb->phydro->u(IDN,k,j,i);
						my_fields.internal_energy[n] = (pmb->phydro->u(IEN,k,j,i)
														-0.5*pmb->phydro->u(IDN,k,j,i)*(pmb->phydro->w(IVX,k,j,i)*pmb->phydro->w(IVX,k,j,i)+
																			pmb->phydro->w(IVY,k,j,i)*pmb->phydro->w(IVY,k,j,i)+
																			pmb->phydro->w(IVZ,k,j,i)*pmb->phydro->w(IVZ,k,j,i)))/pmb->phydro->u(IDN,k,j,i);
					}
					else{
						my_fields.density[n]         = 1.0e3*m_h;
						my_fields.internal_energy[n] = (1.0e3*m_h/pmb->phydro->u(IDN,k,j,i))*(pmb->phydro->u(IEN,k,j,i)
														-0.5*pmb->phydro->u(IDN,k,j,i)*(pmb->phydro->w(IVX,k,j,i)*pmb->phydro->w(IVX,k,j,i)+
																			pmb->phydro->w(IVY,k,j,i)*pmb->phydro->w(IVY,k,j,i)+
																			pmb->phydro->w(IVZ,k,j,i)*pmb->phydro->w(IVZ,k,j,i)))/pmb->phydro->u(IDN,k,j,i);
					}
					my_fields.x_velocity[n]      = pmb->phydro->w(IVX,k,j,i);
					my_fields.y_velocity[n]      = pmb->phydro->w(IVY,k,j,i);
					my_fields.z_velocity[n]      = pmb->phydro->w(IVZ,k,j,i);
					my_fields.metal_density[n]   = 0.0;
					my_fields.isrf_habing[n]     = pmb->ruser_meshblock_data[11](k,j,i)/1.6e-3;
					n++;
				}
			}
		}
		
		
		
		
		int cooling_bool=0;
		gr_float *cooling_time_metal_free = new gr_float[cellnum];
		if (calculate_cooling_time(&my_units, &my_fields, cooling_time_metal_free) == 0){
			std::stringstream nan_msg;
			nan_msg<<"Fatal Error: Error in calculate_cooling_time"<<std::endl;
			ATHENA_ERROR(nan_msg);
		}
		
		//rearrange field data to include the effect of metal
		for (int i=0; i<cellnum; i++){
			my_fields.metal_density[n] = 1.0*grackle_data->SolarMetalFractionByMass*my_fields.density[n];
		}
		
		gr_float *cooling_time_solar = new gr_float[cellnum];
		if ((cooling_bool==0)&&(calculate_cooling_time(&my_units, &my_fields, cooling_time_solar) == 0)){
			std::stringstream nan_msg;
			nan_msg<<"Fatal Error: Error in calculate_cooling_time"<<std::endl;
			ATHENA_ERROR(nan_msg);
		}
		
		//mesh to final cooling time result
		//energy lost/gain ratio will be calculated with dt divided by cooling time
		//in grackle cooling time is defined as e/(de/dt), when cooling happens it should be a negative value
		
		n=0;
		gr_float cooling_time[cellnum];
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real cell_metallicity = pmb->pscalars->r(0,k,j,i)/grackle_data->SolarMetalFractionByMass;
					if (cell_metallicity>5.0){
						cell_metallicity = 5.0;
					}
					cooling_time[n]= 1.0/((1.0/cooling_time_solar[n]-1.0/cooling_time_metal_free[n])*cell_metallicity+1.0/cooling_time_metal_free[n]);
					n++;
				}
			}
		}
		
		n=0;
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					//if ((user_cfl_number*std::abs(cooling_time[n])<dt_min)&&(pmb->iuser_meshblock_data[3](k,j,i)!=1)){
					if (std::isnan(cooling_time[n])==false){
						if (3.0*user_cfl_number*std::abs(cooling_time[n]/pmb->ruser_meshblock_data[10](k,j,i))<dt_min){
							dt_min=3.0*user_cfl_number*std::abs(cooling_time[n]/pmb->ruser_meshblock_data[10](k,j,i));
						}
					}
					n++;
				}
			}
		}
		
		//delete my_fields->grid_rank;
		delete [] my_fields.grid_dimension;
		delete [] my_fields.grid_start;
		delete [] my_fields.grid_end;
		delete [] my_fields.density;
		delete [] my_fields.internal_energy;
		delete [] my_fields.x_velocity;
		delete [] my_fields.y_velocity;
		delete [] my_fields.z_velocity;
		delete [] my_fields.metal_density;
		delete [] my_fields.HI_density;
		delete [] my_fields.HII_density;
		delete [] my_fields.HeI_density;
		delete [] my_fields.HeII_density;
		delete [] my_fields.HeIII_density;
		delete [] my_fields.e_density;
		delete [] my_fields.HM_density;
		delete [] my_fields.H2I_density;
		delete [] my_fields.H2II_density;
		delete [] my_fields.DI_density;
		delete [] my_fields.DII_density;
		delete [] my_fields.HDI_density;
		delete [] my_fields.volumetric_heating_rate;
		delete [] my_fields.specific_heating_rate;
		delete [] my_fields.RT_HI_ionization_rate;
		delete [] my_fields.RT_HeI_ionization_rate;
		delete [] my_fields.RT_HeII_ionization_rate;
		delete [] my_fields.RT_H2_dissociation_rate;
		delete [] my_fields.RT_heating_rate;
		delete [] my_fields.isrf_habing;
		delete [] my_fields.dust_density;
		delete [] my_fields.H2_self_shielding_length;
		delete [] cooling_time_metal_free;
		delete [] cooling_time_solar;
		//cooling_time = NULL;
	}
	
	if (gp_bool==1){
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real dx1 = pmb->pcoord->dx1v(j);
					Real odx1 = 1.0/dx1;
					Real dphi = pmb->ruser_meshblock_data[0](k,j,i+1)-pmb->ruser_meshblock_data[0](k,j,i-1);
					Real dm1 = 0.5*odx1*pmb->phydro->u(IDN,k,j,i)*(dphi);
					Real dengodt = std::abs(dm1*pmb->phydro->w(IVX,k,j,i));
					if (dt_min > pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt){
						dt_min = pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt;
					}
				}
			}
		}
		if (pmb->block_size.nx2 > 1) {
			// acceleration in 2-direction
			for (int k=pmb->ks; k<=pmb->ke; ++k) {
				for (int j=pmb->js; j<=pmb->je; ++j) {
					for (int i=pmb->is; i<=pmb->ie; ++i) {
						Real dx2 = pmb->pcoord->dx2v(j);
						Real odx2 = 1.0/dx2;
						Real dphi = pmb->ruser_meshblock_data[0](k,j+1,i)-pmb->ruser_meshblock_data[0](k,j-1,i);
						Real dm2 = 0.5*odx2*pmb->phydro->u(IDN,k,j,i)*(dphi);
						Real dengodt = std::abs(dm2*pmb->phydro->w(IVY,k,j,i));
						if (dt_min > pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt){
							dt_min = pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt;
						}
					}
				}
			}
		}

		if (pmb->block_size.nx3 > 1) {
			// acceleration in 3-direction
			for (int k=pmb->ks; k<=pmb->ke; ++k) {
				for (int j=pmb->js; j<=pmb->je; ++j) {
					for (int i=pmb->is; i<=pmb->ie; ++i) {
						Real dx3 = pmb->pcoord->dx3v(j);
						Real odx3 = 1.0/dx3;
						Real dphi = pmb->ruser_meshblock_data[0](k+1,j,i)-pmb->ruser_meshblock_data[0](k-1,j,i);
						Real dm3 = 0.5*odx3*pmb->phydro->u(IDN,k,j,i)*(dphi);
						Real dengodt = std::abs(dm3*pmb->phydro->w(IVZ,k,j,i));
						if (dt_min > pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt){
							dt_min = pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt;
						}
					}
				}
			}
		}
	}
	
	if (gp_bool==2){
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real x = pmb->pcoord->x1v(i);
					Real y = pmb->pcoord->x2v(j);
					Real z = pmb->pcoord->x3v(k);
					Real r = std::sqrt(SQR(x) + SQR(y));
					Real dengodt =0.0;
					if (r!=0.0){
						Real dphidr=(phi_m82_without_gas(r*1.01,0.0,z)-phi_m82_without_gas(r*0.99,0.0,z))/(0.02*r);
						Real dm1 = 0.5*pmb->phydro->w(IDN,k,j,i)*(dphidr)*(x/r);
						Real dm2 = 0.5*pmb->phydro->w(IDN,k,j,i)*(dphidr)*(x/r);
						dengodt -= dm1*pmb->phydro->w(IVX,k,j,i)+dm2*pmb->phydro->w(IVY,k,j,i);
					}
					if (z!=0.0){
						Real dphidz=(phi_m82_without_gas(r,0.0,z*1.01)-phi_m82_without_gas(r,0.0,z*0.99))/(0.02*z);
						Real dm3 = 0.5*pmb->phydro->w(IDN,k,j,i)*(dphidz);
						dengodt -= dm3*pmb->phydro->w(IVZ,k,j,i);
					}
					if (dt_min > pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt){
						dt_min = pmb->phydro->u(IEN,k,j,i)*user_cfl_number/dengodt;
					}
				}
			}
		}
	}
	
	
	/*
	if (star_bool==1){
		dt_min=1.0e8;
	}
	*/
	return dt_min;
}

//radiactive cooling function
void CoolingF(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){
	
	
	//std::cout<<time<<" "<<dt<<std::endl;
	
	/*
	if (nan_bool==1){
		//std::stringstream nan_msg;
		//nan_msg<<"Fatal Error:NaN Detected!";
		//ATHENA_ERROR(nan_msg);
	}
	*/
	/*
	if (scalar_nan_num>0){
		std::stringstream nan_msg;
		nan_msg<<"Fatal Error:Scalar NaN Detected!"<<std::endl
		<<"num="<<scalar_nan_num<<std::endl;
		ATHENA_ERROR(nan_msg);
	}
	*/
	
	//time check
	t_mainstep=time;
	dt_mainstep=dt;
	//cooling function
	//std::cout<<"x_1 range:"<<pmb->ks<<"----->"<<pmb->ke<<std::endl;
	if (cooling==1){
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					
					Real den=cons(IDN,k,j,i);
					Real ine=cons(IEN,k,j,i)-0.5*(cons(IM1,k,j,i)*cons(IM1,k,j,i)
						+cons(IM2,k,j,i)*cons(IM2,k,j,i)
						+cons(IM3,k,j,i)*cons(IM3,k,j,i))/den;
					//calculate local temp(K)
					Real real_temp=(ine/den)*m_h/(1.5*k_b);
					//Real real_temp=invert(pfromt,0.1*sim_temp,10*sim_temp,ine/(1.5*den));
					
					
					Real ne=nefromt(real_temp);
					
					Real xx=ne/cds[90][1];
					//mu with 1 M_solar goes as
					Real mu=4/(1+3*0.91);
					//Real cooling_coef=(den/1.4)*5e-33*pow(real_temp,1.5)/1.380649e-16*3.397278e14;
					Real cooling_coef=0;
					if (pow(den/(mu*m_h),2)<50){
						cooling_coef=lambdafunc(real_temp)*pow(den/(mu*m_h),2);
					}
					else{
						cooling_coef=lambdafunc(real_temp)*50;
					}
					
					//lower limit of temp with which x_3~0
					
					if ((((ine+dt*2.0e-26*den/(mu*m_h))/den)*m_h/(1.5*k_b)<1.0e4)&&(den/(mu*m_h)>0.01)){
						cooling_coef=cooling_coef-2.0e-26*den/(mu*m_h);
					}
					if (ine-cooling_coef*dt>0.8*ine){
						ine=ine-cooling_coef*dt;
					}
					else{
						ine=0.8*ine;
					}
					if (ine<1.5*den*t_floor*k_b/m_h){
						ine=1.5*den*t_floor*k_b/m_h;
					}
					
					cons(IEN,k,j,i)= ine+0.5*(cons(IM1,k,j,i)*cons(IM1,k,j,i)
											+cons(IM2,k,j,i)*cons(IM2,k,j,i)
											+cons(IM3,k,j,i)*cons(IM3,k,j,i))/den;
				}
			}
		}
	}
	if (cooling==2){
		
		//create struct for storing grackle field data
		grackle_field_data my_fields;
		int cellnum = (pmb->ke-pmb->ks+1)*(pmb->je-pmb->js+1)*(pmb->ie-pmb->is+1);
		my_fields.grid_rank = 3;
		my_fields.grid_dimension = new int[3];
		my_fields.grid_start = new int[3];
		my_fields.grid_end = new int[3];
		
		//int grid_dimension[3] = {pmb->ke-pmb->ks+1, pmb->je-pmb->js+1, pmb->ie-pmb->is+1};
		//int grid_start[3] = {0, 0, 0};
		//int grid_end[3] = {pmb->ke-pmb->ks, pmb->je-pmb->js, pmb->ie-pmb->is};
		my_fields.grid_dimension[0]=pmb->ke-pmb->ks+1;
		my_fields.grid_dimension[1]=pmb->je-pmb->js+1;
		my_fields.grid_dimension[2]=pmb->ie-pmb->is+1;
		my_fields.grid_start[0]=0;
		my_fields.grid_start[1]=0;
		my_fields.grid_start[2]=0;
		my_fields.grid_end[0]=pmb->ke-pmb->ks;
		my_fields.grid_end[1]=pmb->je-pmb->js;
		my_fields.grid_end[2]=pmb->ie-pmb->is;
		
		my_fields.grid_dx = 0.0;
		//creating cell data grips for grackle
		//since Athena++ does not provide direct path to density/internal energy/velocity field to which a pointer can point, we have to create our own arrays
		//but the major drawback of such method is that it is memory-consuming and time-consuming
		my_fields.density         = new gr_float[cellnum];
		my_fields.internal_energy = new gr_float[cellnum];
		my_fields.x_velocity      = new gr_float[cellnum];
		my_fields.y_velocity      = new gr_float[cellnum];
		my_fields.z_velocity      = new gr_float[cellnum];
		my_fields.metal_density   = new gr_float[cellnum];
		
		// for primordial_chemistry >= 1
		my_fields.HI_density      = new gr_float[cellnum];
		my_fields.HII_density     = new gr_float[cellnum];
		my_fields.HeI_density     = new gr_float[cellnum];
		my_fields.HeII_density    = new gr_float[cellnum];
		my_fields.HeIII_density   = new gr_float[cellnum];
		my_fields.e_density       = new gr_float[cellnum];
		// for primordial_chemistry >= 2
		my_fields.HM_density      = new gr_float[cellnum];
		my_fields.H2I_density     = new gr_float[cellnum];
		my_fields.H2II_density    = new gr_float[cellnum];
		// for primordial_chemistry >= 3
		my_fields.DI_density      = new gr_float[cellnum];
		my_fields.DII_density     = new gr_float[cellnum];
		my_fields.HDI_density     = new gr_float[cellnum];
		
		// volumetric heating rate (provide in units [erg s^-1 cm^-3])
		my_fields.volumetric_heating_rate = new gr_float[cellnum];
		// specific heating rate (provide in units [egs s^-1 g^-1]
		my_fields.specific_heating_rate = new gr_float[cellnum];
		
		// radiative transfer ionization / dissociation rate fields (provided in units of [1/s])
		my_fields.RT_HI_ionization_rate = new gr_float[cellnum];
		my_fields.RT_HeI_ionization_rate = new gr_float[cellnum];
		my_fields.RT_HeII_ionization_rate = new gr_float[cellnum];
		my_fields.RT_H2_dissociation_rate = new gr_float[cellnum];
		// radiative transfer heating rate (provide in units [erg s^-1 cm^-3])
		my_fields.RT_heating_rate = new gr_float[cellnum];
		
		// interstellar radiation field strength
		my_fields.isrf_habing = new gr_float[cellnum];
		
		//unused parameters
		my_fields.dust_density = new gr_float[cellnum];
		my_fields.H2_self_shielding_length = new gr_float[cellnum];
		
		//writing data into arrays declared above and 
		int n=0;
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real mu=4/(1+3*0.91);
					Real den=cons(IDN,k,j,i);
					Real ine=cons(IEN,k,j,i)-0.5*cons(IDN,k,j,i)*(prim(IVX,k,j,i)*prim(IVX,k,j,i)+
																	prim(IVY,k,j,i)*prim(IVY,k,j,i)+
																	prim(IVZ,k,j,i)*prim(IVZ,k,j,i));
					Real real_temp=(ine/den)*m_h/(1.5*k_b);
					if (real_temp<1.0*t_floor){
						//lower temp limit
						cons(IEN,k,j,i) = 1.5*den*t_floor*k_b/m_h+0.5*cons(IDN,k,j,i)*(prim(IVX,k,j,i)*prim(IVX,k,j,i)+
																						prim(IVY,k,j,i)*prim(IVY,k,j,i)+
																						prim(IVZ,k,j,i)*prim(IVZ,k,j,i));
					}
					if (real_temp>1.0e9){
						//higher temp limit = 1.0e8.5 K
						cons(IEN,k,j,i) = 1.5*den*1.0e9*k_b/m_h+0.5*cons(IDN,k,j,i)*(prim(IVX,k,j,i)*prim(IVX,k,j,i)+
																						prim(IVY,k,j,i)*prim(IVY,k,j,i)+
																						prim(IVZ,k,j,i)*prim(IVZ,k,j,i));
					}
					
					if (cons(IDN,k,j,i)<=1.0e3*m_h){
						my_fields.density[n]         = cons(IDN,k,j,i);
						my_fields.internal_energy[n] = (cons(IEN,k,j,i)
														-0.5*cons(IDN,k,j,i)*(prim(IVX,k,j,i)*prim(IVX,k,j,i)+
																			prim(IVY,k,j,i)*prim(IVY,k,j,i)+
																			prim(IVZ,k,j,i)*prim(IVZ,k,j,i)))/cons(IDN,k,j,i);
					}
					else{
						my_fields.density[n]         = 1.0e3*m_h;
						my_fields.internal_energy[n] = (1.0e3*m_h/cons(IDN,k,j,i))*(cons(IEN,k,j,i)
														-0.5*cons(IDN,k,j,i)*(prim(IVX,k,j,i)*prim(IVX,k,j,i)+
																			prim(IVY,k,j,i)*prim(IVY,k,j,i)+
																			prim(IVZ,k,j,i)*prim(IVZ,k,j,i)))/cons(IDN,k,j,i);
					}
					my_fields.x_velocity[n]      = prim(IVX,k,j,i);
					my_fields.y_velocity[n]      = prim(IVY,k,j,i);
					my_fields.z_velocity[n]      = prim(IVZ,k,j,i);
					my_fields.metal_density[n]   = 0.0;
					my_fields.isrf_habing[n]     = pmb->ruser_meshblock_data[11](k,j,i)/1.6e-3;
					n++;
				}
			}
		}
		
		int cooling_bool=0;
		gr_float *cooling_time_metal_free = new gr_float[cellnum];
		if (calculate_cooling_time(&my_units, &my_fields, cooling_time_metal_free) == 0){
			cooling_bool=1;
		}
		
		//rearrange field data to include the effect of metal
		for (int i=0; i<cellnum; i++){
			my_fields.metal_density[n] = 1.0*grackle_data->SolarMetalFractionByMass*my_fields.density[n];
		}
		
		gr_float *cooling_time_solar = new gr_float[cellnum];
		if ((cooling_bool==0)&&(calculate_cooling_time(&my_units, &my_fields, cooling_time_solar) == 0)){
			cooling_bool=1;
		}
		
		//mesh to final cooling time result
		//energy lost/gain ratio will be calculated with dt divided by cooling time
		//in grackle cooling time is defined as e/(de/dt), when cooling happens it should be a negative value
		
		n=0;
		gr_float cooling_time[cellnum];
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real cell_metallicity = prim_scalar(0,k,j,i)/grackle_data->SolarMetalFractionByMass;
					if (cell_metallicity>5.0){
						cell_metallicity = 5.0;
					}
					if ((std::isnan(cooling_time_metal_free[n])==true)||(std::isnan(cooling_time_solar[n])==true)){
						std::stringstream nan_msg;
						nan_msg<<"Fatal Error: nan cooling time "<<std::endl
						<<"den = "<<my_fields.density[n]<<std::endl
						<<"ine = "<<my_fields.internal_energy[n]<<std::endl
						<<"uv field = "<<my_fields.isrf_habing[n]<<std::endl
						<<"metal free cooling time = "<<cooling_time_metal_free[n]<<std::endl
						<<"solar metallicity cooling time = "<<cooling_time_solar[n]<<std::endl;
						ATHENA_ERROR(nan_msg);
					}
							
					cooling_time[n]= 1.0/((1.0/cooling_time_solar[n]-1.0/cooling_time_metal_free[n])*cell_metallicity+1.0/cooling_time_metal_free[n]);
					n++;
				}
			}
		}
		
		gr_float *temperature= new gr_float[cellnum];
		if (calculate_temperature(&my_units, &my_fields, temperature) == 0){
			cooling_bool=1;
		}
		
		//mesh to final temp result
		//temp will be used to determine whether a cell is below T_floor.
		/*
		gr_float *temperature;
		temperature = new gr_float[cellnum];
		if (calculate_temperature(&my_units, &my_fields,temperature) == 0) {
			std::stringstream nan_msg;
			nan_msg<<"Fatal Error: Error in calculate_temperature"<<std::endl;
			ATHENA_ERROR(nan_msg);
		}
		*/
		
		if (cooling_bool==0){
			n=0;
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						//if (pmb->iuser_meshblock_data[3](k,j,i)!=1){
							Real mu=4/(1+3*0.91);
							Real den=cons(IDN,k,j,i);
							Real ine=cons(IEN,k,j,i)-0.5*cons(IDN,k,j,i)*(prim(IVX,k,j,i)*prim(IVX,k,j,i)+
																			prim(IVY,k,j,i)*prim(IVY,k,j,i)+
																			prim(IVZ,k,j,i)*prim(IVZ,k,j,i));
							Real real_temp=(ine/den)*m_h/(1.5*k_b);
							if (std::isnan(cooling_time[n])==false){
								if (((real_temp>t_floor)||(cooling_time[n]>0))&&(ine*exp(dt/cooling_time[n])>1.5*den*t_floor*k_b/m_h)){
									//cons(IEN,k,j,i)+=pmb->ruser_meshblock_data[10](k,j,i)*ine*(dt/cooling_time[n]);
									cons(IEN,k,j,i)+=pmb->ruser_meshblock_data[10](k,j,i)*ine*(exp(dt/cooling_time[n])-1.0);
								}
								else{
									cons(IEN,k,j,i)+=-ine+1.5*den*t_floor*k_b/m_h;
								}
							}
						n++;
					}
				}
			}
		}
		
		//delete my_fields.grid_rank;
		delete [] my_fields.grid_dimension;
		delete [] my_fields.grid_start;
		delete [] my_fields.grid_end;
		delete [] my_fields.density;
		delete [] my_fields.internal_energy;
		delete [] my_fields.x_velocity;
		delete [] my_fields.y_velocity;
		delete [] my_fields.z_velocity;
		delete [] my_fields.metal_density;
		delete [] my_fields.HI_density;
		delete [] my_fields.HII_density;
		delete [] my_fields.HeI_density;
		delete [] my_fields.HeII_density;
		delete [] my_fields.HeIII_density;
		delete [] my_fields.e_density;
		delete [] my_fields.HM_density;
		delete [] my_fields.H2I_density;
		delete [] my_fields.H2II_density;
		delete [] my_fields.DI_density;
		delete [] my_fields.DII_density;
		delete [] my_fields.HDI_density;
		delete [] my_fields.volumetric_heating_rate;
		delete [] my_fields.specific_heating_rate;
		delete [] my_fields.RT_HI_ionization_rate;
		delete [] my_fields.RT_HeI_ionization_rate;
		delete [] my_fields.RT_HeII_ionization_rate;
		delete [] my_fields.RT_H2_dissociation_rate;
		delete [] my_fields.RT_heating_rate;
		delete [] my_fields.isrf_habing;
		delete [] my_fields.dust_density;
		delete [] my_fields.H2_self_shielding_length;
		delete [] cooling_time_metal_free;
		delete [] cooling_time_solar;
		delete [] temperature;
	}
	
	/*
	if (heating==1){
		//calculate flux within each meshblock
		Real stellar_mass_mb=0.0;
		Real j_FUV=0.0;
		Real heating_rate = 2.0e-26;
		
		//heating
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real den=cons(IDN,k,j,i);
					cons(IEN,k,j,i)+=dt*(den/m_h)*heating_rate;
				}
			}
		}
	}
	*/
	if (gp_bool==1){
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real dx1 = pmb->pcoord->dx1v(j);
					Real dtodx1 = dt/dx1;
					Real dphi = pmb->ruser_meshblock_data[0](k,j,i+1)-pmb->ruser_meshblock_data[0](k,j,i-1);
					Real dm1 = 0.5*dtodx1*prim(IDN,k,j,i)*(dphi);
					cons(IM1,k,j,i) -= dm1;
					if (NON_BAROTROPIC_EOS){
						cons(IEN,k,j,i) -= dm1*prim(IVX,k,j,i);
					}
				}
			}
		}
		if (pmb->block_size.nx2 > 1) {
			// acceleration in 2-direction
			for (int k=pmb->ks; k<=pmb->ke; ++k) {
				for (int j=pmb->js; j<=pmb->je; ++j) {
					for (int i=pmb->is; i<=pmb->ie; ++i) {
						Real dx2 = pmb->pcoord->dx2v(j);
						Real dtodx2 = dt/dx2;
						Real dphi = pmb->ruser_meshblock_data[0](k,j+1,i)-pmb->ruser_meshblock_data[0](k,j-1,i);
						Real dm2 = 0.5*dtodx2*prim(IDN,k,j,i)*(dphi);
						cons(IM2,k,j,i) -= dm2;
						if (NON_BAROTROPIC_EOS){
							cons(IEN,k,j,i) -= dm2*prim(IVY,k,j,i);
						}
					}
				}
			}
		}

		if (pmb->block_size.nx3 > 1) {
			// acceleration in 3-direction
			for (int k=pmb->ks; k<=pmb->ke; ++k) {
				for (int j=pmb->js; j<=pmb->je; ++j) {
					for (int i=pmb->is; i<=pmb->ie; ++i) {
						Real dx3 = pmb->pcoord->dx3v(j);
						Real dtodx3 = dt/dx3;
						Real dphi = pmb->ruser_meshblock_data[0](k+1,j,i)-pmb->ruser_meshblock_data[0](k-1,j,i);
						Real dm3 = 0.5*dtodx3*prim(IDN,k,j,i)*(dphi);
						cons(IM3,k,j,i) -= dm3;
						if (NON_BAROTROPIC_EOS){
							cons(IEN,k,j,i) -= dm3*prim(IVZ,k,j,i);
						}
					}
				}
			}
		}
	}
	if (gp_bool==2){
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					Real x = pmb->pcoord->x1v(i);
					Real y = pmb->pcoord->x2v(j);
					Real z = pmb->pcoord->x3v(k);
					Real r = std::sqrt(SQR(x) + SQR(y));
					if (r!=0.0){
						Real dphidr=(phi_m82_without_gas(r*1.01,0.0,z)-phi_m82_without_gas(r*0.99,0.0,z))/(0.02*r);
						Real dm1 = 0.5*dt*prim(IDN,k,j,i)*(dphidr)*(x/r);
						cons(IM1,k,j,i) -= dm1;
						Real dm2 = 0.5*dt*prim(IDN,k,j,i)*(dphidr)*(x/r);
						cons(IM2,k,j,i) -= dm2;
						if (NON_BAROTROPIC_EOS){
							cons(IEN,k,j,i) -= dm1*prim(IVX,k,j,i)+dm2*prim(IVY,k,j,i);
						}
					}
					if (z!=0.0){
						Real dphidz=(phi_m82_without_gas(r,0.0,z*1.01)-phi_m82_without_gas(r,0.0,z*0.99))/(0.02*z);
						Real dm3 = 0.5*dt*prim(IDN,k,j,i)*(dphidz);
						cons(IM3,k,j,i) -= dm3;
						if (NON_BAROTROPIC_EOS){
							cons(IEN,k,j,i) -= dm3*prim(IVZ,k,j,i);
						}
					}
					
				}
			}
		}
	}
	
	return;
}

int RefinementCondition(MeshBlock *pmb) {
	/*
	int refine_bo=0;
	
	for (int sup_num=0;sup_num<max_sup_num;sup_num++){
		if ((sds[sup_num][1]*3.15576e7 >= 0.0)&&(t_mainstep <= sds[sup_num][1]*3.15576e7)&&(t_mainstep+10*dt_mainstep > sds[sup_num][1]*3.15576e7)){
			Real x0=sds[sup_num][6];
			Real y0=sds[sup_num][7];
			Real z0=sds[sup_num][8];
			
			Real r_s      =sigma1;
			
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						Real x = pmb->pcoord->x1v(i);
						Real y = pmb->pcoord->x2v(j);
						Real z = pmb->pcoord->x3v(k);
						Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
						
						if (rad<=2.0*r_s){
							return 1;
						}
					}
				}
			}
		}
	}
	
	AthenaArray<Real> &w = pmb->phydro->w;
	Real epsmax = 0.0;
	for (int k=pmb->ks; k<=pmb->ke; k++) {
		for (int j=pmb->js-1; j<=pmb->je+1; j++) {
			for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
				Real epsx = std::abs(w(IDN,k,j,i+1) - w(IDN,k,j,i-1))/w(IDN,k,j,i);
				Real epsy = std::abs(w(IDN,k,j+1,i) - w(IDN,k,j-1,i))/w(IDN,k,j,i);
				Real epsz = std::abs(w(IDN,k+1,j,i) - w(IDN,k-1,j,i))/w(IDN,k,j,i);
				Real eps  = std::sqrt(SQR(epsx) + SQR(epsy) + SQR(epsz));
				
				if ((NSCALARS>0)&&(pmb->pscalars->r(0,k,j,i)>1.0e-4)){
					Real epsx_r =std::abs(pmb->pscalars->r(0,k,j,i+1) - pmb->pscalars->r(0,k,j,i-1))/pmb->pscalars->r(0,k,j,i);
					Real epsy_r =std::abs(pmb->pscalars->r(0,k,j+1,i) - pmb->pscalars->r(0,k,j-1,i))/pmb->pscalars->r(0,k,j,i);
					Real epsz_r =std::abs(pmb->pscalars->r(0,k+1,j,i) - pmb->pscalars->r(0,k-1,j,i))/pmb->pscalars->r(0,k,j,i);
					Real eps_r  =std::sqrt(SQR(epsx_r) + SQR(epsy_r) + SQR(epsz_r));
					if (eps_r*2.0>eps){
						eps=eps_r;
					}
				}
				
				Real x = pmb->pcoord->x1v(i);
				Real y = pmb->pcoord->x2v(j);
				Real z = pmb->pcoord->x3v(k);
				Real r=std::sqrt(SQR(x) + SQR(y) + SQR(z));
				Real sigma_amr=3.085678e20*0.85;
				Real contract_factor=exp(-(r*r)/(2*sigma_amr*sigma_amr));
				
				if (contract_factor*eps > epsmax)
					epsmax = contract_factor*eps;
			}
		}
	}
	if (epsmax > threshold) return 1;
	//if (epsmax < 0.05) return -1;//highly restricted derefinement condition
	*/
	for (int k=pmb->ks; k<=pmb->ke; k++) {
		for (int j=pmb->js-1; j<=pmb->je+1; j++) {
			for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
				Real x = pmb->pcoord->x1v(i);
				Real y = pmb->pcoord->x2v(j);
				Real z = pmb->pcoord->x3v(k);
				Real rr = std::sqrt(SQR(x)+SQR(y));
				if ((rr<500.0*3.0e18)&&(std::abs(z)<150.0*3.0e18)){
					return 1;
				}
			}
		}
	}
	return 0;
}

//Conduction coefficient function for hydrogen plasma
//Remember to set kappa_iso to a non-zero value in the input file, otherwise this function
//will not be called
//Remember to set eos to "general/hydrogen"
void HydrogenConductioncoeff(
    HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke){
	Real ine=0;//ine is the internal energy of a certain cell
	Real den=0;//den is the density of the currently examinated cell
	Real real_temp=0;//real_temp is temperature with unit K
	Real real_kappa=0;//??
	Real gamma_c=1.666667;
	Real mu=0;
	Real xx=0;
	Real ne=0;//number density of electron
	//Real kappa_cache[ke-ks+1][je-js+1][ie-is+1];
	
	for (int k=ks; k<=ke; k++) {
		for (int j=js; j<=je; j++) {
			for (int i=is; i<=ie; i++) {
				den=pmb->phydro->u(IDN,k,j,i);
				ine=pmb->phydro->u(IEN,k,j,i)-0.5*(pmb->phydro->u(IM1,k,j,i)*pmb->phydro->u(IM1,k,j,i)
					+pmb->phydro->u(IM2,k,j,i)*pmb->phydro->u(IM2,k,j,i)
					+pmb->phydro->u(IM3,k,j,i)*pmb->phydro->u(IM3,k,j,i))/den;
				
				//real_temp=invert(pfromt,0.1*ine*1.666667/den,10*ine*1.666667/den,ine*1.666667/den);
				Real real_temp=(ine/den)*m_h/(1.5*k_b);
				//real_temp=invert(pfromt,0.1*sim_temp,10*sim_temp,ine/(1.5*den));
				Real logt=log10(real_temp);
				
				ne=nefromt(real_temp);
				
				xx=ne/cds[90][1];
				mu=4/(1+3*0.91+4*0.91*xx);
				ne=xx*den/(mu*m_h);
				Real real_temp_cache=real_temp;
				
				if (real_temp>1.0e8){
					real_temp=1.0e8;
				}
				
				if (real_temp>6.6e4){
					real_kappa=1.70e11*pow(real_temp/1.0e7,2.5)
						/(1+0.029*std::log(real_temp*pow(1.2*ne/0.01,-0.5)/1.0e7));
				}
				else{
					real_kappa=2.5e5*pow(real_temp/1.0e4,0.5);
				}
				
				Real i_l=i-1;
				Real i_r=i+1;
				if (i_l<is){
					i_l=i;
				}
				if (i_r>ie){
					i_r=i;
				}
				Real dx = pmb->pcoord->x1v(i_r)-pmb->pcoord->x1v(i_l);
				Real den_l=den=pmb->phydro->u(IDN,k,j,i_l);
				Real den_r=den=pmb->phydro->u(IDN,k,j,i_r);
				Real ine_l=pmb->phydro->u(IEN,k,j,i_l)-0.5*(pmb->phydro->u(IM1,k,j,i_l)*pmb->phydro->u(IM1,k,j,i_l)
						+pmb->phydro->u(IM2,k,j,i_l)*pmb->phydro->u(IM2,k,j,i_l)
						+pmb->phydro->u(IM3,k,j,i_l)*pmb->phydro->u(IM3,k,j,i_l))/den_l;
				Real ine_r=pmb->phydro->u(IEN,k,j,i_r)-0.5*(pmb->phydro->u(IM1,k,j,i_r)*pmb->phydro->u(IM1,k,j,i_r)
						+pmb->phydro->u(IM2,k,j,i_r)*pmb->phydro->u(IM2,k,j,i_r)
						+pmb->phydro->u(IM3,k,j,i_r)*pmb->phydro->u(IM3,k,j,i_r))/den_r;
				Real dtemp_x=((ine_r/den_r)*m_h/(1.5*k_b)-(ine_l/den_l)*m_h/(1.5*k_b))/(dx);
				
				Real j_l=j-1;
				Real j_r=j+1;
				if (j_l<js){
					j_l=j;
				}
				if (j_r>je){
					j_r=j;
				}
				Real dy = pmb->pcoord->x2v(j_r)-pmb->pcoord->x2v(j_l);
				den_l=den=pmb->phydro->u(IDN,k,j_l,i);
				den_r=den=pmb->phydro->u(IDN,k,j_r,i);
				ine_l=pmb->phydro->u(IEN,k,j_l,i)-0.5*(pmb->phydro->u(IM1,k,j_l,i)*pmb->phydro->u(IM1,k,j_l,i)
					+pmb->phydro->u(IM2,k,j_l,i)*pmb->phydro->u(IM2,k,j_l,i)
					+pmb->phydro->u(IM3,k,j_l,i)*pmb->phydro->u(IM3,k,j_l,i))/den_l;
				ine_r=pmb->phydro->u(IEN,k,j_r,i)-0.5*(pmb->phydro->u(IM1,k,j_r,i)*pmb->phydro->u(IM1,k,j_r,i)
					+pmb->phydro->u(IM2,k,j_r,i)*pmb->phydro->u(IM2,k,j_r,i)
					+pmb->phydro->u(IM3,k,j_r,i)*pmb->phydro->u(IM3,k,j_r,i))/den_r;
				Real dtemp_y=((ine_r/den_r)*m_h/(1.5*k_b)-(ine_l/den_l)*m_h/(1.5*k_b))/(dy);
				
				
				Real k_l=k-1;
				Real k_r=k+1;
				if (k_l<ks){
					k_l=k;
				}
				if (k_r>ke){
					k_r=k;
				}
				Real dz = pmb->pcoord->x3v(k_r)-pmb->pcoord->x3v(k_l);
				den_l=den=pmb->phydro->u(IDN,k_l,j,i);
				den_r=den=pmb->phydro->u(IDN,k_r,j,i);
				ine_l=pmb->phydro->u(IEN,k_l,j,i)-0.5*(pmb->phydro->u(IM1,k_l,j,i)*pmb->phydro->u(IM1,k_l,j,i)
					+pmb->phydro->u(IM2,k_l,j,i)*pmb->phydro->u(IM2,k_l,j,i)
					+pmb->phydro->u(IM3,k_l,j,i)*pmb->phydro->u(IM3,k_l,j,i))/den_l;
				ine_r=pmb->phydro->u(IEN,k_r,j,i)-0.5*(pmb->phydro->u(IM1,k_r,j,i)*pmb->phydro->u(IM1,k_r,j,i)
					+pmb->phydro->u(IM2,k_r,j,i)*pmb->phydro->u(IM2,k_r,j,i)
					+pmb->phydro->u(IM3,k_r,j,i)*pmb->phydro->u(IM3,k_r,j,i))/den_r;
				Real dtemp_z=((ine_r/den_r)*m_h/(1.5*k_b)-(ine_l/den_l)*m_h/(1.5*k_b))/(dz);
				
				Real grad_temp=std::sqrt(SQR(dtemp_x) + SQR(dtemp_y) + SQR(dtemp_z));
				
				Real final_kappa=0;
				
				if (gt_limit<0){
					final_kappa=1/(1/real_kappa+grad_temp/(1.5*den*pow(gamma_c*k_b*real_temp/(mu*m_h),1.5)));
				}
				else{
					if (gt_limit>real_temp_cache*0.5){
						final_kappa=2/(1/real_kappa+grad_temp/(real_kappa*real_temp_cache*0.5)+grad_temp/(1.5*den*pow(gamma_c*k_b*real_temp/(mu*m_h),1.5)));
					}
					else{
						final_kappa=2/(1/real_kappa+grad_temp/(real_kappa*gt_limit)+grad_temp/(1.5*den*pow(gamma_c*k_b*real_temp/(mu*m_h),1.5)));
					}
				}
				
				//kappa(thermal diffusion coefficient) within Athena standard (cm^2s^-1)  
				phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i)=(final_kappa/den)*(mu*m_h/k_b);
			}
		}
	}
	//setting kappa to the vicinal minium value with the hope of reducing heat flux
	/*
	for (int k=ks; k<=ke; ++k) {
		for (int j=js; j<=je; ++j) {
			for (int i=is; i<=ie; ++i) {
				
				kappa_cache[k-ks][j-js][i-is]=phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i);
				Real weight=0;
				Real HM=0;//Harmonic mean value, the HM value is always of the same order of magnitude of the minimum value.
				
				for (int kk=-2; kk<=2; ++kk){
					if ((k+kk<=ke)&&(k+kk>=ks)){
						for (int jj=-2; jj<=2; ++jj){
							if ((j+jj<=je)&&(j+jj>=js)){
								for (int ii=-2; ii<=2; ++ii){
									if ((i+ii<=ie)&&(i+ii>=is)){
										//if (phdif->kappa(HydroDiffusion::DiffProcess::iso,k+kk,j+jj,i+ii)<kappa_cache[k-ks][j-js][i-is]){
											//kappa_cache[k-ks][j-js][i-is]=phdif->kappa(HydroDiffusion::DiffProcess::iso,k+kk,j+jj,i+ii);
										weight=weight+1;
										HM=HM+1/phdif->kappa(HydroDiffusion::DiffProcess::iso,k+kk,j+jj,i+ii);
									
									}
								}
							}
						}
					}
				}
				
				HM=weight/HM;
				kappa_cache[k-ks][j-js][i-is]=HM;
			}
		}
	}
	for (int k=ks; k<=ke; ++k) {
		for (int j=js; j<=je; ++j) {
			for (int i=is; i<=ie; ++i) {
				phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i)=0.1*kappa_cache[k-ks][j-js][i-is];
			}
		}
	}
	*/
	return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief this generator generates a field with even density and pressure
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
	//assume that ism temp is low enough to be considered as ideal gas
	Real gamma_cache= 1.666667;
	gm1=gamma_cache- 1.0;
	Real xx=3.45;
	
	//only applies when disk_sim==0
	Real en= 1.5*rho_cc*t_amb*k_b/m_h;
	
	//for m82 simulations
	//Real t_halo = 6.7e6;
	//Real t_disk = 4.0e5;//combined effective temperature of turbulence and thermal pressure
	//Real c_s_h = std::sqrt(gamma_cahce*k_b*t_halo/(m_h));
	Real c_s_h = 3.0e7;
	//Real c_s_d = std::sqrt(gamma_cahce*k_b*t_disk/(m_h));
	//Real c_s_d = 7.0e6;//(cm*s^-1) coordinated to the energy density enough to support the disk
	Real c_s_d =5.0e6;
	Real cs_without_turb = std::sqrt(SQR(c_s_d)-SQR(2.0e6));
	
	for (int k=ks; k<=ke; k++) {
		for (int j=js; j<=je; j++) {
			for (int i=is; i<=ie; i++) {
				
				Real x = pcoord->x1v(i);
				Real y = pcoord->x2v(j);
				Real z = pcoord->x3v(k);
				Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
				Real rr = std::sqrt(SQR(x) + SQR(y));
				
				xx+=x+y*y+z*z*z;
				Real den=0.0;
				Real ine=0.0;
				/*
				if ((disk_sim==0)&&(m82_sim==0)){
					den = (rho0 + (rho_cc-rho0)*exp(-(rad*rad)/(2*sigma4*sigma4)))*(1+sup_dis(prand(xx),sigma3));
					//Real den = rho0*(1+sup_dis(prand(xx),sigma3));
					
					
					//without initial blast
					Real ind = 0.0;
					ine = 0.0;
					
					phydro->u(IDN,k,j,i) = den+ind;
					phydro->u(IM1,k,j,i) = 0.0;
					phydro->u(IM2,k,j,i) = 0.0;
					phydro->u(IM3,k,j,i) = 0.0;
					if (NON_BAROTROPIC_EOS) {
						//Real pres = p0;
						
						phydro->u(IEN,k,j,i) = en+ine;
						
						if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
							phydro->u(IEN,k,j,i) += den+ind;
					}
				}
				if (disk_sim==1){
					den=rho0*exp(-a1*rho0*(std::sqrt(SQR(z)+SQR(z0))-z0)/p0-a2*rho0*SQR(z)/(2*p0));
					ine=1.5*den*t_amb*k_b/m_h;
					phydro->u(IDN,k,j,i) = den;
					phydro->u(IM1,k,j,i) = 0.0;
					phydro->u(IM2,k,j,i) = 0.0;
					phydro->u(IM3,k,j,i) = 0.0;
					if (NON_BAROTROPIC_EOS) {
						//Real pres = p0;
						
						phydro->u(IEN,k,j,i) = ine;
						
						if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
							phydro->u(IEN,k,j,i) += den;
					}
				}
				*/
				if (m82_sim==1){
					//rotational support factor
					Real e_rot = rot_factor*exp(-std::abs(z)/(5.0*1000.0*3.0e18));
					
					Real rho_halo = rho_h_0*exp(-(phi_m82_with_gas(x,y,z)-SQR(e_rot)*phi_m82_with_gas(x,y,0.0)-(1-SQR(e_rot))*phi_m82_with_gas(0.0,0.0,0.0))/SQR(c_s_h));
					
					Real rho_disk = rho_d_0*exp(-(phi_m82_with_gas(x,y,z)-SQR(e_rot)*phi_m82_with_gas(x,y,0.0)-(1-SQR(e_rot))*phi_m82_with_gas(0.0,0.0,0.0))/SQR(c_s_d));
					
					den = rho_halo+rho_disk;
					//den = rho_3m(x,y,z,e_gas,r_hm_gas,m_gas);
					
					Real rotationalvelo=-rtov(x,y,z);
					
					phydro->u(IDN,k,j,i) = den;//rho_halo+rho_disk;
					phydro->u(IM1,k,j,i) = 0.0;
					phydro->u(IM2,k,j,i) = 0.0;
					phydro->u(IM3,k,j,i) = 0.0;
					
					if (rr>0.0){
						phydro->u(IM1,k,j,i) = rotationalvelo*phydro->u(IDN,k,j,i)*(-y/rr);
						phydro->u(IM2,k,j,i) = rotationalvelo*phydro->u(IDN,k,j,i)*(x/rr);
						phydro->u(IM3,k,j,i) = 0.0;
					}
					
					if (NON_BAROTROPIC_EOS) {
						//phydro->u(IEN,k,j,i) = rho_halo*SQR(c_s_h) + rho_disk*SQR(c_s_d) + 0.5*(rho_halo+rho_disk)*SQR(rotationalvelo);
						if (turb_cuz==0){
							phydro->u(IEN,k,j,i) = den*(SQR(c_s_d)+0.5*SQR(rotationalvelo));
						}
						else{
							phydro->u(IEN,k,j,i) = den*(SQR(cs_without_turb)+0.5*SQR(rotationalvelo));
						}
						
						if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
							phydro->u(IEN,k,j,i) += phydro->u(IDN,k,j,i);
					}
				}
				if (NSCALARS>0){
					pscalars->s(0,k,j,i)=0.02*0.013*den;
				}
			}
		}
	}
	
	if (turb_cuz==1){
		Real zone_scale=0.0;
		/*
		if (disk_sim==1){
			zone_scale=4.0*3.085678e20;
		}
		else if(m82_sim==1){
			zone_scale=3.0e21;
		}
		else{
			zone_scale=2.0*3.085678e20;
		}
		*/
		if(m82_sim==1){
			zone_scale=6.0e21;
		}
		Real dx = pcoord->dx1v(ie);
		Real dy = pcoord->dx2v(je);
		Real dz = pcoord->dx3v(ke);
		
		for (int k=ks; k<=ke; k++) {
			for (int j=js; j<=je; j++) {
				for (int i=is; i<=ie; i++) {
					
					//set cell data
					Real x = pcoord->x1v(i);
					Real y = pcoord->x2v(j);
					Real z = pcoord->x3v(k);
					Real den = phydro->u(IDN,k,j,i);
					
					if (m82_sim==1){
						Real e_rot = rot_factor*exp(-std::abs(z)/(5.0*1000.0*3.0e18));
						//in m82 ism simulation turbulence is only applied to disk component, hot halo is set to be even and still
						den = rho_d_0*exp(-(phi_m82_with_gas(x,y,z)-SQR(e_rot)*phi_m82_with_gas(x,y,0.0)-(1-SQR(e_rot))*phi_m82_with_gas(0.0,0.0,0.0))/SQR(c_s_d));
					}
					
					//original velocity square
					//Real v_abs2_org=SQR(pmb->phydro->w(IVX,k,j,i))+SQR(pmb->phydro->w(IVX,k,j,i))+SQR(pmb->phydro->w(IVX,k,j,i));
					Real v_abs2_org=(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i)))/SQR(phydro->u(IDN,k,j,i));
					
					//Real ph[3]={2.5238,1.6427,0.9139};
					Real v[3]={0.0,0.0,0.0};
					
					//Real theta=2.5238;
					//Real delta=5.6241;
					
					std::mt19937 mt(20);
					for (int nx=-nhigh/2; nx<=nhigh/2; nx++){
						
						Real kx=4.0*M_PI*nx/zone_scale;
						
						for (int ny=-nhigh/2; ny<=nhigh/2; ny++){
							
							Real ky=4.0*M_PI*ny/zone_scale;
							
							for (int nz=-nhigh/2; nz<=nhigh/2; nz++){
								
								Real kz=4.0*M_PI*nz/zone_scale;
								//set phase
								//for (int di=0;di<3;di++){
								//	ph[di]=fmod((ph[di]*ph[di]+delta),2.0*M_PI);
								//}
								//check n range
								Real nmag=std::sqrt(SQR(nx)+SQR(ny)+SQR(nz));
								Real kmag=std::sqrt(SQR(kx)+SQR(ky)+SQR(kz));
								if ((nmag>=nlow)&&(nmag<=nhigh)&&(zone_scale/nmag>2*dx)){
									//d0:compress direction
									//d1 & d2 the two shearing direction orthogonal to d0
									//d1=d0 cross unit_vector_reference
									//d2=d0 cross d1
									
									//Real d0[3]={kx/kmag,ky/kmag,kz/kmag};
									//Real vf[3]={0.577350, 0.577350, -0.577350};
									//Real d1[3]={d0[1]*vf[2]-d0[2]*vf[1], d0[2]*vf[0]-d0[0]*vf[2], d0[0]*vf[1]-d0[1]*vf[0]};
									//Real d2[3]={d0[1]*d1[2]-d0[2]*d1[1], d0[2]*d1[0]-d0[0]*d1[2], d0[0]*d1[1]-d0[1]*d1[0]};
									
									//set velocity
									
									for (int di=0;di<3;di++){
										Real theta=2*M_PI*(Real)(mt())/(Real)(mt.max());
										v[di]+=(1.0/pow(kmag,1.0+expo/2.0))*cos(kx*x+ky*y+kz*z+theta);
									}
									
									/*
									for (int di=0;di<3;di++){
										v[di]+=(std::sqrt(1-f_shear)/0.577350)*d0[di]*(1.0/pow(kmag,expo/2.0))*cos(kx*x+ky*y+kz*z+ph[0])
										       +(std::sqrt(f_shear/2)/0.577350)*d1[di]*(1.0/pow(kmag,expo/2.0))*cos(kx*x+ky*y+kz*z+ph[1])
										       +(std::sqrt(f_shear/2)/0.577350)*d2[di]*(1.0/pow(kmag,expo/2.0))*cos(kx*x+ky*y+kz*z+ph[2]);
									*/
									
								}
							}
						}
					}
					
					phydro->u(IM1,k,j,i)+=scale_factor*den*v[0];
					phydro->u(IM2,k,j,i)+=scale_factor*den*v[1];
					phydro->u(IM3,k,j,i)+=scale_factor*den*v[2];
					//phydro->u(IEN,k,j,i)+=0.5*pow(scale_factor,2.0)*den*(SQR(v[0])+SQR(v[1])+SQR(v[2]));
					//final velocity square
					Real v_abs2_fin=(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i)))/SQR(phydro->u(IDN,k,j,i));
					//correction made for kinetic energy (IEN in conservatives includes k-energy 
					//therefore energy gain in grav potential needs to be taken into consideration)
					//with out this correction some cell with high acceleration could end up with
					//unphysical cooling and even negative pressure
					phydro->u(IEN,k,j,i) += 0.5*(v_abs2_fin - v_abs2_org)*phydro->u(IDN,k,j,i);
				}
			}
		}
	}
	
	return;
}

//========================================================================================
//Initialize gravitional potential field data using MeshBlock::InitUserMeshBlockData
//field data or future particle data will be automatically saved and reloaded during a 
//restart
//parameters are to be set 
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
	AllocateIntUserMeshBlockDataField(4);
	AllocateRealUserMeshBlockDataField(12);
	if (gp_bool==1){
		int nx1 = block_size.nx1 + 2*NGHOST;
		int nx2 = block_size.nx2 + 2*NGHOST;
		int nx3 = block_size.nx3 + 2*NGHOST;
		//in this version gravitational potential phi will be store here
		Real phi_edge = phi_3m(900*3.0e18,0.0,0.0,e_gas,r_hm_gas,m_gas);
		ruser_meshblock_data[0].NewAthenaArray(nx3,nx2,nx1);
		for (int k=ks-NGHOST; k<=ke+NGHOST; k++) {
			for (int j=js-NGHOST; j<=je+NGHOST; j++) {
				for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
					Real x = pcoord->x1v(i);
					Real y = pcoord->x2v(j);
					Real z = pcoord->x3v(k);
					Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
					if (disk_sim==1){
						//=====local disk like profile=====
						ruser_meshblock_data[0](k,j,i) = -a1*z/std::sqrt(SQR(z0) + SQR(z))-a2*z;
					}
					
					if (m82_sim==1){
						//=====central m82 disk profile=====
						/*
						//previous method, now abandoned
						//according to C.Melioli(2012) disk only consisits of gas and SS only consists of star
						//which I found doubtable, some article indicates that only 30-40% of the tot mass is in the ISM
						//where else can they be? halo? dark matter(not to be considered in close central region)? stellar fraction?
						//by far C.Melioli(2012)'s method is adopted.
						//further examination is needed.
						//stellar spheroid
						Real phi_ss = (-g*m_ss/omega_0)*(log((rad+1.0e10)/omega_0+std::sqrt(1+SQR(rad+1.0e10)/SQR(omega_0)))/((rad+1.0e10)/omega_0));
						//disk
						Real phi_disk = -g*m_disk/std::sqrt(SQR(x)+SQR(y)+SQR(m82para_a+std::sqrt(SQR(z)+SQR(m82para_b)))+0.1);
						*/
						
						//===star + DM halo only fraction===
						//this setup only works together with gas self-gravity solver
						//conmpatable with or without sink particle
						ruser_meshblock_data[0](k,j,i) = phi_m82_without_gas(x,y,z);
						
						//===with src_mask===
						//Real phi_3m_local = phi_3m(x,y,z,e_gas,r_hm_gas,m_gas);
						//if (phi_3m_local<phi_edge){
						//	ruser_meshblock_data[0](k,j,i) = phi_m82_without_gas(x,y,z) + phi_edge;
						//}
						//else{
						//	ruser_meshblock_data[0](k,j,i) = phi_m82_with_gas(x,y,z);
						//}
						
						//===star + gas===
						//this setup only works without gas self-gravity solver
						//however incompatable with sink particle
						//ruser_meshblock_data[0](k,j,i) = phi_ss+phi_disk;
					}
				}
			}
		}
	}
	else{
		ruser_meshblock_data[0].NewAthenaArray(1);
	}
	//if (particle_bool==1){
		int nx1 = block_size.nx1 + 2*NGHOST;
		int nx2 = block_size.nx2 + 2*NGHOST;
		int nx3 = block_size.nx3 + 2*NGHOST;
		//num of particles
		iuser_meshblock_data[0].NewAthenaArray(1);
		iuser_meshblock_data[0](0)=0;
		//particle signature
		iuser_meshblock_data[1].NewAthenaArray(512);//!!!!!!!!!
		//particle types:
		//-3 for unresolved star particles
		//-2 for eliminated old particles
		//-1 for unnamed new particles
		//0 for cloudlet(coupled with velo field),
		//1 for sink particles (affected by gravity, accretion takes place here)
		//2 for star particles
		iuser_meshblock_data[2].NewAthenaArray(512);
		iuser_meshblock_data[3].NewAthenaArray(512);
		//iuser_meshblock_data[3].NewAthenaArray(nx3,nx2,nx1);
		ruser_meshblock_data[1].NewAthenaArray(512);//mass
		ruser_meshblock_data[2].NewAthenaArray(512);//age
		ruser_meshblock_data[3].NewAthenaArray(512);//x1
		ruser_meshblock_data[4].NewAthenaArray(512);//x2
		ruser_meshblock_data[5].NewAthenaArray(512);//x3
		ruser_meshblock_data[6].NewAthenaArray(512);//v1
		ruser_meshblock_data[7].NewAthenaArray(512);//v2
		ruser_meshblock_data[8].NewAthenaArray(512);//v3
		ruser_meshblock_data[9].NewAthenaArray(512);//metal mass
		ruser_meshblock_data[10].NewAthenaArray(nx3,nx2,nx1);//cooling mask
		ruser_meshblock_data[11].NewAthenaArray(nx3,nx2,nx1);//UV intensity in cgs unit:erg cm-2 s-1
		
		Real dx = pcoord->dx1v(ie);//should better use pco->dx1v but not quite sure how it works
		Real dy = pcoord->dx2v(je);
		Real dz = pcoord->dx3v(ke);
		Real xl = pcoord->x1v(is)-0.5*dx;
		Real xr = pcoord->x1v(ie)+0.5*dx;
		Real yl = pcoord->x2v(js)-0.5*dy;
		Real yr = pcoord->x2v(je)+0.5*dy;
		Real zl = pcoord->x3v(ks)-0.5*dz;
		Real zr = pcoord->x3v(ke)+0.5*dz;
		for (int n=0; n<particle_num; n++){
			if ((pds[n][0]>=xl)&(pds[n][0]<xr)&
			    (pds[n][1]>=yl)&(pds[n][1]<yr)&
			    (pds[n][2]>=zl)&(pds[n][2]<zr)){
				//num of particles ++
				iuser_meshblock_data[0](0)++;
				int nc=iuser_meshblock_data[0](0)-1;
				//signature
				iuser_meshblock_data[1](nc)=n;
				//particle type
				iuser_meshblock_data[2](nc)=2;
				//mass
				ruser_meshblock_data[1](nc)=5.0e4*2.0e33;
				//age
				ruser_meshblock_data[2](nc)=0.0;
				//x1-x3
				ruser_meshblock_data[3](nc)=pds[n][0];
				ruser_meshblock_data[4](nc)=pds[n][1];
				ruser_meshblock_data[5](nc)=pds[n][2];
				
				Real rotationalvelo=rtov(pds[n][0],pds[n][1],pds[n][2]);
				Real rr=std::sqrt(SQR(pds[n][0])+SQR(pds[n][1]));
				
				//v1-v3
				ruser_meshblock_data[6](nc)=rotationalvelo*(-pds[n][1]/rr);
				ruser_meshblock_data[7](nc)=rotationalvelo*(pds[n][0]/rr);
				ruser_meshblock_data[8](nc)=0.0;
				//metal mass
				if (NSCALARS>0){
					ruser_meshblock_data[9](nc)=0.0;
				}
				std::cout<<"got one"<<std::endl;
				std::cout<<"signature"<<iuser_meshblock_data[1](nc)<<std::endl;
				std::cout<<" "<<ruser_meshblock_data[3](nc)<<" "<<ruser_meshblock_data[4](nc)<<" "<<ruser_meshblock_data[5](nc)<<std::endl;
			}
			
		}
	//}
	return;
}

void Mesh::UserWorkInLoop() {
	//nan and abnormalty detection as well as injection process has been moved to here
	//Real time_0=time;
	//Real time_1=time+dt;
	
	//NaN and other abnomalty detection
	int nan_bool=0;
	//int scalar_nan_num=0;
	for (int bn=0; bn<nblocal; ++bn) {
		MeshBlock *pmb = my_blocks(bn);
		LogicalLocation &loc = pmb->loc;
		for (int k=pmb->ks; k<=pmb->ke; k++) {
			for (int j=pmb->js; j<=pmb->je; j++) {
				for (int i=pmb->is; i<=pmb->ie; i++) {
					
					Real x = pmb->pcoord->x1v(i);
					Real y = pmb->pcoord->x2v(j);
					Real z = pmb->pcoord->x3v(k);
					
					
					int local_nan_bool=0;
					//conservative
					if (std::isnan(pmb->phydro->u(IDN,k,j,i))==true){
						std::cout<<"Density  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->u(IEN,k,j,i))==true){
						std::cout<<"Energy  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->u(IM1,k,j,i))==true){
						std::cout<<"Momentum_dir1  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->u(IM2,k,j,i))==true){
						std::cout<<"Momentum_dir2  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->u(IM3,k,j,i))==true){
						std::cout<<"Momentum_dir3  ";
						local_nan_bool=1;
						nan_bool=1;
					} 
					//primitive
					if (std::isnan(pmb->phydro->w(IDN,k,j,i))==true){
						std::cout<<"Primitive Density  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->w(IPR,k,j,i))==true){
						std::cout<<"Pressure  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->w(IVX,k,j,i))==true){
						std::cout<<"Velocity_dir1  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->w(IVY,k,j,i))==true){
						std::cout<<"Velocity_dir2  ";
						local_nan_bool=1;
						nan_bool=1;
					}
					if (std::isnan(pmb->phydro->w(IVZ,k,j,i))==true){
						std::cout<<"Velocity_dir3  ";
						local_nan_bool=1;
						nan_bool=1;
					} 
					
					/*
					if (NSCALARS>0){
						if ((std::isnan(cons_scalar(0,k,j,i))==false)&&(prim_scalar(0,k,j,i)>0.6)){
							std::stringstream nan_msg;
							nan_msg<<"Fatal Error:Passive Scalar Diverges!"<<std::endl;
							ATHENA_ERROR(nan_msg);
						}
						if (std::isnan(cons_scalar(0,k,j,i))==true){
							std::cout<<"Momentum_dir3  ";
							local_nan_bool=1;
							nan_bool=1;
						}
					}
					*/
					if (NSCALARS>0){
						if ((std::isnan(pmb->pscalars->s(0,k,j,i))==true)||(std::isnan(pmb->pscalars->r(0,k,j,i))==true)){
							std::cout<<"Scalar  ";
							local_nan_bool=1;
							nan_bool=1;
						}
					}
					
					if(local_nan_bool==1){
						std::cout<<std::endl;
						std::cout<<" at x="<<x
								 <<" y="<<y
								 <<" z="<<z
								 <<" appear to be nan"<<std::endl;
						
						//reconstruct data stored in the cell
						Real num_nnan=0.0;
						Real den=0.0;
						Real pre=0.0;
						Real v1=0.0;
						Real v2=0.0;
						Real v3=0.0;
						Real metallicity=0.0;
						/*
						for (int dk=-1; dk<=1; dk++) {
							for (int dj=-1; dj<=1; dj++) {
								for (int di=-1; di<=1; di++) {
									if ((k + dk>=pmb->ks)&&(k + dk<=pmb->ke)&&
										(j + dj>=pmb->js)&&(j + dj<=pmb->je)&&
										(i + di>=pmb->is)&&(i + di<=pmb->ie)){
										if ((std::isnan(pmb->phydro->w(IDN,k+dk,j+dj,i+di))==false)&&
											(std::isnan(pmb->phydro->w(IPR,k+dk,j+dj,i+di))==false)&&
											(std::isnan(pmb->phydro->w(IVX,k+dk,j+dj,i+di))==false)&&
											(std::isnan(pmb->phydro->w(IVY,k+dk,j+dj,i+di))==false)&&
											(std::isnan(pmb->phydro->w(IVZ,k+dk,j+dj,i+di))==false)){
											int recon_bo=0;
											if (NSCALARS>0){
												if (std::isnan(pmb->pscalars->r(0,k+dk,j+dj,i+di))==true){
													recon_bo=1;
												}
											}
											if (recon_bo==0){
												num_nnan+=1.0;
												den  += pmb->phydro->w(IDN,k+dk,j+dj,i+di);
												pre  += pmb->phydro->w(IPR,k+dk,j+dj,i+di);
												v1   += pmb->phydro->w(IVX,k+dk,j+dj,i+di);
												v2   += pmb->phydro->w(IVY,k+dk,j+dj,i+di);
												v3   += pmb->phydro->w(IVZ,k+dk,j+dj,i+di);
												if (NSCALARS>0){
													metallicity+=pmb->pscalars->r(0,k+dk,j+dj,i+di);
												}
											}
										}
									}
								}
							}
						}
						*/
						num_nnan = 1.0;
						den      = 1.0*m_h;
						pre      = 1.5*1.0*1.0e4*k_b;
						v1       = 0.0;
						v2       = 0.0;
						v3       = 0.0;
						metallicity=0.0;
						
						if (num_nnan>0){
							//consevative
							pmb->phydro->u(IDN,k,j,i) = den/num_nnan;
							pmb->phydro->u(IM1,k,j,i) = (den/num_nnan)*(v1/num_nnan);
							pmb->phydro->u(IM2,k,j,i) = (den/num_nnan)*(v2/num_nnan);
							pmb->phydro->u(IM3,k,j,i) = (den/num_nnan)*(v3/num_nnan);
							pmb->phydro->u(IEN,k,j,i) = pre/num_nnan+
														0.5*(den/num_nnan)*(v1/num_nnan)*(v1/num_nnan)+
														0.5*(den/num_nnan)*(v2/num_nnan)*(v2/num_nnan)+
														0.5*(den/num_nnan)*(v3/num_nnan)*(v3/num_nnan);
							
							//primitive
							pmb->phydro->w(IDN,k,j,i) = den/num_nnan;
							pmb->phydro->w(IVX,k,j,i) = v1/num_nnan;
							pmb->phydro->w(IVY,k,j,i) = v2/num_nnan;
							pmb->phydro->w(IVZ,k,j,i) = v3/num_nnan;
							pmb->phydro->w(IPR,k,j,i) = pre/num_nnan;
							if (NSCALARS>0){
								//cons_scalar(0,k,j,i) = (den/num_nnan)*(metallicity/num_nnan);
								pmb->pscalars->r(0,k,j,i)=metallicity/num_nnan;
								pmb->pscalars->s(0,k,j,i)=(den/num_nnan)*(metallicity/num_nnan);
							}
						}
						else{
							std::stringstream nan_msg;
							nan_msg<<"Fatal Error:Too Much Information Lost to Interpolate!"<<std::endl;
							ATHENA_ERROR(nan_msg);
						}
					}
				}
			}
		}
	}
	
	if (NSCALARS>0){
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			LogicalLocation &loc = pmb->loc;
			
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						if ((std::isnan(pmb->pscalars->r(0,k,j,i))==true)||(pmb->pscalars->r(0,k,j,i)>10.0*grackle_data->SolarMetalFractionByMass)){
							/*
							Real den=pmb->phydro->u(IDN,k,j,i);
							Real ine=pmb->phydro->u(IEN,k,j,i)-0.5*pmb->phydro->u(IDN,k,j,i)*(pmb->phydro->w(IVX,k,j,i)*pmb->phydro->w(IVX,k,j,i)+
																	pmb->phydro->w(IVY,k,j,i)*pmb->phydro->w(IVY,k,j,i)+
																	pmb->phydro->w(IVZ,k,j,i)*pmb->phydro->w(IVZ,k,j,i));
							Real real_temp=(ine/den)*m_h/(1.5*k_b);
							std::cout<<"Location of Divergence:"<<std::endl;
							std::cout<<"x = "<<pmb->pcoord->x1v(i)<<std::endl;
							std::cout<<"y = "<<pmb->pcoord->x2v(j)<<std::endl;
							std::cout<<"z = "<<pmb->pcoord->x3v(k)<<std::endl;
							std::cout<<"r = "<<pmb->pscalars->r(0,k,j,i)<<std::endl;
							std::cout<<"den = "<<pmb->phydro->u(IDN,k,j,i)<<std::endl;
							std::cout<<"temp = "<<real_temp<<std::endl;
							std::stringstream nan_msg;
							nan_msg<<"Fatal Error:Passive Scalar Diverges!"<<std::endl;
							ATHENA_ERROR(nan_msg);
							*/
							pmb->pscalars->r(0,k,j,i)=10.0*grackle_data->SolarMetalFractionByMass;
							pmb->pscalars->s(0,k,j,i)=10.0*grackle_data->SolarMetalFractionByMass*pmb->phydro->u(IDN,k,j,i);
						}
					}
				}
			}
		}
	}
	
	dt_mainstep = dt;
	t_mainstep  = time+dt;
	
	if (time>0.0*1.0e6*3.154e7){
		particle_bool=1;
		cooling=2;
		heating=1;
		t_step_delay=1;
		gp_bool=1;
	}
	
	if (particle_bool==1){
		
		//stellar feedback
		
		
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			LogicalLocation &loc = pmb->loc;
			Real dx = pmb->pcoord->dx1v(pmb->ie);
			Real dy = pmb->pcoord->dx2v(pmb->je);
			Real dz = pmb->pcoord->dx3v(pmb->ke);
			Real xl = pmb->pcoord->x1v(pmb->is)-0.5*dx;
			Real xr = pmb->pcoord->x1v(pmb->ie)+0.5*dx;
			Real yl = pmb->pcoord->x2v(pmb->js)-0.5*dy;
			Real yr = pmb->pcoord->x2v(pmb->je)+0.5*dy;
			Real zl = pmb->pcoord->x3v(pmb->ks)-0.5*dz;
			Real zr = pmb->pcoord->x3v(pmb->ke)+0.5*dz;
			
			for (int pn=0;pn<pmb->iuser_meshblock_data[0](0);pn++){
				if ((pmb->iuser_meshblock_data[2](pn)==2)||(pmb->iuser_meshblock_data[2](pn)==-3)){
					Real x0=pmb->ruser_meshblock_data[3](pn);
					Real y0=pmb->ruser_meshblock_data[4](pn);
					Real z0=pmb->ruser_meshblock_data[5](pn);
					
					Real vx=pmb->ruser_meshblock_data[6](pn);
					Real vy=pmb->ruser_meshblock_data[7](pn);
					Real vz=pmb->ruser_meshblock_data[8](pn);
					
					Real r0=pmb->ruser_meshblock_data[9](pn)/pmb->ruser_meshblock_data[1](pn);//metallicity
					
					Real p_age=pmb->ruser_meshblock_data[2](pn);
					
					//closest cell center
					int ip = pmb->is+(int)((pmb->ruser_meshblock_data[3](pn)-xl)/dx);
					int jp = pmb->js+(int)((pmb->ruser_meshblock_data[4](pn)-yl)/dy);
					int kp = pmb->ks+(int)((pmb->ruser_meshblock_data[5](pn)-zl)/dz);
					
					int il=ip-(int)(sigma1/dx);
					int ir=ip+(int)(sigma1/dx);
					if (il<pmb->is){
						il=pmb->is;
					}
					if (ir>pmb->ie){
						ir=pmb->ie;
					}
					
					int jl=jp-(int)(sigma1/dy);
					int jr=jp+(int)(sigma1/dy);
					if (jl<pmb->js){
						jl=pmb->js;
					}
					if (jr>pmb->je){
						jr=pmb->je;
					}
					
					int kl=kp-(int)(sigma1/dz);
					int kr=kp+(int)(sigma1/dz);
					if (kl<pmb->ks){
						kl=pmb->ks;
					}
					if (kr>pmb->ke){
						kr=pmb->ke;
					}
					
					
					//radiation pressure
					//luminosity from starburst99
					
					if (pmb->ruser_meshblock_data[2](pn)<1.0e7*3.15576e7){
						Real luminosity = 1.0e42*(particle_mass_threshold/(1.0e6*2.0e33));
						Real m_rate = (1.0+2.0)*luminosity/c;
						for (int k=kl; k<=kr; k++) {
							for (int j=jl; j<=jr; j++) {
								for (int i=il; i<=ir; i++) {
									Real x = pmb->pcoord->x1v(i);
									Real y = pmb->pcoord->x2v(j);
									Real z = pmb->pcoord->x3v(k);
									Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
									Real ine=pmb->phydro->u(IEN,k,j,i)-0.5*pmb->phydro->u(IDN,k,j,i)*(pmb->phydro->w(IVX,k,j,i)*pmb->phydro->w(IVX,k,j,i)+
																	pmb->phydro->w(IVY,k,j,i)*pmb->phydro->w(IVY,k,j,i)+
																	pmb->phydro->w(IVZ,k,j,i)*pmb->phydro->w(IVZ,k,j,i));
									Real real_temp=(ine/pmb->phydro->u(IDN,k,j,i))*m_h/(1.5*k_b);
									
									//radiation pressure only applys to cool gas where dust exists
									if (real_temp<=1.0e4){
										Real weight = (1.0/0.65)*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
										
										pmb->phydro->u(IM1,k,j,i) += weight*dt*m_rate*((x-x0)/rad);
										pmb->phydro->u(IM2,k,j,i) += weight*dt*m_rate*((y-y0)/rad);
										pmb->phydro->u(IM3,k,j,i) += weight*dt*m_rate*((z-z0)/rad);
										
										pmb->phydro->w(IVX,k,j,i) = pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
										pmb->phydro->w(IVY,k,j,i) = pmb->phydro->u(IM2,k,j,i)/pmb->phydro->u(IDN,k,j,i);
										pmb->phydro->w(IVZ,k,j,i) = pmb->phydro->u(IM3,k,j,i)/pmb->phydro->u(IDN,k,j,i);
										
										pmb->phydro->u(IEN,k,j,i) += 0.5*SQR(weight*dt*m_rate)/pmb->phydro->u(IDN,k,j,i);
									}
								}
							}
						}
					}
					
					
					
					/*
					//stellar wind
					//mass loss rate and stellar wind velocity  calculation please see https://arxiv.org/pdf/2103.12736.pdf
					//data here only account for 1000 M_solar clusters
					
					
					if (pmb->ruser_meshblock_data[2](pn)<1.0e7*3.15576e7){
						
						Real ml = 1.0e-10;//mass loss rate in solar unit
						Real wm = 0.0;//wind momentum injection rate in m_solar*km*s^-1
						
						for (int sup_num=pmb->iuser_meshblock_data[1](pn)%20;sup_num<max_sup_num;sup_num+=20){
							if ((sds[sup_num][1]*3.15576e7 >= 0.0)&&(pmb->ruser_meshblock_data[2](pn) <= sds[sup_num][1]*3.15576e7)){
								Real dml=sds[sup_num][10]+(sds[sup_num][12]-sds[sup_num][10])*(r0/0.0196); //mass injection rate
								ml+=dml;
								Real v_final=sds[sup_num][9]+(sds[sup_num][11]-sds[sup_num][9])*(r0/0.0196); //momentum injection rate
								wm+=dml*v_final;
							}
						}
						
						Real mass_loss_rate = ml*2.0e33/3.154e7;
						Real wind_velo = (wm/ml)*1000.0*100.0;
						Real wind_temp = 1.0e6;
						Real kne_inject_rate = 0.5*mass_loss_rate*SQR(wind_velo);
						Real ine_inject_rate = 1.5*mass_loss_rate*wind_temp*k_b/m_h;
						Real eng_inject_rate = ine_inject_rate+kne_inject_rate;
						
						//closest cell center
						int ip = pmb->is+(int)((pmb->ruser_meshblock_data[3](pn)-xl)/dx);
						int jp = pmb->js+(int)((pmb->ruser_meshblock_data[4](pn)-yl)/dy);
						int kp = pmb->ks+(int)((pmb->ruser_meshblock_data[5](pn)-zl)/dz);
						
						int il=ip-(int)(sigma1/dx);
						int ir=ip+(int)(sigma1/dx);
						if (il<pmb->is){
							il=pmb->is;
						}
						if (ir>pmb->ie){
							ir=pmb->ie;
						}
						
						int jl=jp-(int)(sigma1/dy);
						int jr=jp+(int)(sigma1/dy);
						if (jl<pmb->js){
							jl=pmb->js;
						}
						if (jr>pmb->je){
							jr=pmb->je;
						}
						
						int kl=kp-(int)(sigma1/dz);
						int kr=kp+(int)(sigma1/dz);
						if (kl<pmb->ks){
							kl=pmb->ks;
						}
						if (kr>pmb->ke){
							kr=pmb->ke;
						}
										
						for (int k=kl; k<=kr; k++) {
							for (int j=jl; j<=jr; j++) {
								for (int i=il; i<=ir; i++) {
									Real x = pmb->pcoord->x1v(i);
									Real y = pmb->pcoord->x2v(j);
									Real z = pmb->pcoord->x3v(k);
									Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
									
									//if (rad<=3*sigma1){
										
										Real den = (1.0/0.65)*dt*mass_loss_rate*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
										Real eng = (1.0/0.65)*dt*eng_inject_rate*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
										
										pmb->phydro->u(IDN,k,j,i) += den;
										pmb->phydro->u(IM1,k,j,i) += den*vx;//den*wind_velo*(x-x0)/rad;
										pmb->phydro->u(IM2,k,j,i) += den*vy;//den*wind_velo*(y-y0)/rad;
										pmb->phydro->u(IM3,k,j,i) += den*vz;//den*wind_velo*(y-y0)/rad;
										pmb->phydro->u(IEN,k,j,i) += eng;
										
										pmb->phydro->w(IDN,k,j,i) += den;
										pmb->phydro->w(IVX,k,j,i) = pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
										pmb->phydro->w(IVY,k,j,i) = pmb->phydro->u(IM2,k,j,i)/pmb->phydro->u(IDN,k,j,i);
										pmb->phydro->w(IVZ,k,j,i) = pmb->phydro->u(IM3,k,j,i)/pmb->phydro->u(IDN,k,j,i);
										pmb->phydro->w(IPR,k,j,i) += eng;//dt*ine_inject_rate*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
										
										if (NSCALARS>0){
											pmb->pscalars->s(0,k,j,i) += 0.1*0.013*den;//conservative passive scalars 
											pmb->pscalars->r(0,k,j,i) = pmb->pscalars->s(0,k,j,i)/pmb->phydro->w(IDN,k,j,i);
										}
									//}
								}
							}
						}
					}
					*/
					
					//supernova
					
					//Sukhbold variable supernova model
					
					//if (pmb->ruser_meshblock_data[2](pn)>1.0e5*3.15576e7){
					//	for (int sup_num=pmb->iuser_meshblock_data[1](pn)%20;sup_num<max_sup_num;sup_num+=20){
					//		if ((sds[sup_num][1]*3.15576e7 >= 0.0)&&(pmb->ruser_meshblock_data[2](pn) <= sds[sup_num][1]*3.15576e7)&&(pmb->ruser_meshblock_data[2](pn)+dt > sds[sup_num][1]*3.15576e7)){
					
					int sup_num=500*(pmb->iuser_meshblock_data[1](pn)%20)+pmb->iuser_meshblock_data[3](pn);
					if ((pmb->iuser_meshblock_data[3](pn)<500)&&(pmb->ruser_meshblock_data[2](pn)>sds[sup_num][1]*3.15576e7)){
						pmb->iuser_meshblock_data[3](pn)+=1;
						std::cout<<"Injection happens at"<<std::endl
								 <<"x0="<<x0<<std::endl
								 <<"y0="<<y0<<std::endl
								 <<"z0="<<z0<<std::endl;
						
						Real m_in=sds[sup_num][0]*1.989e33;
						Real e_in=sds[sup_num][4];
						Real metal_mass_fraction=sds[sup_num][5];
						
						std::cout<<"With parameters"<<std::endl
								 <<"m_in="<<m_in<<std::endl
								 <<"e_in="<<e_in<<std::endl;
						
										
						for (int k=kl; k<=kr; k++) {
							for (int j=jl; j<=jr; j++) {
								for (int i=il; i<=ir; i++) {
						
									//getting the coordinate
									Real x = pmb->pcoord->x1v(i);
									Real y = pmb->pcoord->x2v(j);
									Real z = pmb->pcoord->x3v(k);
									Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
									
									//Gauss part
									
									Real den = (1.0/0.65)*m_in*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
									Real ine = (1.0/0.65)*e_in*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
									
									//conservatives
									
									pmb->phydro->u(IDN,k,j,i) += den;
									pmb->phydro->u(IM1,k,j,i) += den*vx;
									pmb->phydro->u(IM2,k,j,i) += den*vy;
									pmb->phydro->u(IM3,k,j,i) += den*vz;
									pmb->phydro->u(IEN,k,j,i) += ine;
									
									//primitives
									
									pmb->phydro->w(IDN,k,j,i) += den;
									pmb->phydro->w(IVX,k,j,i) = pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
									pmb->phydro->w(IVY,k,j,i) = pmb->phydro->u(IM2,k,j,i)/pmb->phydro->u(IDN,k,j,i);
									pmb->phydro->w(IVZ,k,j,i) = pmb->phydro->u(IM3,k,j,i)/pmb->phydro->u(IDN,k,j,i);
									pmb->phydro->w(IPR,k,j,i) += ine;
									
									//passive scalar
									
									if (NSCALARS>0){
										pmb->pscalars->s(0,k,j,i) += metal_mass_fraction*den;//conservative passive scalars
										pmb->pscalars->r(0,k,j,i) = pmb->pscalars->s(0,k,j,i)/pmb->phydro->u(IDN,k,j,i);//primitive form, which is ratio
									}
								}
							}
						}
						
						//mass loss in particle
						//pmb->ruser_meshblock_data[5](pn) += -m_in;
					}
					
					
					
					
					//time-dependent method
					
					//for stellar wind
					
					
					//further modification is needed
					Real ml = 1.0e-10;//mass loss rate in solar unit
					Real wm = 0.0;//wind momentum injection rate in m_solar*km*s^-1
					
					Real dml=sds_t[(int)(p_age/(3.154e7*1.0e4))][5]+(sds_t[(int)(p_age/(3.154e7*1.0e4))][7]-sds_t[(int)(p_age/(3.154e7*1.0e4))][5])*(r0/0.0196); //mass injection rate
					ml+=dml*(particle_mass_threshold/(1.0e6*2.0e33));
					Real v_final=sds_t[(int)(p_age/(3.154e7*1.0e4))][4]+(sds_t[(int)(p_age/(3.154e7*1.0e4))][6]-sds_t[(int)(p_age/(3.154e7*1.0e4))][4])*(r0/0.0196); //momentum injection rate
					wm+=dml*(particle_mass_threshold/(1.0e6*2.0e33))*v_final;
							
					Real mass_loss_rate = ml*2.0e33/3.154e7;
					Real wind_velo = (wm/ml)*1000.0*100.0;
					Real wind_temp = 1.0e6;
					Real kne_inject_rate = 0.5*mass_loss_rate*SQR(wind_velo);
					Real ine_inject_rate = 1.5*mass_loss_rate*wind_temp*k_b/m_h;
					Real eng_inject_rate = ine_inject_rate+kne_inject_rate;
					
									
					for (int k=kl; k<=kr; k++) {
						for (int j=jl; j<=jr; j++) {
							for (int i=il; i<=ir; i++) {
								Real x = pmb->pcoord->x1v(i);
								Real y = pmb->pcoord->x2v(j);
								Real z = pmb->pcoord->x3v(k);
								Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
								
								//stellar wind
								Real den = (1.0/0.65)*dt*mass_loss_rate*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
								Real eng = (1.0/0.65)*dt*eng_inject_rate*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
								
								pmb->phydro->u(IDN,k,j,i) += den;
								pmb->phydro->u(IM1,k,j,i) += den*vx;//den*wind_velo*(x-x0)/rad;
								pmb->phydro->u(IM2,k,j,i) += den*vy;//den*wind_velo*(y-y0)/rad;
								pmb->phydro->u(IM3,k,j,i) += den*vz;//den*wind_velo*(y-y0)/rad;
								pmb->phydro->u(IEN,k,j,i) += eng;
								
								pmb->phydro->w(IDN,k,j,i) += den;
								pmb->phydro->w(IVX,k,j,i) = pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
								pmb->phydro->w(IVY,k,j,i) = pmb->phydro->u(IM2,k,j,i)/pmb->phydro->u(IDN,k,j,i);
								pmb->phydro->w(IVZ,k,j,i) = pmb->phydro->u(IM3,k,j,i)/pmb->phydro->u(IDN,k,j,i);
								pmb->phydro->w(IPR,k,j,i) += eng;//dt*ine_inject_rate*exp(-(rad*rad)/(2*sigma1*sigma1))/(sigma1*sigma1*sigma1*2*M_PI*std::sqrt(2*M_PI));
								
								if (NSCALARS>0){
									pmb->pscalars->s(0,k,j,i) += 0.1*0.013*den;//conservative passive scalars 
									pmb->pscalars->r(0,k,j,i) = pmb->pscalars->s(0,k,j,i)/pmb->phydro->w(IDN,k,j,i);
								}
							}
						}
					}
					
					//mass loss in particle
					//pmb->ruser_meshblock_data[5](pn) += -dt*mass_loss_rate;
				}
			}
		}
		
		
		
		//==========part 1 initialization==========
		//finished
		//init mpi parameters
		//rank of the mesh
		
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
		//std::cout<<"current rank num is "<<rank<<std::endl;
		iuser_mesh_data[0](0)=rank;
		
		int node_num;
		MPI_Comm_size(MPI_COMM_WORLD,&node_num);
		
		//==========part 2 merging==========
		//finished & tested
		//signature of type of eliminated particles should be set to a certain value
		//-2 to be exact
		//after finishing this part mpi communication section should be changed according, to filter 
		//whenever the control volume of two sink particles overlaps, they merges
		
		
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			int particle_num_mb=pmb->iuser_meshblock_data[0](0);
			
			for (int n=0; n<particle_num_mb-1; n++){
				Real dx = pmb->pcoord->dx1v(pmb->ie);
				
				if (pmb->iuser_meshblock_data[2](n)==1){
					Real x = pmb->ruser_meshblock_data[3](n);
					Real y = pmb->ruser_meshblock_data[4](n);
					Real z = pmb->ruser_meshblock_data[5](n);
					Real om = pmb->ruser_meshblock_data[1](n);
					Real omm = pmb->ruser_meshblock_data[9](n);
					
					for (int m=n+1; m<particle_num_mb; m++){
						if (pmb->iuser_meshblock_data[2](m)==1){
							Real px = pmb->ruser_meshblock_data[3](m);
							Real py = pmb->ruser_meshblock_data[4](m);
							Real pz = pmb->ruser_meshblock_data[5](m);
							Real pm = pmb->ruser_meshblock_data[1](m);
							Real pmm = pmb->ruser_meshblock_data[9](m);
							
							Real rad = std::sqrt(SQR(x-px) + SQR(y-py) + SQR(z-pz));
							
							if (rad<5*dx){
								//set latter particle's type to eliminated(-2)
								pmb->iuser_meshblock_data[2](m) = -2;
								
								//position
								x = 0.5*x+0.5*px;
								y = 0.5*y+0.5*py;
								z = 0.5*z+0.5*pz;
								
								//velocity
								pmb->ruser_meshblock_data[6](n) = (pm*pmb->ruser_meshblock_data[6](m)
																+om*pmb->ruser_meshblock_data[6](n))
																/(pm+om);
								pmb->ruser_meshblock_data[7](n) = (pm*pmb->ruser_meshblock_data[7](m)
																+om*pmb->ruser_meshblock_data[7](n))
																/(pm+om);
								pmb->ruser_meshblock_data[8](n) = (pm*pmb->ruser_meshblock_data[8](m)
																+om*pmb->ruser_meshblock_data[8](n))
																/(pm+om);
								
								//mass
								om += pm;
								
								//metal mass
								omm += pmm;
							}
						}
					}
					pmb->ruser_meshblock_data[3](n) = x;
					pmb->ruser_meshblock_data[4](n) = y;
					pmb->ruser_meshblock_data[5](n) = z;
					pmb->ruser_meshblock_data[1](n) = om;
					pmb->ruser_meshblock_data[9](n) = omm;
				}
			}
		}
		
		
		
		//==========part 3 particle creation==========
		//finished
		
		
		//creation of new particles
		int particle_create_num_send=0;
		int particle_create_num_all_rec[node_num];
		
		//create new particle and set new particle singature to -1
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			LogicalLocation &loc = pmb->loc;
			
			//getting coord data because threshold density depends on cell size
			Real dx = pmb->pcoord->dx1v(pmb->ie);
			Real dy = pmb->pcoord->dx2v(pmb->je);
			Real dz = pmb->pcoord->dx3v(pmb->ke);
			Real xl = pmb->pcoord->x1v(pmb->is)-0.5*dx;
			Real xr = pmb->pcoord->x1v(pmb->ie)+0.5*dx;
			Real yl = pmb->pcoord->x2v(pmb->js)-0.5*dy;
			Real yr = pmb->pcoord->x2v(pmb->je)+0.5*dy;
			Real zl = pmb->pcoord->x3v(pmb->ks)-0.5*dz;
			Real zr = pmb->pcoord->x3v(pmb->ke)+0.5*dz;
			
			int sink_num=0;
			for (int pn=0;pn<pmb->iuser_meshblock_data[0](0);pn++){
				if (pmb->iuser_meshblock_data[2](pn)==1){
					sink_num++;
				}
			}
			
			if (sink_num>=3){
				//do nothing to avoid creating new sink particle
			}
			else{
				//sink particles are only created under strict conditions, all three need to be met at the same time for
				//a sink particle to form in the cell center
				//1 density threshold(jeans mass within contrl volume check)
				//2 converging flow (\nabla \cdot v < 0)
				//3 gravitationally bounded
				
				//pmb->pgrav->phi(k,j,i)
				
				int max_p_num=pmb->iuser_meshblock_data[1].GetSize();
				int overloadbool=0;
				
				Real d_throcs2=(M_PI/16)/(6.6743e-8*SQR(dx));
				for (int k=pmb->ks; k<=pmb->ke; k++) {
					for (int j=pmb->js; j<=pmb->je; j++) {
						for (int i=pmb->is; i<=pmb->ie; i++) {
							//particle overload check
							if (pmb->iuser_meshblock_data[0](0)>=max_p_num){
								overloadbool=1;
								break;
							}
							
							//position
							Real x = pmb->pcoord->x1v(i);
							Real y = pmb->pcoord->x2v(j);
							Real z = pmb->pcoord->x3v(k);
							//assume gamma=1.6667 for monoatomic gas
							Real c_s2=1.6667*pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
							//rho*c_s^-2 threshold check
							int dencheck=0;
							if (pmb->phydro->w(IDN,k,j,i)/c_s2>d_throcs2){
								dencheck =1 ;
							}
							
							//distance check
							int discheck=1;
							if (dencheck==1){
								//control volume check
								//threshold distance = 5 dx
								for (int pn=0;pn<pmb->iuser_meshblock_data[0](0);pn++){
									if (((pmb->iuser_meshblock_data[2](pn)==1)||(pmb->iuser_meshblock_data[2](pn)==2)) 
									//if ((pmb->iuser_meshblock_data[2](pn)==1) 
										&& (SQR(pmb->ruser_meshblock_data[3](pn)-x) + SQR(pmb->ruser_meshblock_data[4](pn)-y) + SQR(pmb->ruser_meshblock_data[5](pn)-z) < SQR(5*dx))){
										discheck=0;
										break;
									}
								}
							}
							
							//flow converging check
							int fccheck=0;
							if ((discheck==1)&&(dencheck==1)){
								if ((pmb->phydro->w(IVX,k,j,i+1)-pmb->phydro->w(IVX,k,j,i-1))/dx+
									(pmb->phydro->w(IVY,k,j+1,i)-pmb->phydro->w(IVY,k,j-1,i))/dy+
									(pmb->phydro->w(IVZ,k+1,j,i)-pmb->phydro->w(IVZ,k-1,j,i))/dz<0.0){
										fccheck=1;
									}
							}
							
							//gravitationally bound
							//is not necessarily valid with fast moving clouds
							
							/*
							int gbcheck=0;
							if ((discheck==1)&&(dencheck==1)&&(fccheck==1)){
								Real eng_tot=0.0;
								for (int dk=-2;dk<=2;dk++){
									for (int dj=-2;dj<=2;dj++){
										for (int di=-2;di<=2;di++){
											Real rad=std::sqrt(SQR(dk) + SQR(dj) + SQR(di));
											if (rad<=2.5){
												//devided into three parts:
												//1 old star fraction(grav potential background)
												//2 gas grav potential
												//3 thermal & kinetic energy
												//2022.3.2 change: only consider thermal and gas potential
												eng_tot += pmb->pgrav->phi(k+dk,j+dj,i+di)*pmb->phydro->u(IDN,k+dk,j+dj,i+di)+
															//pmb->ruser_meshblock_data[0](k+dk,j+dj,i+di)*pmb->phydro->u(IDN,k+dk,j+dj,i+di)+
															pmb->phydro->w(IPR,k+dk,j+dj,i+di);
											}
											
											//4 particles within the same meshblock
										}
									}
								}
								if (eng_tot<0){
									gbcheck=1;
								}
								
							}
							*/
							
							//local gravitational potential minimum
							int gbcheck=1;
							int neighbor_over_dens=0;
							if ((discheck==1)&&(dencheck==1)&&(fccheck==1)){
								for (int dk=-2;dk<=2;dk++){
									for (int dj=-2;dj<=2;dj++){
										for (int di=-2;di<=2;di++){
											Real rad=std::sqrt(SQR(dk) + SQR(dj) + SQR(di));
											if (rad<=2.5){
												Real c_s2_neighbor=1.6667*pmb->phydro->w(IPR,k+dk,j+dj,i+di)/pmb->phydro->w(IDN,k+dk,j+dj,i+di);
												if ((pmb->phydro->w(IDN,k+dk,j+dj,i+di)/c_s2_neighbor>d_throcs2)&&(pmb->pgrav->phi(k+dk,j+dj,i+di)<pmb->pgrav->phi(k,j,i))){
													gbcheck=0;
												}
												if (pmb->phydro->w(IDN,k+dk,j+dj,i+di)/c_s2_neighbor>d_throcs2){
													neighbor_over_dens++;
												}
											}
											if (gbcheck==0){
												break;
											}
										}
										if (gbcheck==0){
											break;
										}
									}
									if (gbcheck==0){
										break;
									}
								}
							}
							if (neighbor_over_dens<7){
								gbcheck==0;
							}
							
							
							//gravitational poisson equation
							/*
							int gbcheck=1;
							if ((discheck==1)&&(dencheck==1)&&(fccheck==1)){
								if ((pmb->pgrav->phi(k,j,i+1)+pmb->pgrav->phi(k,j,i-1)-2*pmb->pgrav->phi(k,j,i)<0)||
									(pmb->pgrav->phi(k,j+1,i)+pmb->pgrav->phi(k,j-1,i)-2*pmb->pgrav->phi(k,j,i)<0)||
									(pmb->pgrav->phi(k+1,j,i)+pmb->pgrav->phi(k-1,j,i)-2*pmb->pgrav->phi(k,j,i)<0)){
									gbcheck=0;
								}
							}
							*/
							
							if ((discheck==1)&&(dencheck==1)&&(fccheck==1)&&(gbcheck==1)){
								particle_create_num_send++;
								//int data
								pmb->iuser_meshblock_data[0](0)+=1;
								pmb->iuser_meshblock_data[1](pmb->iuser_meshblock_data[0](0)-1)=-1;//set particle signature to unnamed
								pmb->iuser_meshblock_data[2](pmb->iuser_meshblock_data[0](0)-1)=1;//set particle type to sink
								pmb->iuser_meshblock_data[3](pmb->iuser_meshblock_data[0](0)-1)=0;
								//real data
								//mass
								//set to a relatively small but non-zero value
								pmb->ruser_meshblock_data[1](pmb->iuser_meshblock_data[0](0)-1)=1.0e5;
								pmb->ruser_meshblock_data[2](pmb->iuser_meshblock_data[0](0)-1)=0.0;
								pmb->ruser_meshblock_data[3](pmb->iuser_meshblock_data[0](0)-1)=x;
								pmb->ruser_meshblock_data[4](pmb->iuser_meshblock_data[0](0)-1)=y;
								pmb->ruser_meshblock_data[5](pmb->iuser_meshblock_data[0](0)-1)=z;
								pmb->ruser_meshblock_data[6](pmb->iuser_meshblock_data[0](0)-1)=0.0;
								pmb->ruser_meshblock_data[7](pmb->iuser_meshblock_data[0](0)-1)=0.0;
								pmb->ruser_meshblock_data[8](pmb->iuser_meshblock_data[0](0)-1)=0.0;
								pmb->ruser_meshblock_data[9](pmb->iuser_meshblock_data[0](0)-1)=0.0;
							}
						}
						if (overloadbool==1){
							break;
						}
					}
					if (overloadbool==1){
						break;
					}
				}
			}
		}
		
		//use allgather to determine how many particles are created in total
		//and name them accordingly
		MPI_Allgather(&particle_create_num_send,1,MPI_INT,&particle_create_num_all_rec,1,MPI_INT,MPI_COMM_WORLD);
		if (particle_create_num_send>0){
			int num_start=iuser_mesh_data[1](0);
			for (int n=0; n<iuser_mesh_data[0](0); n++){
				num_start+=particle_create_num_all_rec[n];
			}
			for (int bn=0; bn<nblocal; ++bn) {
				MeshBlock *pmb = my_blocks(bn);
				LogicalLocation &loc = pmb->loc;
				int particle_num_mb=pmb->iuser_meshblock_data[0](0);
				if (particle_num_mb>0){
					for (int n=0; n<particle_num_mb; n++){
						if (pmb->iuser_meshblock_data[1](n)==-1){
							num_start+=1;
							pmb->iuser_meshblock_data[1](n)=num_start;
						}
					}
				}
			}
		}
		//update total name(int signature) used
		for (int n=0; n<node_num; n++){
			iuser_mesh_data[1](0)+=particle_create_num_all_rec[n];
		}
		
		
		//==========part 4 accretion==========
		//finished
		
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			LogicalLocation &loc = pmb->loc;
			
			Real dx = pmb->pcoord->dx1v(pmb->ie);
			Real dy = pmb->pcoord->dx2v(pmb->je);
			Real dz = pmb->pcoord->dx3v(pmb->ke);
			Real xl = pmb->pcoord->x1v(pmb->is)-0.5*dx;
			Real xr = pmb->pcoord->x1v(pmb->ie)+0.5*dx;
			Real yl = pmb->pcoord->x2v(pmb->js)-0.5*dy;
			Real yr = pmb->pcoord->x2v(pmb->je)+0.5*dy;
			Real zl = pmb->pcoord->x3v(pmb->ks)-0.5*dz;
			Real zr = pmb->pcoord->x3v(pmb->ke)+0.5*dz;
			
			Real d_throcs2=(M_PI/16)/(6.6743e-8*SQR(dx));
			
			for (int pn=0;pn<pmb->iuser_meshblock_data[0](0);pn++){
				if (pmb->iuser_meshblock_data[2](pn)==1){
					//closest cell center
					int i = pmb->is+(int)((pmb->ruser_meshblock_data[3](pn)-xl)/dx);
					int j = pmb->js+(int)((pmb->ruser_meshblock_data[4](pn)-yl)/dy);
					int k = pmb->ks+(int)((pmb->ruser_meshblock_data[5](pn)-zl)/dz);
					
					//accretion
					
					for (int dk=-2;dk<=2;dk++){
						for (int dj=-2;dj<=2;dj++){
							for (int di=-2;di<=2;di++){
								Real rad=std::sqrt(SQR(dk) + SQR(dj) + SQR(di));
								if (rad<=2.5){
									Real c_s2=1.6667*pmb->phydro->w(IPR,k+dk,j+dj,i+di)/pmb->phydro->w(IDN,k+dk,j+dj,i+di);
									if (pmb->phydro->w(IDN,k+dk,j+dj,i+di)/c_s2>d_throcs2){
										//update particle information
										Real dm = dx*dy*dz*(pmb->phydro->w(IDN,k+dk,j+dj,i+di)-d_throcs2*c_s2);
										//v1-v3
										pmb->ruser_meshblock_data[6](pn) = (pmb->ruser_meshblock_data[6](pn)*pmb->ruser_meshblock_data[1](pn)+pmb->phydro->w(IVX,k+dk,j+dj,i+di)*dm)/(pmb->ruser_meshblock_data[1](pn)+dm);
										pmb->ruser_meshblock_data[7](pn) = (pmb->ruser_meshblock_data[7](pn)*pmb->ruser_meshblock_data[1](pn)+pmb->phydro->w(IVY,k+dk,j+dj,i+di)*dm)/(pmb->ruser_meshblock_data[1](pn)+dm);
										pmb->ruser_meshblock_data[8](pn) = (pmb->ruser_meshblock_data[8](pn)*pmb->ruser_meshblock_data[1](pn)+pmb->phydro->w(IVZ,k+dk,j+dj,i+di)*dm)/(pmb->ruser_meshblock_data[1](pn)+dm);
										//mass
										pmb->ruser_meshblock_data[1](pn) += dm;
										//metal mass
										if (NSCALARS>0){
											pmb->ruser_meshblock_data[9](pn) += pmb->pscalars->r(0,k,j,i)*dm;
										}
										
										//update gridcell information
										if (((k+dk>=pmb->ks)&&(k+dk<=pmb->ke))&&
											((j+dj>=pmb->js)&&(j+dj<=pmb->je))&&
											((i+di>=pmb->is)&&(i+di<=pmb->ie))){
											Real frac=d_throcs2*c_s2/pmb->phydro->w(IDN,k+dk,j+dj,i+di);
											//conservatives
											pmb->phydro->u(IM1,k+dk,j+dj,i+di) = pmb->phydro->u(IM1,k+dk,j+dj,i+di)*frac;
											pmb->phydro->u(IM2,k+dk,j+dj,i+di) = pmb->phydro->u(IM2,k+dk,j+dj,i+di)*frac;
											pmb->phydro->u(IM3,k+dk,j+dj,i+di) = pmb->phydro->u(IM3,k+dk,j+dj,i+di)*frac;
											pmb->phydro->u(IEN,k+dk,j+dj,i+di) = pmb->phydro->u(IEN,k+dk,j+dj,i+di)*frac;
											pmb->phydro->u(IDN,k+dk,j+dj,i+di) = d_throcs2*c_s2;
											//primitives
											pmb->phydro->w(IPR,k+dk,j+dj,i+di) = pmb->phydro->w(IPR,k+dk,j+dj,i+di)*frac;
											pmb->phydro->w(IDN,k+dk,j+dj,i+di) = d_throcs2*c_s2;
											
											if (NSCALARS>0){
												pmb->pscalars->s(0,k,j,i) =pmb->pscalars->s(0,k,j,i)*frac;//conservative passive scalars
												//pmb->pscalars->r(0,k,j,i) does not change
											}
										}
									}
								}
							}
						}
					}
					
					//check if larger than mass threshold
					//if so convert to star particles
					//if (pmb->ruser_meshblock_data[1](pn)>=5.0e4*2.0e33){
					if (pmb->ruser_meshblock_data[1](pn)>=particle_mass_threshold){
						//age
						//pmb->ruser_meshblock_data[2](pn) = 0.0;
						//here age can be set to negative free fall time so to better estimate the time scale of star formation, and make the result
						//independent to neither the scale of the control volume/ mass of the final cluster
						pmb->ruser_meshblock_data[2](pn) = 0.0;//-std::sqrt(3.0*PI/(32.0*g*pmb->ruser_meshblock_data[1](pn)/(4*PI*std::pow(2.5*3.0e21/256,3)/3)));
						//particle type
						pmb->iuser_meshblock_data[2](pn) = 2;
						std::cout<<"Star formed!"<<std::endl;
						star_bool=1;
					}
				}
			}
		}
		
		
		//==========part 5 MPI communication==========
		//finished
		//preprocessing data to be sent later
		//in this part data will be intergrated to the next timestep and stored in two dynamic arrays
		//num of particles in this mesh
		//three data sets to be sent during mpi all-all communication
		
		//A drift-kick method is adopted here
		//only update position, mass(during accretion), age before redistributing each particle
		//velocity will only be calculated on its new position
		//it can be more complicated by writing a kdk scheme
		//2022-3-11
		
		int particle_num_m_send = 0;
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			Real sar = 1.0*sigma1;
			Real dx = pmb->pcoord->dx1v(pmb->ie);
			Real dy = pmb->pcoord->dx2v(pmb->je);
			Real dz = pmb->pcoord->dx3v(pmb->ke);
			Real xl = pmb->pcoord->x1v(pmb->is)-0.5*dx;
			Real xr = pmb->pcoord->x1v(pmb->ie)+0.5*dx;
			Real yl = pmb->pcoord->x2v(pmb->js)-0.5*dy;
			Real yr = pmb->pcoord->x2v(pmb->je)+0.5*dy;
			Real zl = pmb->pcoord->x3v(pmb->ks)-0.5*dz;
			Real zr = pmb->pcoord->x3v(pmb->ke)+0.5*dz;
			LogicalLocation &loc = pmb->loc;
			int particle_num_mb=pmb->iuser_meshblock_data[0](0);
			if (particle_num_mb>0){
				for (int n=0; n<particle_num_mb; n++){
					if ((pmb->iuser_meshblock_data[2](n)==0)||(pmb->iuser_meshblock_data[2](n)==1)||(pmb->iuser_meshblock_data[2](n)==2)){
						//drift
						pmb->ruser_meshblock_data[2](n)+=dt;
						pmb->ruser_meshblock_data[3](n)+=dt*pmb->ruser_meshblock_data[6](n);
						pmb->ruser_meshblock_data[4](n)+=dt*pmb->ruser_meshblock_data[7](n);
						pmb->ruser_meshblock_data[5](n)+=dt*pmb->ruser_meshblock_data[8](n);
						//determine whether a particle will be sent
						if (((pmb->ruser_meshblock_data[3](n) < xl + sar) || (pmb->ruser_meshblock_data[3](n) > xr - sar))||
						    ((pmb->ruser_meshblock_data[4](n) < yl + sar) || (pmb->ruser_meshblock_data[4](n) > yr - sar))||
						    ((pmb->ruser_meshblock_data[5](n) < zl + sar) || (pmb->ruser_meshblock_data[5](n) > zr - sar))){
							particle_num_m_send+=1;
						}
					}
				}
			}
		}
		
		//max_particle_num_m is used to determine the size of array to receive particle data
		int max_particle_num_m=0;
		//MPI_Allreduce(&particle_num_m_send,&max_particle_num_m,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
		
		int particle_num_all_rec[node_num];
		MPI_Allgather(&particle_num_m_send,1,MPI_INT,&particle_num_all_rec,1,MPI_INT,MPI_COMM_WORLD);
		for (int n=0; n<node_num; n++){
			if (particle_num_all_rec[n]>max_particle_num_m){
				max_particle_num_m=particle_num_all_rec[n];
			}
		}
		
		//tot meshblock number in a mesh
		int mb_num_m_send = nblocal;
		
		//max_nb_num_m is used to determine the size of array to receive coarse meshblock data
		int max_mb_num_m=0;
		//MPI_Allreduce(&mb_num_m_send,&max_mb_num_m,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
		
		int mb_num_all_rec[node_num];
		MPI_Allgather(&mb_num_m_send,1,MPI_INT,&mb_num_all_rec,1,MPI_INT,MPI_COMM_WORLD);
		for (int n=0; n<node_num; n++){
			if (mb_num_all_rec[n]>max_mb_num_m){
				max_mb_num_m=mb_num_all_rec[n];
			}
		}
		
		
		//init send data set
		int idata_m_send[3*max_particle_num_m];
		Real rdata_m_send[9*max_particle_num_m];
		Real rdata_ptotal_send[4*max_mb_num_m];
		for (int i=0; i<4*nblocal;i++){
			rdata_ptotal_send[i] = 0.0;
		}
		for (int i=0; i<3*max_particle_num_m;i++){
			idata_m_send[i] = -1000;
		}
		
		int pn_counter=0;
		if (particle_num_m_send>0){
			for (int bn=0; bn<nblocal; ++bn) {
				MeshBlock *pmb = my_blocks(bn);
				LogicalLocation &loc = pmb->loc;
				//num of particles in this meshblock
				int particle_num_mb=pmb->iuser_meshblock_data[0](0);
				if (particle_num_mb>0){
					//coordinate data of the meshblock
					Real sar = 1.0*sigma1;
					Real dx = pmb->pcoord->dx1v(pmb->ie);
					Real dy = pmb->pcoord->dx2v(pmb->je);
					Real dz = pmb->pcoord->dx3v(pmb->ke);
					Real xl = pmb->pcoord->x1v(pmb->is)-0.5*dx;
					Real xr = pmb->pcoord->x1v(pmb->ie)+0.5*dx;
					Real yl = pmb->pcoord->x2v(pmb->js)-0.5*dy;
					Real yr = pmb->pcoord->x2v(pmb->je)+0.5*dy;
					Real zl = pmb->pcoord->x3v(pmb->ks)-0.5*dz;
					Real zr = pmb->pcoord->x3v(pmb->ke)+0.5*dz;
					
					//particle_num_m+=particle_num_mb;
					for (int n=0; n<particle_num_mb; n++){
						if ((pmb->iuser_meshblock_data[2](n)==0)||(pmb->iuser_meshblock_data[2](n)==1)||(pmb->iuser_meshblock_data[2](n)==2)){
							if (pmb->iuser_meshblock_data[2](n)==0){
								//fluid particle
								//locating the neariest cell
								int i = pmb->is+(int)((pmb->ruser_meshblock_data[3](n)-xl)/dx);
								int j = pmb->js+(int)((pmb->ruser_meshblock_data[4](n)-yl)/dy);
								int k = pmb->ks+(int)((pmb->ruser_meshblock_data[5](n)-zl)/dz);
								//v1-v3
								pmb->ruser_meshblock_data[6](n) = pmb->phydro->w(IVX,k,j,i);
								pmb->ruser_meshblock_data[7](n) = pmb->phydro->w(IVY,k,j,i);
								pmb->ruser_meshblock_data[8](n) = pmb->phydro->w(IVZ,k,j,i);
							}
							
							//calculate the total particle mass and c.g. of particles contained in each meshblock
							rdata_ptotal_send[4*bn+1] = (rdata_ptotal_send[4*bn+1]*rdata_ptotal_send[4*bn] + pmb->ruser_meshblock_data[3](n)*pmb->ruser_meshblock_data[1](n))/
														(                          rdata_ptotal_send[4*bn] +                                 pmb->ruser_meshblock_data[1](n));
							rdata_ptotal_send[4*bn+2] = (rdata_ptotal_send[4*bn+2]*rdata_ptotal_send[4*bn] + pmb->ruser_meshblock_data[4](n)*pmb->ruser_meshblock_data[1](n))/
														(                          rdata_ptotal_send[4*bn] +                                 pmb->ruser_meshblock_data[1](n));
							rdata_ptotal_send[4*bn+3] = (rdata_ptotal_send[4*bn+3]*rdata_ptotal_send[4*bn] + pmb->ruser_meshblock_data[5](n)*pmb->ruser_meshblock_data[1](n))/
														(                          rdata_ptotal_send[4*bn] +                                 pmb->ruser_meshblock_data[1](n));
							rdata_ptotal_send[4*bn] += pmb->ruser_meshblock_data[1](n);
							
							
							if (((pmb->ruser_meshblock_data[3](n) < xl + sar) || (pmb->ruser_meshblock_data[3](n) > xr - sar))||
								((pmb->ruser_meshblock_data[4](n) < yl + sar) || (pmb->ruser_meshblock_data[4](n) > yr - sar))||
								((pmb->ruser_meshblock_data[5](n) < zl + sar) || (pmb->ruser_meshblock_data[5](n) > zr - sar))){
								//signature
								//std::cout<<pmb->iuser_meshblock_data[1](n)<<" send!"<<std::endl;
								idata_m_send[3*pn_counter]   = pmb->iuser_meshblock_data[1](n);
								//std::cout<<"before send signature "<<pmb->iuser_meshblock_data[1](n)<<std::endl;
								//particle type
								idata_m_send[3*pn_counter+1] = pmb->iuser_meshblock_data[2](n);
								//SN number
								idata_m_send[3*pn_counter+2] = pmb->iuser_meshblock_data[3](n);
								//set particle type to be eliminated later
								pmb->iuser_meshblock_data[2](n) = -2;
								//age
								rdata_m_send[9*pn_counter+1] = pmb->ruser_meshblock_data[2](n);
								//x1-x3
								rdata_m_send[9*pn_counter+2] = pmb->ruser_meshblock_data[3](n);
								rdata_m_send[9*pn_counter+3] = pmb->ruser_meshblock_data[4](n);
								rdata_m_send[9*pn_counter+4] = pmb->ruser_meshblock_data[5](n);
								//mass
								rdata_m_send[9*pn_counter]   = pmb->ruser_meshblock_data[1](n);
								//metal mass
								rdata_m_send[9*pn_counter+8] = pmb->ruser_meshblock_data[9](n);
								//v1-v3
								rdata_m_send[9*pn_counter+5] = pmb->ruser_meshblock_data[6](n);
								rdata_m_send[9*pn_counter+6] = pmb->ruser_meshblock_data[7](n);
								rdata_m_send[9*pn_counter+7] = pmb->ruser_meshblock_data[8](n);
								
								pn_counter++;
								
								if (pn_counter>particle_num_m_send){
									std::cout<<"???"<<std::endl;
									std::cout<<"pn_counter = "<<pn_counter<<std::endl;
									std::cout<<"pn_counter_init = "<<particle_num_m_send<<std::endl;
								}
							}
						}
					}
				}
			}
		}
		
		//eliminate particles no longer hosted by current meshblock
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			LogicalLocation &loc = pmb->loc;
			int n = 0;
			while (n<pmb->iuser_meshblock_data[0](0)){
				if ((pmb->iuser_meshblock_data[2](n)==0)||(pmb->iuser_meshblock_data[2](n)==1)||(pmb->iuser_meshblock_data[2](n)==2)){
					n++;
				}
				else{
					pmb->iuser_meshblock_data[0](0) = pmb->iuser_meshblock_data[0](0)-1;
					int nc = n;
					while (nc<pmb->iuser_meshblock_data[0](0)){
						//signature
						pmb->iuser_meshblock_data[1](nc)=pmb->iuser_meshblock_data[1](nc+1);
						//particle type
						pmb->iuser_meshblock_data[2](nc)=pmb->iuser_meshblock_data[2](nc+1);
						//SN number
						pmb->iuser_meshblock_data[3](nc)=pmb->iuser_meshblock_data[3](nc+1);
						//mass
						pmb->ruser_meshblock_data[1](nc)=pmb->ruser_meshblock_data[1](nc+1);
						//age
						pmb->ruser_meshblock_data[2](nc)=pmb->ruser_meshblock_data[2](nc+1);
						//x1-x3
						pmb->ruser_meshblock_data[3](nc)=pmb->ruser_meshblock_data[3](nc+1);
						pmb->ruser_meshblock_data[4](nc)=pmb->ruser_meshblock_data[4](nc+1);
						pmb->ruser_meshblock_data[5](nc)=pmb->ruser_meshblock_data[5](nc+1);
						//v1-v3
						pmb->ruser_meshblock_data[6](nc)=pmb->ruser_meshblock_data[6](nc+1);
						pmb->ruser_meshblock_data[7](nc)=pmb->ruser_meshblock_data[7](nc+1);
						pmb->ruser_meshblock_data[8](nc)=pmb->ruser_meshblock_data[8](nc+1);
						//metal mass
						if (NSCALARS>0){
							pmb->ruser_meshblock_data[9](nc)=pmb->ruser_meshblock_data[9](nc+1);
						}
						nc++;
					}
				}
			}
		}
		
		
		//mpi communication
		//here data will be pass through each and every node
		int idata_all_rec[node_num*max_particle_num_m*3];
		Real rdata_all_rec[node_num*max_particle_num_m*9];
		Real rdata_ptotal_rec[node_num*max_mb_num_m*4];
		MPI_Allgather(&idata_m_send ,max_particle_num_m*3 ,MPI_INT ,&idata_all_rec ,max_particle_num_m*3 ,MPI_INT ,MPI_COMM_WORLD);
		MPI_Allgather(&rdata_m_send ,max_particle_num_m*9 ,MPI_DOUBLE ,&rdata_all_rec ,max_particle_num_m*9 ,MPI_DOUBLE ,MPI_COMM_WORLD);
		MPI_Allgather(&rdata_ptotal_send ,max_mb_num_m*4 ,MPI_DOUBLE ,&rdata_ptotal_rec ,max_mb_num_m*4 ,MPI_DOUBLE ,MPI_COMM_WORLD);
				
		//recollect data belonging to each mesh
		
		for (int bn=0; bn<nblocal; ++bn) {
			MeshBlock *pmb = my_blocks(bn);
			LogicalLocation &loc = pmb->loc;
			Real dx = pmb->pcoord->dx1v(pmb->ie);
			Real dy = pmb->pcoord->dx2v(pmb->je);
			Real dz = pmb->pcoord->dx3v(pmb->ke);
			Real xl = pmb->pcoord->x1v(pmb->is)-0.5*dx;
			Real xr = pmb->pcoord->x1v(pmb->ie)+0.5*dx;
			Real yl = pmb->pcoord->x2v(pmb->js)-0.5*dy;
			Real yr = pmb->pcoord->x2v(pmb->je)+0.5*dy;
			Real zl = pmb->pcoord->x3v(pmb->ks)-0.5*dz;
			Real zr = pmb->pcoord->x3v(pmb->ke)+0.5*dz;
			Real xc = 0.5*(xl+xr);
			Real yc = 0.5*(yl+yr);
			Real zc = 0.5*(zl+zr);
			Real g1 = 0.0;
			Real g2 = 0.0;
			Real g3 = 0.0;
			
						
			//supernova affected radius
			Real sar = 1.0*sigma1;
			
			//calculate gravity potential
			for (int n=0; n<node_num; n++){
				for (int m=0; m<mb_num_all_rec[n]; m++){
					Real px = rdata_ptotal_rec[n*max_mb_num_m*4+m*4+1];
					Real py = rdata_ptotal_rec[n*max_mb_num_m*4+m*4+2];
					Real pz = rdata_ptotal_rec[n*max_mb_num_m*4+m*4+3];
					if ((px<xl)||(px>=xr)||
						(py<yl)||(py>=yr)||
						(pz<zl)||(pz>=zr)){
						Real rad = std::sqrt(SQR(xc-px) + SQR(yc-py) + SQR(zc-pz));
						g1 += g*rdata_ptotal_rec[n*max_mb_num_m*4+m*4]*((xc-px)/rad)/SQR(rad);
						g2 += g*rdata_ptotal_rec[n*max_mb_num_m*4+m*4]*((yc-py)/rad)/SQR(rad);
						g3 += g*rdata_ptotal_rec[n*max_mb_num_m*4+m*4]*((zc-pz)/rad)/SQR(rad);
					}
				}
			}
			
			int max_p_num=100;
			
			//recollect
			for (int n=0; n<node_num; n++){
				if (particle_num_all_rec[n]>0){
					for (int m=0; m<particle_num_all_rec[n]; m++){
						if (idata_all_rec[n*max_particle_num_m*3+m*3+1] >= 0){
							if ((rdata_all_rec[n*max_particle_num_m*9+m*9+2]>=xl)&(rdata_all_rec[n*max_particle_num_m*9+m*9+2]<xr)&
								(rdata_all_rec[n*max_particle_num_m*9+m*9+3]>=yl)&(rdata_all_rec[n*max_particle_num_m*9+m*9+3]<yr)&
								(rdata_all_rec[n*max_particle_num_m*9+m*9+4]>=zl)&(rdata_all_rec[n*max_particle_num_m*9+m*9+4]<zr)){
								//num of particles ++
								pmb->iuser_meshblock_data[0](0)++;
								int nc=pmb->iuser_meshblock_data[0](0)-1;
								
									
								if (pmb->iuser_meshblock_data[0](0)>=max_p_num){
									std::cout<<"Error occour at rank "<<iuser_mesh_data[0](0)<<std::endl;
									std::cout<<"With particle signature "<<idata_all_rec[n*max_particle_num_m*3+m*3]<<std::endl;
									std::stringstream overload_msg;
									overload_msg<<"Fatal Error: particle overloaded"<<std::endl;
									ATHENA_ERROR(overload_msg);
								}
								
								//signature
								//std::cout<<idata_all_rec[n*max_particle_num_m*2+m*2]<<" reveived!"<<std::endl;
								pmb->iuser_meshblock_data[1](nc)=idata_all_rec[n*max_particle_num_m*3+m*3];
								//std::cout<<"received signature "<<idata_all_rec[n*max_particle_num_m*2+m*2]<<std::endl;
								//particle type
								pmb->iuser_meshblock_data[2](nc)=idata_all_rec[n*max_particle_num_m*3+m*3+1];
								//SN number
								pmb->iuser_meshblock_data[3](nc)=idata_all_rec[n*max_particle_num_m*3+m*3+2];
								//mass
								pmb->ruser_meshblock_data[1](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9];
								//age
								pmb->ruser_meshblock_data[2](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+1];
								//x1-x3
								pmb->ruser_meshblock_data[3](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+2];
								pmb->ruser_meshblock_data[4](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+3];
								pmb->ruser_meshblock_data[5](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+4];
								//v1-v3
								pmb->ruser_meshblock_data[6](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+5];
								pmb->ruser_meshblock_data[7](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+6];
								pmb->ruser_meshblock_data[8](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+7];
								//metal mass
							}
							else{
								if (((rdata_all_rec[n*max_particle_num_m*9+m*9+2]>=xl-sar)&&(rdata_all_rec[n*max_particle_num_m*9+m*9+2]<xr+sar)&&
									(rdata_all_rec[n*max_particle_num_m*9+m*9+3]>=yl-sar)&&(rdata_all_rec[n*max_particle_num_m*9+m*9+3]<yr+sar)&&
									(rdata_all_rec[n*max_particle_num_m*9+m*9+4]>=zl-sar)&&(rdata_all_rec[n*max_particle_num_m*9+m*9+4]<zr+sar))&&
									(idata_all_rec[n*max_particle_num_m*3+m*3+1] == 2)){
									//num of particles ++
									pmb->iuser_meshblock_data[0](0)++;
									int nc=pmb->iuser_meshblock_data[0](0)-1;
									
									
									if (pmb->iuser_meshblock_data[0](0)>=max_p_num){
										std::cout<<"Error occour at rank "<<iuser_mesh_data[0](0)<<std::endl;
										std::cout<<"With particle signature "<<idata_all_rec[n*max_particle_num_m*3+m*3]<<std::endl;
										std::stringstream overload_msg;
										overload_msg<<"Fatal Error: particle overloaded"<<std::endl;
										ATHENA_ERROR(overload_msg);
									}
									
									
									//signature
									pmb->iuser_meshblock_data[1](nc)=idata_all_rec[n*max_particle_num_m*3+m*3];
									//std::cout<<"received signature "<<idata_all_rec[n*max_particle_num_m*2+m*2]<<std::endl;
									//particle type
									pmb->iuser_meshblock_data[2](nc)=-3;
									//SN number
									pmb->iuser_meshblock_data[3](nc)=idata_all_rec[n*max_particle_num_m*3+m*3+2];
									//mass
									pmb->ruser_meshblock_data[1](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9];
									//age
									pmb->ruser_meshblock_data[2](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+1];
									//x1-x3
									pmb->ruser_meshblock_data[3](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+2];
									pmb->ruser_meshblock_data[4](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+3];
									pmb->ruser_meshblock_data[5](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+4];
									//v1-v3
									pmb->ruser_meshblock_data[6](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+5];
									pmb->ruser_meshblock_data[7](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+6];
									pmb->ruser_meshblock_data[8](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+7];
									//metal mass
									pmb->ruser_meshblock_data[9](nc)=rdata_all_rec[n*max_particle_num_m*9+m*9+8];
								}
							}
						}
					}
				}
				
				
			}
			
			
			//renew cooling mask
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						pmb->ruser_meshblock_data[10](k,j,i)=1.0;
					}
				}
			}
			
			//setting cooling mask
			/*
			for (int pn=0;pn<pmb->iuser_meshblock_data[0](0);pn++){
				if ((pmb->iuser_meshblock_data[2](pn)==2)||(pmb->iuser_meshblock_data[2](pn)==-3)){
					Real x0=pmb->ruser_meshblock_data[3](pn);
					Real y0=pmb->ruser_meshblock_data[4](pn);
					Real z0=pmb->ruser_meshblock_data[5](pn);
					
					for (int k=pmb->ks; k<=pmb->ke; k++) {
						for (int j=pmb->js; j<=pmb->je; j++) {
							for (int i=pmb->is; i<=pmb->ie; i++) {
								Real x = pmb->pcoord->x1v(i);
								Real y = pmb->pcoord->x2v(j);
								Real z = pmb->pcoord->x3v(k);
								Real rad2 = SQR(x-x0) + SQR(y-y0) + SQR(z-z0);
								pmb->ruser_meshblock_data[10](k,j,i)=pmb->ruser_meshblock_data[10](k,j,i)*(1.0-exp(-(rad2)/(2*SQR(1.5*sigma1))));
							}
						}
					}
				}
			}
			*/
			
			//renew radiation field
			//initialze null radiation field
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						pmb->ruser_meshblock_data[11](k,j,i)=0.0;
					}
				}
			}
			//setting radiation field
			//calculating optical depth for photo-electric heating
			Real avg_den=0.0;
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						avg_den += pmb->phydro->u(IDN,k,j,i)/((pmb->ke-pmb->ks+1)*(pmb->je-pmb->js+1)*(pmb->ie-pmb->is+1));
					}
				}
			}
			//accounting fuv flux from nearby meshblocks
			Real r_init = 1.0*3.0e18;
			Real kappa = 100.0;
			Real peh_tau = 1.0/(kappa*avg_den);//two temperature planck vag optical depth for photo-electric heating
			Real Fuv_neighbor = 0.0;
			for (int n=0; n<node_num; n++){
				for (int m=0; m<mb_num_all_rec[n]; m++){
					Real px = rdata_ptotal_rec[n*max_mb_num_m*4+m*4+1];
					Real py = rdata_ptotal_rec[n*max_mb_num_m*4+m*4+2];
					Real pz = rdata_ptotal_rec[n*max_mb_num_m*4+m*4+3];
					if ((px<xl)||(px>=xr)||
						(py<yl)||(py>=yr)||
						(pz<zl)||(pz>=zr)){
						Real rad = std::sqrt(SQR(xc-px) + SQR(yc-py) + SQR(zc-pz));
						Real stellar_mass_neighbor = rdata_ptotal_rec[n*max_mb_num_m*4+m*4];
						if ((rad<5.0*peh_tau)&&(rad<100.0*3.0e18)){
							Fuv_neighbor += ((1.0e42*(stellar_mass_neighbor/1.0e6*2.0e33))/(4*PI*SQR(r_init)))*exp(-(rad-r_init)/peh_tau)*(1.0/SQR(rad/r_init));
						}
					}
				}
			}
			if (std::isnan(Fuv_neighbor)==true){
				std::stringstream nan_msg;
				nan_msg<<"Fatal Error: neighbor fuv flux nan "<<std::endl;
				ATHENA_ERROR(nan_msg);
			}
			//uv flux from background and neighbor
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						pmb->ruser_meshblock_data[11](k,j,i) += Fuv_neighbor;
					}
				}
			}
			//uv flux from star particles host by local meshblock
			for (int pn=0;pn<pmb->iuser_meshblock_data[0](0);pn++){
				if (pmb->iuser_meshblock_data[2](pn)==2){
					Real x0=pmb->ruser_meshblock_data[3](pn);
					Real y0=pmb->ruser_meshblock_data[4](pn);
					Real z0=pmb->ruser_meshblock_data[5](pn);
					for (int k=pmb->ks; k<=pmb->ke; k++) {
						for (int j=pmb->js; j<=pmb->je; j++) {
							for (int i=pmb->is; i<=pmb->ie; i++) {
								Real x = pmb->pcoord->x1v(i);
								Real y = pmb->pcoord->x2v(j);
								Real z = pmb->pcoord->x3v(k);
								Real rad = std::sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
								Real tau = 1.0/(kappa*avg_den); //computed from kappa=100 cm2 g-1 & n=100 cm-3
								if (rad>5.0*3.0e18){
									rad=5.0*3.0e18;
								}
								pmb->ruser_meshblock_data[11](k,j,i) += (1.0e42/20.0/(4*PI*SQR(r_init)))*exp(-(rad-r_init)/tau)/SQR(rad/r_init);
								if (std::isnan(pmb->ruser_meshblock_data[11](k,j,i))==true){
									std::stringstream nan_msg;
									nan_msg<<"Fatal Error: local star particle uv flux nan "<<std::endl;
									ATHENA_ERROR(nan_msg);
								}
							}
						}
					}
				}
			}
			
			
			//==========part 6 particle gravity solver==========
			
			int particle_num_mb=pmb->iuser_meshblock_data[0](0);
			
			//gravity(particle fraction) solver for gas
			for (int k=pmb->ks; k<=pmb->ke; k++) {
				for (int j=pmb->js; j<=pmb->je; j++) {
					for (int i=pmb->is; i<=pmb->ie; i++) {
						//original velocity square
						Real v_abs2_org=SQR(pmb->phydro->w(IVX,k,j,i))+SQR(pmb->phydro->w(IVY,k,j,i))+SQR(pmb->phydro->w(IVZ,k,j,i));
						
						pmb->phydro->u(IM1,k,j,i) += pmb->phydro->u(IDN,k,j,i)*g1*dt;
						pmb->phydro->u(IM2,k,j,i) += pmb->phydro->u(IDN,k,j,i)*g2*dt;
						pmb->phydro->u(IM3,k,j,i) += pmb->phydro->u(IDN,k,j,i)*g3*dt;
						
						pmb->phydro->w(IVX,k,j,i) += g1*dt;
						pmb->phydro->w(IVY,k,j,i) += g2*dt;
						pmb->phydro->w(IVZ,k,j,i) += g3*dt;
						
			
						if (particle_num_mb>0){
							Real x = pmb->pcoord->x1v(i);
							Real y = pmb->pcoord->x2v(j);
							Real z = pmb->pcoord->x3v(k);
							
							Real gp1 = 0.0;
							Real gp2 = 0.0;
							Real gp3 = 0.0;
							
							for (int n=0; n<particle_num_mb; n++){
								if ((pmb->iuser_meshblock_data[2](n)==1)||(pmb->iuser_meshblock_data[2](n)==2)){
									Real px = pmb->ruser_meshblock_data[3](n);
									Real py = pmb->ruser_meshblock_data[4](n);
									Real pz = pmb->ruser_meshblock_data[5](n);
									Real pm = pmb->ruser_meshblock_data[1](n);
									
									Real rad = std::sqrt(SQR(x-px) + SQR(y-py) + SQR(z-pz));
									
									gp1 += g*pm*((x-px)/rad)*rtof(rad,dx);//G * mass * unit vector * smoothed inverse squared r 
									gp2 += g*pm*((y-py)/rad)*rtof(rad,dx);//assume a isotropic lattice(dx = dy = dz)
									gp3 += g*pm*((z-pz)/rad)*rtof(rad,dx);
								}
							}
							
							pmb->phydro->u(IM1,k,j,i) += pmb->phydro->u(IDN,k,j,i)*gp1*dt;
							pmb->phydro->u(IM2,k,j,i) += pmb->phydro->u(IDN,k,j,i)*gp2*dt;
							pmb->phydro->u(IM3,k,j,i) += pmb->phydro->u(IDN,k,j,i)*gp3*dt;
							
							pmb->phydro->w(IVX,k,j,i) += gp1*dt;
							pmb->phydro->w(IVY,k,j,i) += gp2*dt;
							pmb->phydro->w(IVZ,k,j,i) += gp3*dt;
						}
			
						
						//final velocity square
						Real v_abs2_fin=SQR(pmb->phydro->w(IVX,k,j,i))+SQR(pmb->phydro->w(IVY,k,j,i))+SQR(pmb->phydro->w(IVZ,k,j,i));
						//correction made for kinetic energy (IEN in conservatives includes k-energy 
						//therefore energy gain in grav potential needs to be taken into consideration)
						//with out this correction some cell with high acceleration could end up with
						//unphysical cooling and even negative pressure
						pmb->phydro->u(IEN,k,j,i) += 0.5*(v_abs2_fin - v_abs2_org)*pmb->phydro->u(IDN,k,j,i);
					}
				}
			}
			
			//gravity solver for particles
			
			if (particle_num_mb>0){
				for (int n=0; n<particle_num_mb; n++){
					if ((pmb->iuser_meshblock_data[2](n)==1)||(pmb->iuser_meshblock_data[2](n)==2)){
						Real x = pmb->ruser_meshblock_data[3](n);
						Real y = pmb->ruser_meshblock_data[4](n);
						Real z = pmb->ruser_meshblock_data[5](n);
						Real r = std::sqrt(SQR(x) + SQR(y));
						
						//locating the neariest cell
						int i = pmb->is+(int)((pmb->ruser_meshblock_data[3](n)-xl)/dx);
						int j = pmb->js+(int)((pmb->ruser_meshblock_data[4](n)-yl)/dy);
						int k = pmb->ks+(int)((pmb->ruser_meshblock_data[5](n)-zl)/dz);
						//potential gradient
						
						Real dphi1 = 0.5*(pmb->ruser_meshblock_data[0](k,j,i+1)-pmb->ruser_meshblock_data[0](k,j,i-1));
						Real dphi2 = 0.5*(pmb->ruser_meshblock_data[0](k,j+1,i)-pmb->ruser_meshblock_data[0](k,j-1,i));
						Real dphi3 = 0.5*(pmb->ruser_meshblock_data[0](k+1,j,i)-pmb->ruser_meshblock_data[0](k-1,j,i));
						
						//Real dphidr=(phi_m82_without_gas(r*1.01,0.0,z)-phi_m82_without_gas(r*0.99,0.0,z))/(0.02*r);
						//Real dphidz=(phi_m82_without_gas(r,0.0,z*1.01)-phi_m82_without_gas(r,0.0,z*0.99))/(0.02*z);
						
						Real dphi1_g = 0.5*(pmb->pgrav->phi(k,j,i+1)-pmb->pgrav->phi(k,j,i-1));
						Real dphi2_g = 0.5*(pmb->pgrav->phi(k,j+1,i)-pmb->pgrav->phi(k,j-1,i));
						Real dphi3_g = 0.5*(pmb->pgrav->phi(k+1,j,i)-pmb->pgrav->phi(k-1,j,i));
						
						
						Real gp1 = 0.0;
						Real gp2 = 0.0;
						Real gp3 = 0.0;
						
						
						for (int m=0; m<particle_num_mb; m++){
							if ((m!=n)&&((pmb->iuser_meshblock_data[2](m)==1)||(pmb->iuser_meshblock_data[2](m)==2))){
								Real px = pmb->ruser_meshblock_data[3](m);
								Real py = pmb->ruser_meshblock_data[4](m);
								Real pz = pmb->ruser_meshblock_data[5](m);
								Real pm = pmb->ruser_meshblock_data[1](m);
								
								Real rad = std::sqrt(SQR(x-px) + SQR(y-py) + SQR(z-pz));
								
								
								gp1 += g*pm*((x-px)/rad)*rtof(rad,dx);//G * mass * unit vector * smoothed inverse squared r 
								gp2 += g*pm*((y-py)/rad)*rtof(rad,dx);//assume a isotropic lattice(dx = dy = dz)
								gp3 += g*pm*((z-pz)/rad)*rtof(rad,dx);
								
							}
						}
						
						
						//velocity kick
						//gravity excerted by 
						//a.particles within the same meshblock
						//b.particles beyond
						//c.static grav potential background
						//d.gas
						
						pmb->ruser_meshblock_data[6](n) += dt*(gp1+g1-(dphi1+dphi1_g)/dx);
						pmb->ruser_meshblock_data[7](n) += dt*(gp2+g2-(dphi2+dphi2_g)/dy);
						pmb->ruser_meshblock_data[8](n) += dt*(gp3+g3-(dphi3+dphi3_g)/dz);
						//pmb->ruser_meshblock_data[6](n) += dt*(gp1+g1-(dphi1_g)/dx)-dphidr*(x/r);
						//pmb->ruser_meshblock_data[7](n) += dt*(gp2+g2-(dphi2_g)/dy)-dphidr*(y/r);
						//pmb->ruser_meshblock_data[8](n) += dt*(gp3+g3-(dphi3_g)/dz)-dphidz;
					}
				}
			}
			
		}
		
		
		
		//output scoope
		if (fmod(time,particle_output_dt)<dt){
			//
			int particle_out_num_m_send =0;
			for (int bn=0; bn<nblocal; ++bn) {
				MeshBlock *pmb = my_blocks(bn);
				LogicalLocation &loc = pmb->loc;
				//int particle_num_mb=pmb->iuser_meshblock_data[0](0);
				//particle_num_m_send+=particle_num_mb;
				int particle_num_mb=pmb->iuser_meshblock_data[0](0);
				if (particle_num_mb>0){
					for (int n=0; n<particle_num_mb; n++){
						if ((pmb->iuser_meshblock_data[2](n)==0)||(pmb->iuser_meshblock_data[2](n)==1)||(pmb->iuser_meshblock_data[2](n)==2)){
							particle_out_num_m_send++;
						}
					}
				}
			}
			
			int max_particle_out_num_m=0;
			//max_particle_num_m is used to determine the size of array to receive data
			int particle_out_num_all_rec[node_num];
			MPI_Allgather(&particle_out_num_m_send,1,MPI_INT,&particle_out_num_all_rec,1,MPI_INT,MPI_COMM_WORLD);
			for (int n=0; n<node_num; n++){
				if (particle_out_num_all_rec[n]>max_particle_out_num_m){
					max_particle_out_num_m=particle_out_num_all_rec[n];
				}
			}
			
			int idata_out_m_send[2*max_particle_out_num_m];
			Real rdata_out_m_send[9*max_particle_out_num_m];
			if (particle_out_num_m_send>0){
				pn_counter=0;
				for (int bn=0; bn<nblocal; ++bn) {
					MeshBlock *pmb = my_blocks(bn);
					int particle_num_mb=pmb->iuser_meshblock_data[0](0);
					if (particle_num_mb>0){
						for (int n=0; n<particle_num_mb; n++){
							if ((pmb->iuser_meshblock_data[2](n)==0)||(pmb->iuser_meshblock_data[2](n)==1)||(pmb->iuser_meshblock_data[2](n)==2)){
								//signature
								idata_out_m_send[2*pn_counter]   = pmb->iuser_meshblock_data[1](n);
								//particle type
								idata_out_m_send[2*pn_counter+1] = pmb->iuser_meshblock_data[2](n);
								//age
								rdata_out_m_send[9*pn_counter+1] = pmb->ruser_meshblock_data[2](n)+dt;
								//x1-x3
								rdata_out_m_send[9*pn_counter+2] = pmb->ruser_meshblock_data[3](n)+dt*pmb->ruser_meshblock_data[6](n);
								rdata_out_m_send[9*pn_counter+3] = pmb->ruser_meshblock_data[4](n)+dt*pmb->ruser_meshblock_data[7](n);
								rdata_out_m_send[9*pn_counter+4] = pmb->ruser_meshblock_data[5](n)+dt*pmb->ruser_meshblock_data[8](n);
								//mass
								rdata_out_m_send[9*pn_counter]   = pmb->ruser_meshblock_data[1](n);
								//metal mass
								rdata_out_m_send[9*pn_counter+8] = pmb->ruser_meshblock_data[9](n);
								//velocity
								rdata_out_m_send[9*pn_counter+5] = pmb->ruser_meshblock_data[6](n);
								rdata_out_m_send[9*pn_counter+6] = pmb->ruser_meshblock_data[7](n);
								rdata_out_m_send[9*pn_counter+7] = pmb->ruser_meshblock_data[8](n);
								
								pn_counter++;
							}
						}
					}
				}
				if (pn_counter>0){
				}
			}
			
			int idata_out_all_rec[node_num*max_particle_out_num_m*2];
			Real rdata_out_all_rec[node_num*max_particle_out_num_m*9];
			MPI_Gather(&idata_out_m_send ,max_particle_out_num_m*2 ,MPI_INT ,&idata_out_all_rec ,max_particle_out_num_m*2 ,MPI_INT ,0 ,MPI_COMM_WORLD);
			MPI_Gather(&rdata_out_m_send ,max_particle_out_num_m*9 ,MPI_DOUBLE ,&rdata_out_all_rec ,max_particle_out_num_m*9 ,MPI_DOUBLE ,0 ,MPI_COMM_WORLD);
			
			if (iuser_mesh_data[0](0)==0){
				int pcount=0;
				for (int n=0; n<node_num; n++){
					if (particle_out_num_all_rec[n]>0){
						pcount+=particle_out_num_all_rec[n];
					}
				}
				
				std::ofstream out("particle_out."+std::to_string((int)floor(time/particle_output_dt))+".txt",std::ios_base::app);
				std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
				std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
				std::cout<<"[time] [particle num]"<<std::endl;
				std::cout<<time<<" "<<pcount<<std::endl;
				std::cout<<"[signature] [type] [x1] [x2] [x3]"<<std::endl;
				
				for (int n=0; n<node_num; n++){
					if (particle_out_num_all_rec[n]>0){
						//std::cout<<"meshblock split"<<std::endl;
						for (int m=0; m<particle_out_num_all_rec[n]; m++){
							if (idata_out_all_rec[n*max_particle_num_m*2+m*2+1]>=0){
								std::cout<<idata_out_all_rec[n*max_particle_out_num_m*2+m*2]<<" ";
								std::cout<<idata_out_all_rec[n*max_particle_out_num_m*2+m*2+1]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+1]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+2]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+3]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+4]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+5]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+6]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+7]<<" ";
								std::cout<<rdata_out_all_rec[n*max_particle_out_num_m*9+m*9+8]<<std::endl;
							}
						}
					}
				}
				std::cout.rdbuf(coutbuf);
			}
		}
	}
	return;
}

void srcmask(AthenaArray<Real> &src, int is, int ie, int js, int je, int ks, int ke, const MGCoordinates &coord) {
	
	Real phi_edge = phi_3m(900*3.0e18,0.0,0.0,e_gas,r_hm_gas,m_gas);
	for (int k=ks; k<=ke; ++k) {
		for (int j=js; j<=je; ++j) {
			for (int i=is; i<=ie; ++i) {
				Real x = coord.x1v(i);
				Real y = coord.x2v(j);
				Real z = coord.x3v(k);
				Real rr=std::sqrt(SQR(x)+SQR(y));
				//if (r > maskr)
				//	src(k, j, i) = 0.0;
				
				//Real phi_3m_local = phi_3m(x,y,z,e_gas,r_hm_gas,m_gas);
				//if (phi_3m_local<phi_edge){
				if (rr>=950.0*3.0e18){
					src(k, j, i) = 0.0;
				}
			}
		}
	}
	return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  //do nothing
  return;
}