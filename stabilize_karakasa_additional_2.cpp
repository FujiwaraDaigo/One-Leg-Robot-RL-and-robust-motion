#include <ode/ode.h>
#include <drawstuff/drawstuff.h>
#include <dlib/hash.h>
#include <dlib/matrix.h>
#include <dlib/rand.h>
#include <dlib/optimization.h>
#include <dlib/dnn/cpu_dlib.h>
//#include <dlib/threads/threads_kernel.h>
#include <dlib/matrix/matrix_abstract.h>
#include <dlib/dnn/cuda_dlib.h>
#include <dlib/dnn/tensor.h>
#include <omp.h>
#include <iostream>
#include <time.h>
using namespace dlib;
using namespace cpu;

//using namespace boost;

using namespace std;

#ifdef _MSC_VER
#pragma warning(disable:4244 4305)  // for VC++, no precision loss complaints
#endif

#ifdef dDOUBLE
#define dsDrawBox      dsDrawBoxD
#define dsDrawSphere   dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule  dsDrawCapsuleD
#define leg_max 1
#endif
#define N_number 2
#define DOF_number 3
#define Dnum 6
#define num_base 1
#define num_policy 20
#define all_policy_num num_policy*DOF_number
#define division 1200
#define K_limit 250





#define terminal_step 1200
#define iteration_number 100
#define step_size 0.01
///CMA-ES
#define myutation_rate 0.05
#define myutation_mulitiple 4
#define myutation_num 1000
#define myutation_max 40
#define max_trial_num 20
#define initial_dist 1
#define masatu_num 1
#define initial_myu_step 10

///ロバスト化関連////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define state_number 9
#define palallel_p 4  ///プロセッサ数に合わせる
#define param_samp_num 20
///end/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

dWorldID world[palallel_p];  // 動力学計算用ワールド
dSpaceID space[palallel_p];  // 衝突検出用スペース
dGeomID  ground[palallel_p]; // 地面
dJointGroupID contactgroup[palallel_p]; // コンタクトグループ
dReal r = 0.2, m  = 1.0;
dsFunctions fn;
dlib::rand  rnd;
dlib::rand  rnd2;
typedef struct {       // MyObject構造体
  dBodyID body;        // ボディ(剛体)のID番号（動力学計算用）
  dGeomID geom;        // ジオメトリのID番号(衝突検出計算用）
  double  l,r,m;       // 長さ[m], 半径[m]，質量[kg]
} MyObject;

double ALD_thred=0.2;
static MyObject base[palallel_p][2],Motor[palallel_p][leg_max][3],Pla[palallel_p][leg_max][3],Al[palallel_p][leg_max][2],base_under[palallel_p];    // leg[0]:上脚, leg[1]:下脚
static dReal base_r,base_lz,Motor_lx,Motor_ly,Motor_lz,Pla_lx,Pla_ly,Pla_lz,Al_lx[2],Al_ly[2],Al_lz[2],base_m,Motor_m,Pla_m,Al_m[1],base_x[2],base_y[2],base_z[2];
static dJointID Joint_Motor[palallel_p][leg_max][3],Joint_Pla[palallel_p][leg_max][3],Joint_Al[palallel_p][leg_max],Joint_fix[palallel_p][leg_max],joint_base_under[palallel_p]; // ヒンジ, スライダー
double old_posi;
static int STEPS = -1;            // シミュレーションのステップ数
static dReal S_LENGTH = 0.0;     // スライダー長
static dReal H_ANGLE  = 0.0;
FILE *outputfile20,*outputfile21,*inputfile1,*outputfile22,*outputfile23,*outputfile24,*outputfile25,*outputfile26;
static dReal Motor_x[3];
static dReal Motor_y[3];
static dReal Motor_z[3];
static dReal Pla_x[3];
static dReal Pla_y[3];
static dReal Pla_z[3];
double Pla_ly_2;
double Pla_lx_2;
double Pla_lz_2;
static dReal Al_x[1];
static dReal Al_y[1];
static dReal Al_z[1];
static dReal Joint_M_x[3];
static dReal Joint_M_y[3];
double Al_l,Al_r;
static dReal Joint_M_z[3];
dReal PI=atan(1)*4;
dReal theta2;
int b=0;
int leg_number=0;
dJointFeedback *feedback = new dJointFeedback;

int best_theta_count=0;
//PI2
//PI2パラメータ

dReal ddt=step_size;


//lambda/Sの制約
dReal W_R=0.001;
dReal W_sigma=1;
matrix <double> K_nolm_m(1,K_limit);
double K_nolm;
matrix <double> E(all_policy_num,all_policy_num);
matrix <double> S(all_policy_num,all_policy_num);
matrix <double> sigma(all_policy_num,all_policy_num);
dReal lamda;
matrix <double,num_policy,terminal_step> epst[K_limit];
matrix <double,all_policy_num,1> meanzerot;
matrix<double> WN2(1,terminal_step);


//dmatrix siguma=W_sigma*E;
dReal lambda=W_R*W_sigma;
int update=0;
int eroor=0;
time_t timer1,timer2;
double time_q;
matrix <double,1,terminal_step> statecostkt[K_limit];
matrix <double,1,terminal_step> controlcostkt[K_limit];
matrix <double,1,1> terminalcostkt[K_limit];
matrix <double,DOF_number*2+num_base*3,terminal_step> trajectory[K_limit];
matrix <double> theta(all_policy_num,1);

matrix<double,1,terminal_step> immediate_cost[K_limit];

matrix <double> dthetait(all_policy_num,terminal_step);

int off_noise=0;
double sum_controlcost[K_limit],sum_statecost[K_limit];
double best_taskcount,best_update;
matrix <double> best_theta(all_policy_num,1);

matrix <double,all_policy_num,terminal_step> eps[K_limit];

double cost[K_limit];
double taskcount;
matrix <double> meanzero[K_limit];
matrix <double> dthetai(all_policy_num,terminal_step);
matrix <double> parimmediatecostk[K_limit];
matrix <double> parterminalcostk[K_limit];
matrix <double> thetat;
matrix <double,all_policy_num,all_policy_num> exe_theta[K_limit];
matrix <double> dtheta(all_policy_num,1);
int h;

//要変更
matrix <double,all_policy_num,all_policy_num> M[terminal_step];

//
double check;
int errort;
double cost1,best_cost1,best_cost2;
matrix <double> best_trajectory1;
matrix <double> WN1(terminal_step,terminal_step);
matrix <double> dthetat(all_policy_num,1);
matrix <double,all_policy_num,terminal_step> t_epst[palallel_p];
matrix <double> One_temp;
matrix <double,all_policy_num,all_policy_num> temp_exetheta[K_limit];
matrix <double> t(1,terminal_step);
matrix <double> st(1,terminal_step);
matrix <double> w(num_policy,terminal_step);
matrix <double> G(num_policy,terminal_step);
matrix <double> ws1(all_policy_num,terminal_step);
matrix <double> Gs(all_policy_num,terminal_step);
matrix <double> g0(DOF_number,1);
matrix <double> gg(DOF_number,1);
matrix <double> g(1,terminal_step);
matrix <double> gs(DOF_number,terminal_step);
matrix <double,DOF_number,terminal_step> psi[1];
matrix <double> rr0(DOF_number*2,1);
matrix <double> rrg(DOF_number*2,1);
matrix <double> tt(1,terminal_step);
matrix <double> dt_test;
matrix <double> kernel_output;
matrix <double> k_st;
matrix <double> temp_K(division,K_limit);
matrix <double> Gausian_kernel;
matrix <double,all_policy_num,terminal_step> theta_k;

///ロバスト化関連////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double grav;

matrix <double,param_samp_num,terminal_step> statecost[max_trial_num];
double std_ratio=0.1;

//距離
double distanceY;
matrix <double,1,max_trial_num> pre_distanceY;
matrix <double,1,max_trial_num> post_distanceY;

//状態ベクトルxt
matrix <double,state_number,terminal_step> state_vector[max_trial_num];
matrix <double,state_number,terminal_step> temp_state;

//確率的重みづけ係数
matrix <double,1,max_trial_num> Pi;

//コスト重みづけ係数
double alphaP=20;

//参照状態変数
matrix <double,state_number,terminal_step> ref_state;

//コスト変化に対する重み比
double ration=0.2;

double weight[max_trial_num];

double noise_ratio=0.1;

FILE *outputfile27;

///end///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int memo_num=1;

double ddeltat=0;
int epic=0;
double alphag=0;
double alphaq=25;
int restart_judge=0;
double r0;
double rg;
double tau;
matrix <double> phi;
double tf;
double phi0=0;
int K_max=1;
matrix <double> kt(1,1);
int judge_number=0;
double hyper_p=0.0000001;
int d=1;
int episode=1;
int STEP;
///CMA-ES ============================
matrix <double> p_c(all_policy_num,1);
double cp;
double p_succ,p_succ_ave;

double damp,distribution,distribution_p;
double c_cov,c_c,p_thresh;
double p_succ_target;
double temp_cost,a_best;
int judge_CMA_ES;
int lamda_count;
matrix <double> theta_best(all_policy_num,1);
int CMA_initial=0;
int CMA_PI_JUDGE=0;
matrix <double> old_theta(all_policy_num,1);
double old_distribution=1.1;
int myutation_judge=0;
double temp_dist=0;
int myutation_occur=0;
int epic_myutation_judge=0;
int K_max_rest;
matrix <double> theta_CMA_best(all_policy_num,1);
/// ==================================
dReal umax;
///palalell 関係=====================
int thread_number;
int p_p_p;
int ALD_judge=0;
matrix <double> real_ang(terminal_step,DOF_number);
matrix <double> real_angrate(terminal_step,DOF_number);
int real_judge=0;
///===================================

///角度制御担当分=====================
double angle[1000][leg_max][3];
matrix <double,3*leg_max,terminal_step> ang_vel[500];
double old_cost=0;
///===================================

double best_cost=0;
int K_count=0;
int K_ave=0;
int eroor_count=0;
double distance_num[1000];
double distance_max;
int sub_K_count=0;
int sub_K_ave=0;
int dist_la=0;
double p_succ_dist,p_succ_ave_dist;
double cost_judge;
double cost_judge_ave;
int K_max_o=0;
int old_K_max_o=0;
int old_old_k_max_o=0;
int max_k_max=0;
double max_dist=0;
int max_update=0;
///摩擦係数//////////////////////////////////////////////////////////////////////////////////////////////
double Al_friction;
double Pla_friction;
///end///////////////////////////////////////////////////////////////////////////////////////////////////
matrix<double> copy_m(terminal_step,4);
matrix<double>excel(terminal_step,4*max_trial_num);
double distance_maximam;
int m_count=0;
int sample_total=0;
double myu_step=sqrt(initial_myu_step);
double myu_step_d=0.5;
int jal=0;
double base_under_m=10*0.001;


// 衝突検出計算
static void nearCallback(void *data, dGeomID o1, dGeomID o2){
  static const int N = 30;     // 接触点数
dContact contact[N];
int t_num;
  //int isGround = ((ground[p_p_p] == o1) || (ground[p_p_p] == o2));

  // 2つのボディがジョイントで結合されていたら衝突検出しない
  dBodyID b1 = dGeomGetBody(o1);
  dBodyID b2 = dGeomGetBody(o2);

  if (b1 && b2 && dAreConnectedExcluding(b1,b2,dJointTypeContact)) return;
dWorldID worldd;
  worldd=dBodyGetWorld(b1);
  for(int i=0;i<palallel_p;i++){
    if(worldd==world[i]){
        t_num=i;break;
    }
  }
  int n =  dCollide(o1,o2,N,&contact[0].geom,sizeof(dContact));
  if (n>0)  {
    for (int i = 0; i < n; i++) {
      contact[i].surface.mode   =  dContactSoftERP |
                                  dContactSoftCFM;
      contact[i].surface.soft_erp   = 0.2;   // 接触点のERP
      contact[i].surface.soft_cfm   = 0.001; // 接触点のCFM
        if(b1==Al[t_num][0][0].body||b2==Al[t_num][0][0].body){
         contact[i].surface.mu     = Al_friction; // 摩擦係数:無限大

      }else{
         contact[i].surface.mu     = Pla_friction; // 摩擦係数:無限大
      }
      ///contact[i].surface.mu     = 1; // 摩擦係数:無限大
      dJointID c = dJointCreateContact(  worldd,
                                       contactgroup[t_num],&contact[i]);
                                     //  printf("S\n");
      dJointAttach (c,dGeomGetBody(contact[i].geom.g1),
                      dGeomGetBody(contact[i].geom.g2));
    }
  }
}


void createleg(int parallel_number){
 dMass mass;
 dMassSetZero(&mass);
  dMatrix3 R,RR,YY,TA;
  dReal r1[3],r2[3],r3[2];

    dRFromAxisAndAngle(R,1,0,0,M_PI/2);
 dRFromAxisAndAngle(TA,0,1,0,M_PI/2);


 for (int i=0;i<3;i++){
 //モーターの作成
 b=leg_number%3;
// switch (b){
// case 0: dRFromAxisAndAngle(RR,0,0,1,M_PI/6);break;
// case 1: dRFromAxisAndAngle(RR,0,0,1,-M_PI/6); break;
// case 2:dRFromAxisAndAngle(RR,0,0,1,M_PI/2);break;
//
// }



 r1[i]=sqrt((base_x[leg_number/3]-Motor_x[i])*(base_x[leg_number/3]-Motor_x[i])+(base_y[leg_number/3]-Motor_y[i])*(base_y[leg_number/3]-Motor_y[i]));
 r2[i]=sqrt((base_x[leg_number/3]-Pla_x[i])*(base_x[leg_number/3]-Pla_x[i])+(base_y[leg_number/3]-Pla_y[i])*(base_y[leg_number/3]-Pla_y[i]));

 theta2 =((-5*M_PI/6))+((b)*2*M_PI/3);

// printf("theta2=%f\n",theta2);
 Motor[parallel_number][leg_number][i].body=dBodyCreate(world[parallel_number]);
 if(i==0)
   {dMassSetBoxTotal(&mass,Motor_m,Motor_lz,Motor_lx,Motor_ly);}
else{ dMassSetBoxTotal(&mass,Motor_m,Motor_lx,Motor_ly,Motor_lz);}
 dBodySetMass(Motor[parallel_number][leg_number][i].body,&mass);
//dBodySetRotation(Motor[p_n][leg_number][i].body,RR);

// if (i!=0){dBodySetRotation(Motor[leg_number][i].body,R);}

 dBodySetPosition(Motor[parallel_number][leg_number][i].body,Motor_x[i],Motor_y[i],Motor_z[i]);



if(i==0)
    {
        dBodySetRotation(Motor[parallel_number][leg_number][i].body,R);
        ///Motor[p_n][leg_number][i].geom=dCreateBox(space[p_n],Motor_lx,Motor_lz,Motor_ly);
    Motor[parallel_number][leg_number][i].geom=dCreateBox(space[parallel_number],Motor_lz,Motor_lx,Motor_ly);
    /// dGeomSetBody(Motor[p_n][leg_number][i].geom,Motor[p_n][leg_number][i].body);
    Joint_fix[parallel_number][leg_number]=dJointCreateFixed(world[parallel_number],0);
    dJointAttach(Joint_fix[parallel_number][leg_number],Motor[parallel_number][leg_number][0].body,base[parallel_number][0].body);
    dJointSetFixed(Joint_fix[parallel_number][leg_number]);
    }
else {dBodySetRotation(Motor[parallel_number][leg_number][i].body,R);
        Motor[parallel_number][leg_number][i].geom=dCreateBox(space[parallel_number],Motor_lx,Motor_ly,Motor_lz);

}

dGeomSetBody(Motor[parallel_number][leg_number][i].geom,Motor[parallel_number][leg_number][i].body);

 }
 for(int i=0;i<3;i++){
         //接続用のプラスチック素材の作成
 Pla[parallel_number][leg_number][i].body=dBodyCreate(world[parallel_number]);
 if(i==1){

//        dMassSetBoxTotal(&mass,Pla_m,Pla_lz,Pla_ly,Pla_lx);
//  Pla[p_n][leg_number][i].geom=dCreateBox(space[p_n],Pla_lz,Pla_ly,Pla_lx);
   dMassSetBoxTotal(&mass,Pla_m,Pla_lx_2,Pla_ly_2,Pla_lz_2);

 Pla[parallel_number][leg_number][i].geom=dCreateBox(space[parallel_number],Pla_lx_2,Pla_ly_2,Pla_lz_2);

 }
 else{
        dMassSetBoxTotal(&mass,Pla_m,Pla_lx,Pla_ly,Pla_lz);
 Pla[parallel_number][leg_number][i].geom=dCreateBox(space[parallel_number],Pla_lx,Pla_ly,Pla_lz);
 }
 dBodySetMass(Pla[parallel_number][leg_number][i].body,&mass);
 dBodySetRotation(Pla[parallel_number][leg_number][i].body,R);

 dBodySetPosition(Pla[parallel_number][leg_number][i].body,base_x[0]+Pla_x[i],base_y[0]+Pla_y[i],Pla_z[i]);


 dGeomSetBody(Pla[parallel_number][leg_number][i].geom,Pla[parallel_number][leg_number][i].body);


 }
     //アルミの足先の作成
    // r3[i]=sqrt((base_x[leg_number/3]-Al_x[i])*(base_x[leg_number/3]-Al_x[i])+(base_y[leg_number/3]-Al_y[i])*(base_y[leg_number/3]-Al_y[i]));
 Al[parallel_number][leg_number][0].body=dBodyCreate(world[parallel_number]);
 dMassSetCapsuleTotal(&mass,Al_m[0],2,Al_r,Al_l);
 dBodySetRotation(Al[parallel_number][leg_number][0].body,TA);
 dBodySetMass(Al[parallel_number][leg_number][0].body,&mass);
 //dBodySetRotation(Al[p_n][leg_number][0].body,RR);
 dBodySetPosition(Al[parallel_number][leg_number][0].body,Al_x[0]+base_x[0],Al_y[0]+base_y[0],Al_z[0]);
 Al[parallel_number][leg_number][0].geom=dCreateCapsule(space[parallel_number],Al_r,Al_l);
 dGeomSetBody(Al[parallel_number][leg_number][0].geom,Al[parallel_number][leg_number][0].body);
 //各パーツの接続
 for (int i=0;i<3;i++){
    //モーター回転用
    Joint_Motor[parallel_number][leg_number][i]=dJointCreateHinge(world[parallel_number],0);
    if(i==0){
        dJointAttach(Joint_Motor[parallel_number][leg_number][i],base[parallel_number][leg_number].body,Pla[parallel_number][leg_number][0].body);
    }
    else if(i==2){
        dJointAttach(Joint_Motor[parallel_number][leg_number][i],Motor[parallel_number][leg_number][i].body,Pla[parallel_number][leg_number][i].body);
    }else{
    dJointAttach(Joint_Motor[parallel_number][leg_number][i],Motor[parallel_number][leg_number][i].body,Pla[parallel_number][leg_number][i-1].body);
    }

    dJointSetHingeAnchor(Joint_Motor[parallel_number][leg_number][i],Joint_M_x[i]+base_x[0],base_y[0]+Joint_M_y[i],Joint_M_z[i]);
  if(i==0) { dJointSetHingeAxis(Joint_Motor[parallel_number][leg_number][i],1,0,0);}else{
//        switch (b){
//      case 0:  dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],1,-sqrt(3),0); break;
//      case 1:  dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],1,sqrt(3),0); break;
//      case 2:  dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],1,0,0); break;
//      }
 dJointSetHingeAxis(Joint_Motor[parallel_number][leg_number][i],0,0,1);
        }
        if(i==0){
         dJointSetHingeParam(Joint_Motor[parallel_number][leg_number][i],dParamLoStop,-M_PI+0.005);
    dJointSetHingeParam(Joint_Motor[parallel_number][leg_number][i],dParamHiStop,M_PI-0.005);
        }else{
         dJointSetHingeParam(Joint_Motor[parallel_number][leg_number][i],dParamLoStop,-M_PI+0.005);
    dJointSetHingeParam(Joint_Motor[parallel_number][leg_number][i],dParamHiStop,M_PI-0.005);

        }



 }

 for(int i=0;i<3;i++){
    //プラ固定用
    Joint_Pla[parallel_number][leg_number][i]=dJointCreateFixed(world[parallel_number],0);
   if(i!=2) {dJointAttach(Joint_Pla[parallel_number][leg_number][i],Pla[parallel_number][leg_number][1].body,Motor[parallel_number][leg_number][i+1].body);}
   else {
        dJointAttach(Joint_Pla[parallel_number][leg_number][i],Pla[parallel_number][leg_number][i].body,Al[parallel_number][leg_number][0].body);
   }
    dJointSetFixed(Joint_Pla[parallel_number][leg_number][i]);
 }
// //アルミ固定用
// Joint_Al[p_n][leg_number]=dJointCreateFixed(world[p_n],0);
// dJointAttach(Joint_Al[p_n][leg_number],Al[p_n][leg_number][0].body,Al[p_n][leg_number][1].body);
// dJointSetFixed(Joint_Al[p_n][leg_number]);
leg_number++;

}


void createMonoBot(int parallel_number) {
  dMass mass;
  dMatrix3 TA;
  dRFromAxisAndAngle(TA,0,1,0,M_PI/2);

for (int i=0;i<num_base;i++){
base[parallel_number][i].body=dBodyCreate(world[parallel_number]);
dMassSetCylinderTotal(&mass,base_m,2,base_r,base_lz);
dBodySetMass(base[parallel_number][i].body,&mass);
dBodySetPosition(base[parallel_number][i].body,base_x[i]-base_lz/2-Pla_lx,base_y[i],base_z[i]);
dBodySetRotation(base[parallel_number][i].body,TA);
base[parallel_number][i].geom=dCreateCylinder(space[parallel_number],base_r,base_lz);
dGeomSetBody(base[parallel_number][i].geom,base[parallel_number][i].body);


}
//base_under[p_n].body=dBodyCreate(world[p_n]);
//dMassSetCylinderTotal(&mass,base_under_m,2,base_r,1*0.001);
//dBodySetMass(base_under[p_n].body,&mass);
//dBodySetPosition(base_under[p_n].body,base_x[0]-Pla_lx-1*0.001,base_y[0],base_z[0]);
//dBodySetRotation(base_under[p_n].body,TA);
//base_under[p_n].geom=dCreateCylinder(space[p_n],base_r,1*0.001);
//dGeomSetBody(base_under[p_n].geom,base_under[p_n].body);
//
//joint_base_under[p_n]=dJointCreateFixed(world[p_n],0);
//dJointAttach(joint_base_under[p_n],base[p_n][0].body,base_under[p_n].body);
//dJointSetFixed(joint_base_under[p_n]);
}


void make(int j){
    leg_number=0;

    createMonoBot(j);

    for (int i=0;i<leg_max;i++){

    createleg(j);
//  Joint_fix[j][i]=dJointCreateFixed(world[j],0);
//  dJointAttach( Joint_fix[j][i],base[j][i/3].body,Motor[j][i][0].body);
//  dJointSetFixed(Joint_fix[j][i]);
//  dMatrix3 Rl[3];
    }
}

void make_rhythmic(double r0,double rg){


    double dc ;

    for(int i=0;i<terminal_step;i++){
        tt(0,i)=i*step_size;
    }

    dc=2 * pi / num_policy;
    matrix <double> c(num_policy,1);

    for(int i=0;i<num_policy;i++){
        set_rowm(c,i)=i*dc+dc-M_PI;
    }

    double h = 1 / (2 * (dc*dc) );
    //cout<<phi0*ones_matrix<double>(1,terminal_step)+tt/tau<<endl;
    phi=tt/tau;
    double alphag = alphaq/2;

    ///st調整 tt➡terminal stが目標値に

     st = (r0 - rg) * exp(-alphag*(tt)/tau) + r0 * ones_matrix<double>(1,terminal_step);
    //printf("s4");
    if(st(0,0)==0){
        st(0,0)=0.001;
    }

    matrix <double> temp[num_policy];



    ///w修正必要　
     for(int i=0;i<num_policy;i++){
            ///cosの中身調整　
        set_rowm(w,i)=exp(h*(cos(phi-c(i,0)*ones_matrix<double>(1,terminal_step)))-ones_matrix<double>(1,terminal_step));
     }

     //GPGPU

     for(int i=0;i<terminal_step;i++){
            if(sum(colm(w,i))==0){
            printf("w_eroor");
            }
        set_colm(G,i)=colm(w,i)*st(0,i)/sum(colm(w,i));
     }

}


void make_dmp(matrix <double> dmp,matrix <double> gsi,matrix <double> fi){

    double betaq =alphaq/4;
    matrix <double> d_dmp(DOF_number*2,1);
    double q;
    double qd;
    for(int D=0;D<DOF_number;D++){
        q=dmp(2*D);
        qd=dmp(2*D+1);
       set_rowm(d_dmp,2*D)=qd;
       set_rowm(d_dmp,2*D+1)=(156.25*(gsi(D,0)-q)-25*qd+fi(D,0))/tau;
    }
    dt_test=d_dmp;
}


void excute_policy(matrix <double> thetag,matrix<double> noise){
       // printf("thetat\n");
    //printf("e1");
    matrix <double> thetat1;
    thetat1=thetag*ones_matrix<double>(1,terminal_step)+noise;
    //cout<<thetat<<endl;
    double parnum_policy =num_policy;
    double parnum_DOFs=DOF_number;
    double parnum_base=num_base;
    double partau=tau;
    matrix <double> parstart_state;
    //parstart_state=colm(trajectory[0],0);

    ///dmpは位置と速度を並べた列ベクトル
    matrix <double> dmp(DOF_number*2,1);
    matrix <double> DMP(DOF_number*2,terminal_step);

    matrix <double> f(DOF_number,terminal_step);

    matrix <double> k1;
    matrix <double> k2;
    matrix <double> k3;
    matrix <double> k4;
    matrix <double> ddmp;
    double dt=step_size;

    //初期値
   /// #pragma omp parallel for
    for(int D=0;D<DOF_number;D++){
        set_rowm(dmp,2*D)=0;
        set_rowm(dmp,2*D+1)=0;
    }

    //DMP=dmp;
    //非線形近似f
    //GPGPU
  ///  #pragma omp parallel for
    for (int i=0;i<terminal_step;i++){
            matrix <double> parGs;
    matrix <double> parthetat;
    matrix <double> parf(DOF_number,1);
            parGs=colm(Gs,i);
            parthetat=colm(thetat1,i);

        for(int D=0;D<DOF_number;D++){
            set_rowm(parf,D)=trans(rowm(parGs,range(parnum_policy*D,parnum_policy*(D+1)-1)))*rowm(parthetat,range(parnum_policy*D,parnum_policy*(D+1)-1));
        }
        set_colm(f,i)=parf;
    }

    //ルンゲクッタ
    for(int i=0;i<terminal_step;i++){
        matrix <double> temp1;
        matrix <double> temp2;

        temp1=colm(gs,i);
        temp2=colm(f,i);
        make_dmp(dmp,temp1,temp2);
        k1=dt_test;
        make_dmp(dmp+k1*dt/2,temp1,temp2);
        k2=dt_test;
        make_dmp(dmp+k2*dt/2,temp1,temp2);
        k3=dt_test;
        make_dmp(dmp+k3*dt,temp1,temp2);
        k4=dt_test;
        ddmp = (k1 + 2*k2 + 2*k3 + k4)*dt/6;
        dmp+=ddmp;
        set_colm(DMP,i)=dmp;
    }
   /// #pragma omp parallel for
    for (int D=0;D<DOF_number;D++){
        set_rowm(psi[0],D)=rowm(DMP,2*D);
    }

}


void destroyMonoBot(int p_n){//ボディの破壊

  for (int i = 0; i < leg_max; i++) {
   for(int j=0;j<3;j++){
    dBodyDestroy(Motor[p_n][i][j].body);
    dGeomDestroy(Motor[p_n][i][j].geom);
    dBodyDestroy(Pla[p_n][i][j].body);
    dGeomDestroy(Pla[p_n][i][j].geom);
   }
   for (int j=0;j<1;j++){
    dBodyDestroy(Al[p_n][i][j].body);
    dGeomDestroy(Al[p_n][i][j].geom);
   }
  }
  for(int i=0;i<num_base;i++){
    dBodyDestroy(base[p_n][i].body);
    dGeomDestroy(base[p_n][i].geom);
  }

//ジョイントの破壊
for (int i=0;i<leg_max;i++){
    for (int j=0;j<3;j++){
        dJointDestroy(Joint_Motor[p_n][i][j]);
        dJointDestroy(Joint_Pla[p_n][i][j]);
    }

    dJointDestroy(Joint_fix[p_n][i]);
}

//dJointDestroy(Joint_ff);
  leg_number=0;
}


///制御入力
void control(int thread,int time_step){
    dReal  fMax = 1.0,KD=0.01;
    double u_max,u_min;
    u_max=6.28,u_min=-6.28;
    double Av;
    if(restart_judge==0){
        for(int i=0;i<leg_max;i++){
            for(int j=0;j<3;j++){


                if(time_step%1==0){
                    angle[0+thread][i][j]=dJointGetHingeAngle(Joint_Motor[thread][i][j]);
                    Av=-(angle[0+thread][i][j]-psi[0+thread](3*i+j,time_step+1-(time_step%1)))/(step_size);
                    if(Av>10*u_max){
                        Av=10*u_max;
                    }else if(Av<10*u_min){
                    Av=10*u_min;
                    }

                    ang_vel[0+thread](3*i+j,time_step)=Av;
                        dJointSetHingeParam(Joint_Motor[thread][i][j],  dParamVel, Av);
                }
                else{
                    Av=-(angle[0+thread][i][j]-psi[0+thread](3*i+j,time_step+1-(time_step%1)))/(step_size);
                    if(Av>10*u_max){
                        Av=10*u_max;
                    }
                    else if(Av<10*u_min){
                    Av=10*u_min;
                    }
                    ang_vel[0+thread](3*i+j,time_step)=Av;
                    dJointSetHingeParam(Joint_Motor[thread][i][j],  dParamVel, Av);
                }

                dJointSetHingeParam(Joint_Motor[thread][i][j], dParamFMax, fMax);
            }
        }
    }
    else{
        double angle_r[leg_max][3];
        for(int i=0;i<leg_max;i++){
            for(int j=0;j<3;j++){

                angle_r[i][j]=dJointGetHingeAngle(Joint_Motor[thread][i][j]);
                dJointSetHingeParam(Joint_Motor[thread][i][j],  dParamVel, -angle_r[i][j]/step_size);

                dJointSetHingeParam(Joint_Motor[thread][i][j], dParamFMax, fMax);

            }
        }
    }

}


static void simLoop(int pause,int thread,int time_step){
    double angle_r[leg_max][3];

///挙動安定化のためステップ数１００までは意図的に動作しない
    if(time_step<100){
        for(int i=0;i<leg_max;i++){
            for(int j=0;j<3;j++){

                angle_r[i][j]=dJointGetHingeAngle(Joint_Motor[thread][i][j]);
                dJointSetHingeParam(Joint_Motor[thread][i][j],  dParamVel, -angle_r[i][j]/step_size);
                dJointSetHingeParam(Joint_Motor[thread][i][j], dParamFMax, 2);

            }
        }
    }
    else{

        const dReal *Rotation_base[2];
        double R1[4][10],R2[4][10];
        double vector_c[4],h_g[4],h_g_2[4];
        double inner_nolm[2];
        double mp;
        for(int i=0;i<1;i++){
            Rotation_base[i]=dBodyGetRotation(base[thread][i].body);
        }
        for(int i=1;i<=3;i++){
        if(i!=3){h_g[i]=0;}else {h_g[i]=-1;}
            vector_c[i]=0;
        }
        //printf("start\n");
        for(int i=0;i<3;i++){
            R1[1][i+1]=Rotation_base[0][i];
            R1[2][i+1]=Rotation_base[0][i+4];
            R1[3][i+1]=Rotation_base[0][i+8];
        }
        for(int l=1;l<=3;l++){
            for(int j=1;j<=3;j++){
                vector_c[l]+=R1[l][j]*h_g[j];
            }
        }
        ///ステップ数の調整
        time_step=time_step-100;

        restart_judge=0;
        const dReal *pos_base;
        const dReal *vel_base;
        pos_base=dBodyGetPosition(base[thread][0].body);
        vel_base=dBodyGetLinearVel(base[thread][0].body);
        double pos_x,pos_y;
        double vx,vy;
        double temp;
        pos_x=pos_base[0];
        pos_y=pos_base[1];
        vx=vel_base[0];
        vy=vel_base[1];
        ///temp=sqrt(vx*vx+vy*vy);
        ///temp=vy;
        double roll;
        roll=atan2(R1[3][1],R1[3][2]);
        roll=180*roll/M_PI;

        if(vector_c[1]>0){
            temp=10*vy-fabs(pos_x);
        }
        else{
            temp=10*vy-fabs(pos_x);
        }

        ///temp=sqrt(vx*vx+vy*vy);



        old_posi=pos_x;
        double read,sum_read;
        sum_read=0;
        double qq;

        ///動作確認時
        if(off_noise==1&&myutation_judge==0){
                real_judge=10000;
            for(int i=0;i<leg_max;i++){
                for(int j=0;j<3;j++){
                    double read1,read2;
                    read1=dJointGetHingeAngleRate(Joint_Motor[0][i][j]);
                    read2=dJointGetHingeAngle(Joint_Motor[0][i][j]);
                    real_angrate(time_step,3*i+j)=read1;
                    real_ang(time_step,3*i+j)=read2;

                }
            }

        }


        control(thread,time_step);

        if(time_step!=(terminal_step-1)){

                double u_limit=0;
                double torque=1;
                double u_limit_max=43-(43*torque/3.8);
                u_limit_max=u_limit_max*(2*M_PI/60);
                double to_max=3.8;
            for(int i=0;i<leg_max;i++){
                for(int j=0;j<3;j++){

                     read=ang_vel[0+thread](3*i+j,time_step);

                     if(sqrt(read*read)>u_limit_max){
                     sum_read+=10000*(sqrt(read*read)-u_limit_max);
                     }
                }
            }
        }
        else{
        distance_num[0+thread]=pos_y;
        }

        qq=sum_read;

        ///ロバスト化関連状態量の取得//////////////////////////////////////////////////////////////////////////////////

        set_colm(statecostkt[0+thread],time_step)=(-(temp-sum_read));

        distanceY=pos_y;

        //ベース位置の取得
        const dReal *Positionp;
        Positionp=dBodyGetPosition(base[0][0].body);
        temp_state(0,time_step)=Positionp[0];
        temp_state(1,time_step)=Positionp[1];
        temp_state(2,time_step)=Positionp[2];
        //ベース姿勢の取得
        const dReal *Rotationp;
        Rotationp=dBodyGetRotation(base[0][0].body);
        matrix <double,3,3> Rotation;
        Rotation=
        Rotationp[0],Rotationp[1],Rotationp[2],
        Rotationp[4],Rotationp[5],Rotationp[6],
        Rotationp[8],Rotationp[9],Rotationp[10];

        matrix <double,3,1> ex;
        ex=
        1,
        0,
        0;

        set_subm(temp_state,3,time_step,3,1)=Rotation*ex;

        temp_state(6,time_step)=dJointGetHingeAngle(Joint_Motor[0][0][0]);
        temp_state(7,time_step)=dJointGetHingeAngle(Joint_Motor[0][0][1]);
        temp_state(8,time_step)=dJointGetHingeAngle(Joint_Motor[0][0][2]);


        ///end//////////////////////////////////////////////////////////////////////////////////////////

    }
    dSpaceCollide(space[thread],0,&nearCallback);
    dWorldStep(world[thread],step_size);
    dJointGroupEmpty(contactgroup[thread]);
}


static void start(){
  static float xyz[3] = {   0.8, 0.0, 0.5};
  static float hpr[3] = {-180.0, -30.0, 0.0};
  dsSetViewpoint(xyz,hpr);               // 視点，視線の設定
  dsSetSphereQuality(3);                 // 球の品質設定
}




int main(int argc, char *argv[]){
    thread_number=omp_get_max_threads();
    distribution_p=initial_dist;
    time(&timer1);
    umax=10;
    ref_state=zeros_matrix<double>(state_number,terminal_step);
///寸法パラメータ//////////////////////////////////////////////////////////////////////////////////////////////////////
    Pla_ly_2=37*0.001;
    Pla_lx_2=23*0.001;
    Pla_lz_2=28*0.001;
    base_r=0.064,base_lz=0.086,base_m=(705-90*3-9*3-24)*0.001;
    Motor_ly=0.034,Motor_lx=0.0465,Motor_lz=0.0285;
    Pla_ly=0.041,Pla_lz=20*0.001,Pla_lx=0.028;
    Al_ly[0]=0.039,Al_lx[0]=0.002,Al_lz[0]=0.12;
    Al_ly[1]=0.039,Al_lx[1]=0.013,Al_lz[1]=0.002;
    Motor_m=90*0.001;
    Pla_m=0.009;
    Al_m[0]=24*0.001;
    Al_l=(91-7.632)*0.001;
    Al_r=7.632*0.001;
    for (int i=0;i<num_base;i++){
        base_x[i]=0,base_y[i]=0;base_z[i]=base_r;
    }
    Motor_y[0]=0;
    Motor_x[0]=-Pla_lx-(Motor_lz/2);
    Motor_z[0]=base_z[0]-(46.5/2-11.25)*0.001;
    Motor_x[1]=(Motor_lx/2-11.25*0.001);
    Joint_M_x[0]=-Pla_lx;
    Joint_M_y[0]=0;
    Joint_M_z[0]=base_z[0];
    Joint_M_x[1]=Motor_x[1]+11.25*0.001-Motor_lx/2;
    Joint_M_y[1]=0;
    Motor_x[2]=Joint_M_x[1]+(84-46.5/2+11.25)*0.001;
    Motor_y[2]=0;
    Joint_M_x[2]=Joint_M_x[1]+81*0.001;
    Joint_M_y[2]=0;
    Pla_x[0]=-Pla_lx/2;
    Pla_y[0]=0;
    Pla_x[1]=Motor_x[1]+(Motor_lx)/2+4.138*0.001;
    Pla_y[1]=0;
    Pla_x[2]=Joint_M_x[2]+Pla_lx/2;
    Al_x[0]=Pla_x[2]+(Pla_lx+Al_l)/2;
    Al_y[0]=0;
    Pla_z[0]=base_z[0];
    Motor_z[1]=Pla_z[0];
    Joint_M_z[2]=base_z[0];
    Joint_M_z[1]=Motor_z[1];
    Motor_z[2]=Motor_z[1];
    Pla_z[1]=Motor_z[1];
    Pla_z[2]=Motor_z[1];
    Al_z[0]=Motor_z[1];
    Al_friction=0.6;
    Pla_friction=0.5;
    grav=9.8;
///end////////////////////////////////////////////////////////////////////////////////////////////////////////////

    dInitODE();

    world[0]=dWorldCreate();
    space[0]=dHashSpaceCreate(0);
    contactgroup[0]=dJointGroupCreate(0);
    dWorldSetGravity(world[0],0,0,-grav);
    ground[0]=dCreatePlane(space[0],0,0,1,0);
    dWorldSetERP(world[0], 0.2);          // ERPの設定
    dWorldSetCFM(world[0], 1e-4);         // CFMの設定



    E=identity_matrix<double>(all_policy_num);
    S=W_R*E;
    sigma  =1* W_sigma *E;



    double dt;
    double phi_T;
    dt=step_size;
    tf=dt*terminal_step;
    phi_T=tf/3;
    tau=phi_T/(2*M_PI);

    for(int i=0;i<terminal_step;i++){
        t(0,i)=i;
    }
    printf("E1");

    alphaq = 25;
    r0 = 25*25/4; // 初期振幅
    rg = 25*25/4; // 目標振幅
    //phi0 = 0; // 初期位相
    printf("E2");


    ///make_rythmic
    make_rhythmic(r0,rg);


    //printf("E25");
    //多自由度への拡張

    for(int D=0;D<DOF_number;D++){
        set_rowm(ws1,range(D*num_policy,(D+1)*num_policy-1))=w;
    }//多自由度への拡張
    for(int D=0;D<DOF_number;D++){
        set_rowm(Gs,range(D*num_policy,(D+1)*num_policy-1))=G;
    }



    alphag=alphaq/2;
    double goal_D[DOF_number];
    double start_D[DOF_number];

    goal_D[0]=0;
    start_D[0]=0;
    for(int i=0;i<DOF_number;i++){
        goal_D[i]=0;
        start_D[i]=0;
    }
        ///psi関係でエラー？　今日はここまで
    for(int D=0;D<DOF_number;D++){
        g  = (start_D[D] - goal_D[D]) * exp((-alphag)*tt/tau)+goal_D[D]*ones_matrix<double>(1,terminal_step);
        set_rowm(gs,D)=g;
    }



    WN2=0;
    printf("E3");
    for(int i=0;i<terminal_step;i++){
        set_colm(t,i)=i;
    }

    for(int i=0;i<terminal_step;i++){
        WN2(0,i)=(terminal_step-i-1);
        WN1(i,i)=terminal_step-(i+1);
    }

        //重み行列の作成
            //omploop

    printf("3.5");
        ///初期状態のMでエラーが発生

        #pragma omp parallel for
    for(int i=0;i<terminal_step;i++){

        matrix <double> M_test,M_test2;
        double test;
        M_test=trans(colm(Gs,i))*inv(S)*colm(Gs,i);
        M_test2=inv(M_test);
        test=M_test2(0,0);
        M[i]=(inv(S)*colm(Gs,i)*trans(colm(Gs,i)))*test;
    }
    printf("E4");

    //多変量正規分布の作成 clear
    meanzerot=zeros_matrix<double>(all_policy_num,1);
    printf("E5\n");
    restart_judge=0;


    temp_cost=0;
    off_noise=1;


///出力ファイルの生成////////////////////////////////////////////////////////////////////////////////////////////////
    outputfile20=fopen("state_Xt.txt","w");
    outputfile21=fopen("statecost.txt","w");
    outputfile22=fopen("ref_state.txt","w");
    outputfile23=fopen("sumcost_nominal.txt","w");
    outputfile24=fopen("sumcost_addnoise.txt","w");
    outputfile25=fopen("experiment_data.txt","w");
    outputfile26=fopen("for_plot.txt","w");
    outputfile27=fopen("experiment_data2.txt","w");
///入力ファイルの読み込み/////////////////////////////////////////////////////////////////////////////////////////////
    inputfile1=fopen("policy.txt","r");
    if(inputfile1==NULL){
        printf("file open error");
    }

    matrix<double,max_trial_num,all_policy_num> everytra;
    double data;
    for(int i=0;i<max_trial_num;i++){
        for(int j=0;j<all_policy_num;j++){
            fscanf(inputfile1,"%lf,",&data);
            everytra(i,j)=data;
        }
    }
    fclose(inputfile1);



///end///////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(int noise=0;noise<param_samp_num;noise++){
        if(noise!=0){
    ///noiseありの場合のパラメータ変化//////////////////////////////////////////////////////////////////////////////////////
            Pla_friction=0.5+rand.get_random_gaussian()*(std_ratio*Pla_friction);
    ///end///////////////////////////////////////////////////////////////////////////////////////////////////////////
        }
        for(int s=0;s<max_trial_num;s++){
            printf("loop=%d\n",s+1);

            set_colm(theta,0)=trans(rowm(everytra,s));
            ///モデル生成
            make(0);
            ///モータへの摂動を除去した入力生成
            excute_policy(theta,zeros_matrix<double>(all_policy_num,terminal_step));
            ///シミュレータの動作
            for(int j=0;j<terminal_step+100;j++){
                simLoop(0,0,j);
            }
            if(noise==0){
                state_vector[s]=temp_state;
            }

            if(noise==0){
                pre_distanceY(0,s)=distanceY;
            }
            else{
                post_distanceY(0,s)=distanceY;
            }
            printf("TotalCost=%.5f\n",sum(statecostkt[0]));
            printf("y=%.5f m\n\n",distanceY);

            ///動力学モデルの初期化
            destroyMonoBot(0);
            dJointGroupDestroy(contactgroup[0]);     // ジョイントグループの破壊
            contactgroup[0] = dJointGroupCreate(0);  // ジョイントグループの生成

            if(noise==0){
                fprintf(outputfile20,"%d,\n",s+1);
                pre_statecost[s]=statecostkt[0];
                for(int i=0;i<state_number;i++){
                    for(int j=0;j<terminal_step;j++){
                       fprintf(outputfile20,"%.32f,",state_vector[s](i,j));
                    }
                    fprintf(outputfile20,"\n\n");
                }
                fprintf(outputfile20,"\n");
            }
            else{
                post_statecost[s]=statecostkt[0];
            }
        }
    }
    fclose(outputfile20);

    double pre_cost_mean=0.0;
    double pre_cost_vari=0.0;
    double post_cost_mean=0.0;
    double post_cost_vari=0.0;
    double pre_dist_mean=0.0;
    double pre_dist_vari=0.0;
    double post_dist_mean=0.0;
    double post_dist_vari=0.0;
    double delta_mean=0.0;
    double delta_vari=0.0;
    //コスト変化量
    double delta[max_trial_num];
    double minimum=100000;
    double maximum=0.0;

    for(int s=0;s<max_trial_num;s++){
        fprintf(outputfile21,"%d,\n",s+1);
        for(int i=0;i<terminal_step;i++){
            fprintf(outputfile21,"%.32f,",pre_statecost[s](0,i));
        }
        fprintf(outputfile23,"tra%d cost=%.32f    y=%.32f,\n",s+1,sum(pre_statecost[s]),pre_distanceY(0,s));
        fprintf(outputfile24,"tra%d cost=%.32f    y=%.32f,\n",s+1,sum(post_statecost[s]),post_distanceY(0,s));
        fprintf(outputfile21,"\n\n");
        for(int i=0;i<terminal_step;i++){
            fprintf(outputfile21,"%.32f,",post_statecost[s](0,i));
        }
        fprintf(outputfile21,"\n\n\n");

        pre_cost_mean+=sum(pre_statecost[s])/max_trial_num;
        pre_cost_vari+=pow(sum(pre_statecost[s]),2.0)/max_trial_num;
        post_cost_mean+=sum(post_statecost[s])/max_trial_num;
        post_cost_vari+=pow(sum(post_statecost[s]),2.0)/max_trial_num;

        pre_dist_mean+=pre_distanceY(0,s)/max_trial_num;
        pre_dist_vari+=pow(pre_distanceY(0,s),2.0)/max_trial_num;
        post_dist_mean+=post_distanceY(0,s)/max_trial_num;
        post_dist_vari+=pow(post_distanceY(0,s),2.0)/max_trial_num;




        ///コスト変化の計算
        delta[s]=sum(post_statecost[s])-sum(pre_statecost[s]);
        weight[s]=fabs(delta[s])+ration*sum(pre_statecost[s]);//重みの計算

        delta_mean+=(fabs(delta[s]))/max_trial_num;
        delta_vari+=(fabs(delta[s])*fabs(delta[s]))/max_trial_num;
        if(weight[s]<minimum){
            minimum=weight[s];
        }
        if(weight[s]>=maximum){
            maximum=weight[s];
        }

    }

    pre_cost_vari-=pre_cost_mean*pre_cost_mean;
    post_cost_vari-=post_cost_mean*post_cost_mean;
    pre_dist_vari-=pre_dist_mean*pre_dist_mean;
    post_dist_vari-=post_dist_mean*post_dist_mean;
    delta_vari-=delta_mean*delta_mean;


    for(int s=0;s<max_trial_num;s++){
        fprintf(outputfile25,"%f %f\n",sum(pre_statecost[s]),fabs(delta[s]));
        double temp=(weight[s]-minimum)/(maximum-minimum);
        Pi(0,s)=exp(-alphaP*temp);
        //Pi(0,s)=exp(-alphaP*(fabs(delta[s]/sum(pre_statecost[s]))-minimum)/(maximum-minimum));
        fprintf(outputfile26,"%f 0\n",temp);
    }
    fclose(outputfile26);

    fprintf(outputfile27,"#Psi\n");
    fprintf(outputfile27,"0 %f %f\n\n",pre_cost_mean,sqrt(pre_cost_vari));
    fprintf(outputfile27,"#|DeltaPsi|\n");
    fprintf(outputfile27,"0 %f %f",delta_mean,sqrt(delta_vari));
    fclose(outputfile27);


    fprintf(outputfile25,"\n\n%f %f",pre_cost_mean,delta_mean);

    fclose(outputfile25);




    fprintf(outputfile23,"\n");
    fprintf(outputfile24,"\n");
    fprintf(outputfile23,"mean=%.32f    mean=%.32f\n",pre_cost_mean,pre_dist_mean);
    fprintf(outputfile23,"variance=%.32f    variance=%.32f",pre_cost_vari,pre_dist_vari);
    fprintf(outputfile24,"mean=%.32f    mean=%.32f\n",post_cost_mean,post_dist_mean);
    fprintf(outputfile24,"variance=%.32f    variance=%.32f",post_cost_vari,post_dist_vari);

    fclose(outputfile21);
    fclose(outputfile23);
    fclose(outputfile24);

    Pi=Pi/sum(Pi);


    ///参照状態量の計算
    for(int t=0;t<terminal_step;t++){
        matrix <double> sum_temp;
        sum_temp=zeros_matrix<double>(state_number,1);
        matrix <double> temp;
        for(int s=0;s<max_trial_num;s++){
            sum_temp+=Pi(0,s)*colm(state_vector[s],t);
        }

        set_colm(ref_state,t)=sum_temp;
    }

    for(int i=0;i<state_number;i++){
        for(int j=0;j<terminal_step;j++){
            fprintf(outputfile22,"%.32f,",ref_state(i,j));
        }
        fprintf(outputfile22,"\n\n");
    }
    fclose(outputfile22);

    dCloseODE();
    return 0;
}
