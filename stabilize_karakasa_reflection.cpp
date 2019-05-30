// 簡単！実践！ロボットシミュレーション
// Open Dynamics Engineによるロボットプログラミング
// 出村公成著, 森北出版 (2007) http://demura.net/
// このプログラムは上本のサンプルプログラムです．
// プログラム 2.9:  再実行可能なプログラム hopper2.cpp by Kosei Demura (2007-5-17)
//
// This program is a sample program of my book as follows
//“Robot Simulation - Robot programming with Open Dynamics Engine,
// (260pages, ISBN:978-4627846913, Morikita Publishing Co. Ltd.,
// Tokyo, 2007)” by Kosei Demura, which is written in Japanese (sorry).
// http://demura.net/simulation
// Please use this program if you like. However it is no warranty.
// hello2.cpp by Kosei Demura (2007-5-18)
//
// 更新履歴　(change log)
// 2008-7-7: dInitODE(),dCloseODE()の追加
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
#define num num_policy*DOF_number
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

///ロバスト化関連///////////////////////////////////////////////////////////////////////////////
#define state_number 9
#define reflection_ON 1
#define palallel_p 4
///end////////////////////////////////////////////////////////////////////////////////////////
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
FILE *outputfile1[max_trial_num],*outputfile2[max_trial_num],*outputfile3[max_trial_num],*outputfile4[max_trial_num],*outputfile5[max_trial_num],*outputfile6[max_trial_num],*outputfile7[max_trial_num],*outputfile8[max_trial_num],*outputfile9[max_trial_num],*outputfile10[max_trial_num],*outputfile11[max_trial_num],*outputfile12[max_trial_num],*outputfile13[max_trial_num],*outputfile14[max_trial_num],*outputfile15[max_trial_num],*outputfile16[max_trial_num],*outputfile17[masatu_num],*outputfile18[max_trial_num],*outputfile19[masatu_num];
FILE *output_goal;
FILE *inputfile1;
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
matrix <double> E(num,num);
matrix <double> S(num,num);
matrix <double> sigma(num,num);
dReal lamda;
matrix <double,num_policy,terminal_step> epst[K_limit];
matrix <double,num,1> meanzerot;
matrix<double> WN2(1,terminal_step);

int K_number=0;
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
matrix <double> theta(num,1);

matrix<double,1,terminal_step> immediate_cost[K_limit];

matrix <double> dthetait(num,terminal_step);

int off_noise=0;
double sum_controlcost[K_limit],sum_statecost[K_limit];
double best_cost,best_taskcount,best_update;
matrix <double> best_theta(num,1);

matrix <double,num,terminal_step> eps[K_limit];

double cost[K_limit];
double taskcount;
matrix <double> meanzero[K_limit];
matrix <double> dthetai(num,terminal_step);
matrix <double> parimmediatecostk[K_limit];
matrix <double> parterminalcostk[K_limit];
matrix <double> thetat;
matrix <double,num,num> exe_theta[K_limit];
matrix <double> dtheta(num,1);
int h;

//要変更
matrix <double,num,num> M[terminal_step];

//
double check;
int errort;
double cost1,best_cost1,best_cost2;
matrix <double> best_trajectory1;
matrix <double> WN1(terminal_step,terminal_step);
matrix <double> dthetat(num,1);
matrix <double,num,terminal_step> t_epst[palallel_p];
matrix <double> One_temp;
matrix <double,num,num> temp_exetheta[K_limit];
matrix <double> t(1,terminal_step);
matrix <double> st(1,terminal_step);
matrix <double> w(num_policy,terminal_step);
matrix <double> G(num_policy,terminal_step);
matrix <double> ws1(num,terminal_step);
matrix <double> Gs(num,terminal_step);
matrix <double> g0(DOF_number,1);
matrix <double> gg(DOF_number,1);
matrix <double> g(1,terminal_step);
matrix <double> gs(DOF_number,terminal_step);
matrix <double,DOF_number,terminal_step> psi[K_limit];
matrix <double> rr0(DOF_number*2,1);
matrix <double> rrg(DOF_number*2,1);
matrix <double> tt(1,terminal_step);
matrix <double> dt_test[palallel_p];
matrix <double> kernel_output;
matrix <double> k_st;
matrix <double> temp_K(division,K_limit);
matrix <double> Gausian_kernel;
matrix <double,num,terminal_step> theta_k[K_limit];



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
matrix <double> p_c(num,1);
double cp;
double p_succ,p_succ_ave;

double damp,distribution,distribution_p;
double c_cov,c_c,p_thresh;
double p_succ_target;
double a_p,a_best;
int judge_CMA_ES;
int lamda_count;
matrix <double> theta_best(num,1);
int CMA_initial=0;
int CMA_PI_JUDGE=0;
matrix <double> old_theta(num,1);
double old_distribution=1.1;
int myutation_judge=0;
double temp_dist=0;
int myutation_occur=0;
int epic_myutation_judge=0;
int K_max_rest;
matrix <double> theta_CMA_best(num,1);
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
matrix <double,3*leg_max,terminal_step> real_vel[500];
double old_cost=0;
///===================================

double a_p_best=0;
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
double Al_friction;
matrix<double> copy_m(terminal_step,4);
matrix<double>excel(terminal_step,4*max_trial_num);
double distance_maximam;
int m_count=0;
int sample_total=0;
double myu_step=sqrt(initial_myu_step);
double myu_step_d=0.5;
int jal=0;
double Pla_friction;
double base_under_m=10*0.001;

matrix <double> CMA_old_theta(num,1);

///ロバスト化関連/////////////////////////////////////////////////////////////////////////////////////////////////////////////


//罰則重みづけ行列Wp
matrix <double> Wp(state_number,state_number);

//参照状態量Xref
matrix <double,state_number,terminal_step> ref_state;

matrix <double,state_number,1> state_vector;

double posPena=0.001;//0.001//0.003//0.003


double rotPena=0.001;//=0.001//0.003//0.003

double angPena=0.00005;//=0.00005//0.0001//0.00005

double grav;
///end//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void mvnrnd(int k,matrix <double> meanzerot1,matrix <double>E1){
 //多変量正規分布の作成 clear
double test;
matrix <double,num,1> normal_vector[3];
//r.get_random_gaussian();
test=rnd.get_random_gaussian();
   // for (int i=0;i<terminal_step;i++){
            for(int j=0;j<num;j++){

               normal_vector[0](j,0)= rnd.get_random_gaussian();

            }
    //}
    typedef matrix <double> matrix_exp_type;

     matrix <double> FF[terminal_step];
    matrix <double> GG[terminal_step];

        cholesky_decomposition<matrix_exp_type> cholesky_decomposition(E1);
        matrix <double,num,num> temp_epst;
        temp_epst=cholesky_decomposition.get_l();
        //FF[0]=temp_epst*normal_vector[0]+meanzerot1;
    //#pragma omp parallel for
   // for(int i=0;i<terminal_step;i++){

        FF[0]=temp_epst*normal_vector[0]+meanzerot1;

    //}
    //cout<<temp_epst*normal_vector[0]<<endl;
    for(int i=0;i<terminal_step;i++){
        set_colm(t_epst[k],i)=FF[0];
    }

}

// 衝突検出計算
static void nearCallback(void *data, dGeomID o1, dGeomID o2)
{
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
void transR(){

}

void createleg(int p_n){
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
 Motor[p_n][leg_number][i].body=dBodyCreate(world[p_n]);
 if(i==0)
   {dMassSetBoxTotal(&mass,Motor_m,Motor_lz,Motor_lx,Motor_ly);}
else{ dMassSetBoxTotal(&mass,Motor_m,Motor_lx,Motor_ly,Motor_lz);}
 dBodySetMass(Motor[p_n][leg_number][i].body,&mass);
//dBodySetRotation(Motor[p_n][leg_number][i].body,RR);

// if (i!=0){dBodySetRotation(Motor[leg_number][i].body,R);}

 dBodySetPosition(Motor[p_n][leg_number][i].body,Motor_x[i],Motor_y[i],Motor_z[i]);



if(i==0)
    {
        dBodySetRotation(Motor[p_n][leg_number][i].body,R);
        ///Motor[p_n][leg_number][i].geom=dCreateBox(space[p_n],Motor_lx,Motor_lz,Motor_ly);
    Motor[p_n][leg_number][i].geom=dCreateBox(space[p_n],Motor_lz,Motor_lx,Motor_ly);
    /// dGeomSetBody(Motor[p_n][leg_number][i].geom,Motor[p_n][leg_number][i].body);
    Joint_fix[p_n][leg_number]=dJointCreateFixed(world[p_n],0);
    dJointAttach(Joint_fix[p_n][leg_number],Motor[p_n][leg_number][0].body,base[p_n][0].body);
    dJointSetFixed(Joint_fix[p_n][leg_number]);
    }
else {dBodySetRotation(Motor[p_n][leg_number][i].body,R);
        Motor[p_n][leg_number][i].geom=dCreateBox(space[p_n],Motor_lx,Motor_ly,Motor_lz);

}

dGeomSetBody(Motor[p_n][leg_number][i].geom,Motor[p_n][leg_number][i].body);

 }
 for(int i=0;i<3;i++){
         //接続用のプラスチック素材の作成
 Pla[p_n][leg_number][i].body=dBodyCreate(world[p_n]);
 if(i==1){

//        dMassSetBoxTotal(&mass,Pla_m,Pla_lz,Pla_ly,Pla_lx);
//  Pla[p_n][leg_number][i].geom=dCreateBox(space[p_n],Pla_lz,Pla_ly,Pla_lx);
   dMassSetBoxTotal(&mass,Pla_m,Pla_lx_2,Pla_ly_2,Pla_lz_2);

 Pla[p_n][leg_number][i].geom=dCreateBox(space[p_n],Pla_lx_2,Pla_ly_2,Pla_lz_2);

 }
 else{
        dMassSetBoxTotal(&mass,Pla_m,Pla_lx,Pla_ly,Pla_lz);
 Pla[p_n][leg_number][i].geom=dCreateBox(space[p_n],Pla_lx,Pla_ly,Pla_lz);
 }
 dBodySetMass(Pla[p_n][leg_number][i].body,&mass);
 dBodySetRotation(Pla[p_n][leg_number][i].body,R);

 dBodySetPosition(Pla[p_n][leg_number][i].body,base_x[0]+Pla_x[i],base_y[0]+Pla_y[i],Pla_z[i]);


 dGeomSetBody(Pla[p_n][leg_number][i].geom,Pla[p_n][leg_number][i].body);


 }
     //アルミの足先の作成
    // r3[i]=sqrt((base_x[leg_number/3]-Al_x[i])*(base_x[leg_number/3]-Al_x[i])+(base_y[leg_number/3]-Al_y[i])*(base_y[leg_number/3]-Al_y[i]));
 Al[p_n][leg_number][0].body=dBodyCreate(world[p_n]);
 dMassSetCapsuleTotal(&mass,Al_m[0],2,Al_r,Al_l);
 dBodySetRotation(Al[p_n][leg_number][0].body,TA);
 dBodySetMass(Al[p_n][leg_number][0].body,&mass);
 //dBodySetRotation(Al[p_n][leg_number][0].body,RR);
 dBodySetPosition(Al[p_n][leg_number][0].body,Al_x[0]+base_x[0],Al_y[0]+base_y[0],Al_z[0]);
 Al[p_n][leg_number][0].geom=dCreateCapsule(space[p_n],Al_r,Al_l);
 dGeomSetBody(Al[p_n][leg_number][0].geom,Al[p_n][leg_number][0].body);
 //各パーツの接続
 for (int i=0;i<3;i++){
    //モーター回転用
    Joint_Motor[p_n][leg_number][i]=dJointCreateHinge(world[p_n],0);
    if(i==0){
        dJointAttach(Joint_Motor[p_n][leg_number][i],base[p_n][leg_number].body,Pla[p_n][leg_number][0].body);
    }
    else if(i==2){
        dJointAttach(Joint_Motor[p_n][leg_number][i],Motor[p_n][leg_number][i].body,Pla[p_n][leg_number][i].body);
    }else{
    dJointAttach(Joint_Motor[p_n][leg_number][i],Motor[p_n][leg_number][i].body,Pla[p_n][leg_number][i-1].body);
    }

    dJointSetHingeAnchor(Joint_Motor[p_n][leg_number][i],Joint_M_x[i]+base_x[0],base_y[0]+Joint_M_y[i],Joint_M_z[i]);
  if(i==0) { dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],1,0,0);}else{
//        switch (b){
//      case 0:  dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],1,-sqrt(3),0); break;
//      case 1:  dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],1,sqrt(3),0); break;
//      case 2:  dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],1,0,0); break;
//      }
 dJointSetHingeAxis(Joint_Motor[p_n][leg_number][i],0,0,1);
        }
        if(i==0){
         dJointSetHingeParam(Joint_Motor[p_n][leg_number][i],dParamLoStop,-M_PI+0.005);
    dJointSetHingeParam(Joint_Motor[p_n][leg_number][i],dParamHiStop,M_PI-0.005);
        }else{
         dJointSetHingeParam(Joint_Motor[p_n][leg_number][i],dParamLoStop,-M_PI+0.005);
    dJointSetHingeParam(Joint_Motor[p_n][leg_number][i],dParamHiStop,M_PI-0.005);

        }



 }

 for(int i=0;i<3;i++){
    //プラ固定用
    Joint_Pla[p_n][leg_number][i]=dJointCreateFixed(world[p_n],0);
   if(i!=2) {dJointAttach(Joint_Pla[p_n][leg_number][i],Pla[p_n][leg_number][1].body,Motor[p_n][leg_number][i+1].body);}
   else {
        dJointAttach(Joint_Pla[p_n][leg_number][i],Pla[p_n][leg_number][i].body,Al[p_n][leg_number][0].body);
   }
    dJointSetFixed(Joint_Pla[p_n][leg_number][i]);
 }

leg_number++;

}

void createMonoBot(int p_n) {
  dMass mass;
  dMatrix3 TA;
  dRFromAxisAndAngle(TA,0,1,0,M_PI/2);

for (int i=0;i<num_base;i++){
base[p_n][i].body=dBodyCreate(world[p_n]);
dMassSetCylinderTotal(&mass,base_m,2,base_r,base_lz);
dBodySetMass(base[p_n][i].body,&mass);
dBodySetPosition(base[p_n][i].body,base_x[i]-base_lz/2-Pla_lx,base_y[i],base_z[i]);
dBodySetRotation(base[p_n][i].body,TA);
base[p_n][i].geom=dCreateCylinder(space[p_n],base_r,base_lz);
dGeomSetBody(base[p_n][i].geom,base[p_n][i].body);


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

///RBFカーネル作成
void Kernel(matrix <double> trajectoryk,double hyper_pt,matrix <double> xt,int d){

matrix <double,division+1,1> x[terminal_step];
matrix <double> kstt(d,1);
matrix <double> kernel_output_temp(d,d);
//printf("ss1");
///trajectoryからベクトル抽出
for(int i1=0;i1<d;i1++){
    x[i1]=colm(trajectoryk,i1);
}
for(int i1=0;i1<d;i1++){
        set_rowm(kstt,i1)=exp(-hyper_pt*((trans(x[i1]-xt))*(x[i1]-xt)));
}
k_st=kstt;
for(int i1=0;i1<d;i1++){
    for(int j=0;j<d;j++){
        kernel_output_temp(i1,j)=exp(-hyper_pt*(trans(x[j]-x[i1])*(x[j]-x[i1])));
    }
}
kernel_output=kernel_output_temp;
        kt=exp(-hyper_pt*(trans(xt)*(xt)));
        kt=1;
}
void ALD_algorithm(int P_P){

//matrix <double> xt(DOF_number*2,1);
matrix <double> tra_k(division+1,1);
matrix <double> t_k(1,terminal_step);
matrix <double> ttt(1,terminal_step);
matrix <double> cost_division(division,1);
matrix <double> temp1(1,terminal_step);
matrix <double> temp2(1,terminal_step);
for(int i=0;i<terminal_step;i++){
    ttt(0,i)=i*step_size;
}
//for(int j=1;j<terminal_step;j++){
//    sum_cost(0,j)=sum_cost(0,j-1)+statecostkt[K_number](0,j);
//}
 //set_rowm(cost_division,0)=sum_cost[P_P](0,0);
 //#pragma omp parallel for
for(int i=0;i<division;i++){
  set_rowm(cost_division,i)=statecostkt[K_number+P_P](0,(terminal_step/division-1)+i*terminal_step/division);
}
K_nolm=sqrt(trans(cost_division)*cost_division);
//if(max(cost_division)>sqrt(min(cost_division)*min(cost_division))){
//     set_rowm(tra_k,range(0,division-1))=(cost_division)/sqrt(max(cost_division)*max(cost_division));
//}else{
//set_rowm(tra_k,range(0,division-1))=(cost_division)/sqrt(min(cost_division)*min(cost_division));
//}
 set_rowm(tra_k,range(0,division-1))=normalize(cost_division);
 //set_rowm(tra_k,division)=K_nolm;
 //printf("cost_divisions\n");
 //cout<<(cost_division)<<endl;

double myu=0.001;
if((K_number+P_P)==0){
    judge_number=10000;
    set_colm(temp_K,0)=rowm(tra_k,range(0,division-1));
    //printf("first_tempK\n");
    //cout<<colm(temp_K,0)<<endl;
    set_colm(K_nolm_m,0)=K_nolm;
    //printf("K_nolm_first=%f\n",K_nolm);
    //initial_set=10000;
}
else{
//printf("s3");
matrix <double> kst;
matrix <double> ct;
matrix <double> test(division+1,d);
matrix <double> ktt(1,1);
double deltat;
matrix <double> max_matrix;
matrix <double> xt(division+1,1);
//if(i==0){
//    set_colm(temp_K,0)=colm(trajectoryk,0);
//}

///xt,vtのセット
if(myutation_judge==10000){
//printf("K_nolm[%d]=%f\n",K_number+P_P,K_nolm);
}
set_colm(temp1,range(0,d-1))=colm(K_nolm_m,range(0,d-1));
set_colm(temp1,d)=K_nolm;
temp2=normalize(temp1);
//printf("temp2\n");
//cout<<temp2<<endl;
//xt=normalize(tra_k);
set_rowm(xt,range(0,division-1))=rowm(tra_k,range(0,division-1));
set_rowm(xt,division)=colm(temp2,d);

//test=colm(temp_K,range(0,d-1));
matrix <double> temp_K_temp(division+1,K_limit);
set_rowm(temp_K_temp,range(0,division-1))=temp_K;
set_colm(test,range(0,d-1))=colm(temp_K_temp,range(0,d-1));

set_rowm(test,division)=colm(temp2,range(0,d-1));

max_matrix=abs(test-xt*ones_matrix<double>(1,d));
hyper_p=0.9;
//printf("test\n");
//cout<<test<<endl;
//printf("temp_K\n");
//cout<<colm(temp_K,range(0,d-1))<<endl;

//printf("temp2\n");
//cout<<temp2<<endl;

Kernel(test,hyper_p,xt,d);
kst=k_st;
ktt=kt;
//K[1]=kernel_output;

ct=inv(kernel_output)*kst;
//printf("kernel\n");
//cout<<kernel_output<<endl;
double old_deltat;
double old_ddeltat;
old_deltat=deltat;
old_ddeltat=ddeltat;
deltat=ktt(0,0)-trans(kst)*ct;

if(deltat>=0){


if(deltat>1){
          printf("eroor_deltat\n");
            cout<<deltat<<endl;
        judge_number=10000;
        deltat=0;
    deltat=old_deltat;
    ddeltat=old_ddeltat;
}else{
if(myutation_judge==10000){
   //printf("deltat");
//cout<<deltat<<endl;
if(deltat>1){
        //judge_number=10000;
        judge_number=10000;
//    printf("eroor\n");
//    cout<<trans(kst)*ct<<endl;
//    printf("k_st\n");
//    cout<<kst<<endl;
}
}

//printf("K_nolm=%f\n",K_nolm);
//printf("xt\n");
//cout<<xt<<endl;

//printf("deltat");
//cout<<deltat<<endl;
if(deltat>ALD_thred){

    judge_number=10000;
    set_colm(temp_K,d)=rowm(xt,range(0,division-1));
    set_colm(K_nolm_m,d)=K_nolm;
    ddeltat=deltat;
    d++;
}
else{

    ddeltat=abs(deltat-ddeltat);
    if(myutation_judge==10000){
       //printf("ddeltat=%f\n",ddeltat);
    }
    //printf("ddeltat=%f\n",ddeltat);
    if(ddeltat>ALD_thred/2){
         judge_number=10000;
       //  set_colm(temp_K,d)=rowm(xt,range(0,division-1));
        //set_colm(K_nolm_m,d)=K_nolm;
         // d++;
    }
    ddeltat=deltat;
}

}
}
else{
    if(deltat>-10000){
            printf("eroor_deltat\n");
            cout<<deltat<<endl;
          ddeltat=abs(deltat-ddeltat);
           if(ddeltat>ALD_thred/2){
         judge_number=10000;
       //  set_colm(temp_K,d)=rowm(xt,range(0,division-1));
        //set_colm(K_nolm_m,d)=K_nolm;
         // d++;
    }
       deltat=old_deltat;
    ddeltat=old_ddeltat;

    }else{
        //judge_number=10000;
        //printf("test\n");
//cout<<test<<endl;
//printf("temp_K\n");
//cout<<colm(temp_K,range(0,d-1))<<endl;
//printf("K_nolm_m\n");
//cout<<K_nolm_m<<endl;
//printf("temp2\n");
//cout<<temp2<<endl;
//     printf("eroor\n");
//printf("deltat");
//cout<<deltat<<endl;
    //deltat=0;
judge_number=10000;
    deltat=old_deltat;
    ddeltat=old_ddeltat;
    }


//printf("deltat");
//cout<<deltat<<endl;

}
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
phi=phi0*ones_matrix<double>(1,terminal_step)+tt/tau;
double alphag = alphaq/2;

///st調整 tt➡terminal stが目標値に

 st = (r0 - rg) * exp(-alphag*(tt)/tau) + r0 * ones_matrix<double>(1,terminal_step);
//printf("s4");
if(st(0,0)==0){
    st(0,0)=0.001;

}

matrix <double> temp[num_policy];
//for(int i=0;i<num_policy;i++){
//    temp[i]=(st-c(i,0)*ones_matrix<double>(1,terminal_step));
//    temp[i](0,i)=temp[i](0,i)*temp[i](0,i);
//
//}

for(int i=0;i<num_policy;i++){
//cout<<cos(phi-c(i,0)*ones_matrix<double>(1,terminal_step))<<endl;
}
///w修正必要　
 for(int i=0;i<num_policy;i++){
        ///cosの中身調整　
    set_rowm(w,i)=exp(h*(cos(phi-c(i,0)*ones_matrix<double>(1,terminal_step)))-ones_matrix<double>(1,terminal_step));
//printf("part1[%d]\n",i);
 //cout<< (st-c(i,0)*ones_matrix<double>(1,terminal_step))*trans(st-c(i,0)*ones_matrix<double>(1,terminal_step))<<endl;
 }

 //GPGPU

 for(int i=0;i<terminal_step;i++){
        if(sum(colm(w,i))==0){
    printf("w_eroor");
}
    set_colm(G,i)=colm(w,i)*st(0,i)/sum(colm(w,i));
 }

}
void make_dmp(int k,matrix <double> dmp,matrix <double> gsi,matrix <double> fi){

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
dt_test[k]=d_dmp;
}

void excute_policy(int k,matrix <double> thetag,matrix<double> noise){
   // printf("thetat\n");
//printf("e1");
matrix <double> thetat1;
thetat1=thetag*ones_matrix<double>(1,terminal_step)+noise;
theta_k[k]=thetat1;
//cout<<thetat<<endl;
    double parnum_policy =num_policy;
    double parnum_DOFs=DOF_number;
    double parnum_base=num_base;
    double partau=tau;
    matrix <double> parstart_state;
    //parstart_state=colm(trajectory[0],0);
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
        matrix <double> temp3;
        matrix <double> temp4;
        temp1=colm(gs,i);
        temp2=colm(f,i);
        make_dmp(k-K_number,dmp,temp1,temp2);
        k1=dt_test[k-K_number];
        make_dmp(k-K_number,dmp+k1*dt/2,temp1,temp2);
        k2=dt_test[k-K_number];
        make_dmp(k-K_number,dmp+k2*dt/2,temp1,temp2);
        k3=dt_test[k-K_number];
        make_dmp(k-K_number,dmp+k3*dt,temp1,temp2);
        k4=dt_test[k-K_number];
        ddmp = (k1 + 2*k2 + 2*k3 + k4)*dt/6;
        dmp+=ddmp;
        set_colm(DMP,i)=dmp;
    }
   /// #pragma omp parallel for
    for (int D=0;D<DOF_number;D++){
        set_rowm(psi[k],D)=rowm(DMP,2*D);
    }

}

void CMA_ES(int sample_number){

    lamda_count=0;

for(int i=0;i<1;i++){


        if(a_best>sum(statecostkt[i])||a_best==0){

            a_best=sum(statecostkt[i]);
            //theta_best=theta+colm(eps[i],0);

        }
        printf("a_p=%f\n",a_p);
        printf("sum_state_cost[%d]=%f\n",epic,sum(statecostkt[i]));
       if(a_p>sum(statecostkt[i])){ lamda_count++;}
        //judge_CMA_ES=10000;
}
//sample_number=K_max;
//sample_number=K_max_rest;
//K_max_rest=1;
damp=1+(num/(2*sample_number));
//damp=1;
//damp=1;
p_succ_target=1/(5+(sqrt(sample_number)/2));
///p_succ_target=0.5;
//p_succ_target=100*sqrt(((a_p-a_best)/a_p)*((a_p-a_best)/a_p));
//p_succ_target=0.4;
cp=p_succ_target*sample_number/(2+p_succ_target*sample_number);
///cp=0.7;
double tempq,tempqq;
tempq=sample_number+2;
c_c=2/tempq;
tempqq=num*num+6;
c_cov=2/tempqq;
p_thresh=0.44;



double temp2,temp3;
temp2=lamda_count;
temp3=sample_number;
//p_succ=temp2/temp3;
printf("p_succ_sample\n");
cout<<lamda_count<<endl;
//printf("p_succ\n");
//cout<<p_succ<<endl;
double judge_rnd;
judge_rnd=rnd2.get_random_double();
/// myutation judge
double judge_A;
judge_A=judge_rnd-(myutation_rate*epic/(500-1)-myutation_rate/(500-1));
p_succ_ave=(1-cp)*p_succ_ave+cp*p_succ;
printf("p_succ_ave\n");
cout<<p_succ_ave<<endl;

/// myutation judge
//if(judge_A>(1-myutation_rate)){
//    // old_distribution=0.8*distribution_p;
//    myutation_epic++;
//
//    temp_dist=myutation_mulitiple*distribution_p;
/////    myutation_judge=10000;
//    a_p_myutation=a_p;
//
//}
//else if(epic%20==0)
//{
//epic_myutation_judge=10000;
//temp_dist=myutation_mulitiple*distribution_p;
//
//}







//printf("p_succ_ave\n");
//cout<<p_succ_ave<<endl;
printf("distribution_p\n");
cout<<distribution_p<<endl;
if(p_succ>0){

CMA_PI_JUDGE=10000;
printf("succ_ordinallllllllllll\n");
if(p_succ_ave<p_thresh){

    if(myutation_occur==10000){
         p_c=(1-c_c)*p_c+sqrt(c_c*(2-c_c))*((theta-CMA_old_theta)/temp_dist);
    }else{
     p_c=(1-c_c)*p_c+sqrt(c_c*(2-c_c))*((theta-CMA_old_theta)/distribution_p);
    }

    sigma=(1-c_cov)*sigma+c_cov*p_c*trans(p_c);
}
else{

p_c=(1-c_c)*p_c;
sigma=(1-c_cov)*sigma+c_cov*(p_c*trans(p_c)+c_c*(2-c_c)*sigma);
//distribution_p=2*distribution_p;
}
}
///add
//else if(p_succ_ave<p_succ_target){
//p_c=(1-c_c)*p_c;
//sigma=(1-c_cov)*sigma+c_cov*(p_c*trans(p_c)+c_c*(2-c_c)*sigma);
//}
/// end
   distribution_p=distribution_p*exp((p_succ_ave-p_succ_target)/(damp*(1-p_succ_target)));

if(myutation_occur==10000&&myutation_judge==10000){
    myutation_judge=0;
    myutation_occur=0;
}
if(myutation_occur==10000&&epic_myutation_judge==10000){
    epic_myutation_judge=0;
    myutation_occur=0;
}
a_best=0;
K_max_rest=1;
}


void destroyMonoBot(int p_n)
{//ボディの破壊

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
void control(int p_n_2,int step_p)
{//printf("temp_p[3][0]=%f\n",temp_p[3][0]);
//printf("original_d=%d\n",d);//printf("Q=%f\n",1/Q1);
dReal  fMax = 1.0,KD=0.01;
double u_max,u_min;
u_max=6.28,u_min=-6.28;
double JA;
if(restart_judge==0){
for(int i=0;i<leg_max;i++){
    for(int j=0;j<3;j++){

//printf("ss");
//if(psi[K_number+p_n_2](3*i+j,step_p)>u_max){
//    dJointSetHingeParam(Joint_Motor[p_n_2][i][j],  dParamVel, u_max);
//}else if(psi[K_number+p_n_2](3*i+j,step_p)<u_min){
//dJointSetHingeParam(Joint_Motor[p_n_2][i][j],  dParamVel, u_min);
//}else{
//    dJointSetHingeParam(Joint_Motor[p_n_2][i][j],  dParamVel, psi[K_number+p_n_2](3*i+j,step_p));
//}
//printf("K_number=%d\n",K_number+p_n_2);
if(step_p%1==0){
angle[K_number+p_n_2][i][j]=dJointGetHingeAngle(Joint_Motor[p_n_2][i][j]);
JA=-(angle[K_number+p_n_2][i][j]-psi[K_number+p_n_2](3*i+j,step_p+1-(step_p%1)))/(step_size);
if(JA>10*u_max){
    JA=10*u_max;
}else if(JA<10*u_min){
JA=10*u_min;
}

real_vel[K_number+p_n_2](3*i+j,step_p)=JA;
    dJointSetHingeParam(Joint_Motor[p_n_2][i][j],  dParamVel, JA);
}else{
     JA=-(angle[K_number+p_n_2][i][j]-psi[K_number+p_n_2](3*i+j,step_p+1-(step_p%1)))/(step_size);
     if(JA>10*u_max){
    JA=10*u_max;
}else if(JA<10*u_min){
JA=10*u_min;
}
     real_vel[K_number+p_n_2](3*i+j,step_p)=JA;
 dJointSetHingeParam(Joint_Motor[p_n_2][i][j],  dParamVel, JA);
}

dJointSetHingeParam(Joint_Motor[p_n_2][i][j], dParamFMax, fMax);
      }
    }
}else{
    double angle_r[leg_max][3];
for(int i=0;i<leg_max;i++){
    for(int j=0;j<3;j++){

angle_r[i][j]=dJointGetHingeAngle(Joint_Motor[p_n_2][i][j]);
dJointSetHingeParam(Joint_Motor[p_n_2][i][j],  dParamVel, -angle_r[i][j]/step_size);

dJointSetHingeParam(Joint_Motor[p_n_2][i][j], dParamFMax, fMax);
//old_temp_u[i][j]=ut1[i][j];
//old_temp_p[i][j]=temp_pt[i][j];
//old_temp_v[i][j]=temp_vt[i][j];
      }
    }
}
    ///
}

void PI2(){

        int count_p=0;
        double temp1,temp2;
        a_p_best=100000000000000000;
        for(int i=0;i<K_max;i++){
            if(sum(statecostkt[i])<a_p){
                count_p++;
            }
           // cost_judge+=sum(statecostkt[i]);
            if(sum(statecostkt[i])<a_p_best){
                a_p_best=sum(statecostkt[i]);
                distance_max=distance_num[i];
                theta_CMA_best=theta+colm(eps[i],0);
            }

        }
       // cost_judge_ave=cost_judge/K_max;
        if(old_K_max_o!=0){
        old_old_k_max_o=old_K_max_o;
       }
       if(K_max_o!=0){
        old_K_max_o=K_max_o;
       }
       if(max_k_max<K_max){
            max_update=10000;
        max_k_max=K_max;
        max_dist=distribution_p;
       }
       K_max_o=K_max;
        temp1=count_p;
        temp2=K_max;
        p_succ=temp1/temp2;
        p_succ_dist=p_succ;
        printf("a_p\n");
        cout<<a_p<<endl;

        printf("p_succ_sinnnnnnn\n");
        cout<<count_p<<endl;
//printf("s20");


        if(off_noise==0){
//更新回数のカウント
update++;
//各ロールアウトにおける即時コスト
//#pragma omp parallel for
//for(int k=0;k<K_max;k++){
//for(int j=0;j<terminal_step;j++){
//
//    set_colm(controlcostkt[k],j)=(trans(theta+M[j]*colm(eps[k],j)))*(S*(theta+M[j]*colm(eps[k],j)))/2;
//}
//immediate_cost[k]=controlcostkt[k]+statecostkt[k];
//}

///ver_not_DMP_parameter==========

#pragma omp parallel for
for(int k=0;k<K_max;k++){
immediate_cost[k]=statecostkt[k];
}

///===============================

//各時刻における最適なthetaの変位の生成
//omploop
matrix <double> temp_eps[terminal_step];
#pragma omp parallel for
for(int i=0;i<terminal_step;i++){
        matrix <double> compensation;
matrix <double>  numeratorP;
matrix <double> P;
matrix <double> S1(K_max,1);
 matrix <double> epsik(num,K_max);
    for(int k=0;k<K_max;k++){
        parimmediatecostk[k] = immediate_cost[k];
        parterminalcostk[k]  = terminalcostkt[k];
    }
    // 各ロールアウトの将来コストと確率の計算
    S1=zeros_matrix<double> (K_max,1);
    for(int k=0;k<K_max;k++){
        S1(k,0)= sum(colm(immediate_cost[k],range(i,terminal_step-1)));
    }
    h=0.01;
    compensation=-10*(S1-min(S1)*ones_matrix<double>(K_max,1))/(max(S1)-min(S1));
    numeratorP = exp(compensation);
    P = numeratorP/sum(numeratorP);

    //各時刻におけるthetaの変位の期待値の計算

     for(int k=0;k<K_max;k++){
        set_colm(epsik,k)=colm(eps[k],i);

     }
     set_colm(dthetai,i)=(epsik*P);
}

check=norm((isnan(dthetai)));
//printf("check8");
if(check==1){
    errort++;
    ///ここのdthetaiがnan判定となりthetaの更新がされていない
}
else{
        //各次元における最適なthetaの変位の生成

    ///use_DMP==========================

//    #pragma omp parallel for
//    for(int j=0;j<num;j++){
////printf("check9");
//double temp1;
//    ///ここで問題発生
//    temp1=WN2*trans(rowm(ws1,j));
//    set_rowm(dtheta,j)=(rowm(ws1,j)*WN1*trans(rowm(dthetai,j)))/temp1;
//    }

    ///=================================

    ///ver_not_DMP_parameter===============

   #pragma omp parallel for
    for(int j=0;j<num;j++){
//printf("check9");
double temp1;
    ///ここで問題発生
    temp1=WN2*trans(rowm(ws1,j));
    set_rowm(dtheta,j)=(ones_matrix<double>(1,terminal_step)*WN1*trans(rowm(dthetai,j)))/temp1;

    }

///====================================
}

theta+=dtheta;
double average_theta;
average_theta=sum(theta)/20;
        }
}

void parallel_exe(){


#pragma omp parallel for
for(int i=0;i<thread_number;i++){

eps[K_number+i]         = zeros_matrix<double>(num,terminal_step);
meanzero[K_number+i]=zeros_matrix<double>(num,1);
mvnrnd(i,meanzero[K_number+i],distribution_p*distribution_p*sigma);
//mvnrnd(meanzero[K_number+i],distribution_p*distribution_p*sigma);
eps[K_number+i]=t_epst[i];
excute_policy((K_number+i),theta,eps[K_number+i]);
}


}

static void simLoop(int pause,int t_num,int STEP_p){
    double angle_r[leg_max][3];

    if(STEP_p<100){
        for(int i=0;i<leg_max;i++){
            for(int j=0;j<3;j++){

                angle_r[i][j]=dJointGetHingeAngle(Joint_Motor[t_num][i][j]);
                dJointSetHingeParam(Joint_Motor[t_num][i][j],  dParamVel, -angle_r[i][j]/step_size);
                dJointSetHingeParam(Joint_Motor[t_num][i][j], dParamFMax, 2);

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
            Rotation_base[i]=dBodyGetRotation(base[t_num][i].body);
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
        STEP_p=STEP_p-100;
        restart_judge=0;
        const dReal *vel_base;
        const dReal *pos_base;
        pos_base=dBodyGetPosition(base[t_num][0].body);
        vel_base=dBodyGetLinearVel(base[t_num][0].body);
        double vx,vy;
        double pos_x,pos_y,pos_z;
        double temp;
        pos_x=pos_base[0];
        pos_y=pos_base[1];
        pos_z=pos_base[2];
        vx=vel_base[0];
        vy=vel_base[1];
        ///temp=sqrt(vx*vx+vy*vy);
        ///temp=vy;
        double roll;
        roll=atan2(R1[3][1],R1[3][2]);
        roll=180*roll/M_PI;
        //if(sqrt(roll*roll)<30){
        //    temp=vy-0.5*sqrt(vx*vx)-30/(1+sqrt(roll*roll));
        //}else{
        if(vector_c[1]>0){
            temp=10*vy-fabs(pos_x)-0*(vector_c[1]+1);
        }
        else{
            temp=10*vy-fabs(pos_x)-0*(vector_c[1]+1);

        }

        old_posi=pos_x;
        double read,sum_read;
        sum_read=0;
        double qq;


        if(off_noise==1&&myutation_judge==0){
            real_judge=10000;
            for(int i=0;i<leg_max;i++){
                for(int j=0;j<3;j++){
                    double read1,read2;
                    read1=dJointGetHingeAngleRate(Joint_Motor[0][i][j]);
                    read2=dJointGetHingeAngle(Joint_Motor[0][i][j]);
                    real_angrate(STEP_p,3*i+j)=read1;
                    real_ang(STEP_p,3*i+j)=read2;


                }
            }
        }

        control(t_num,STEP_p);
        if(STEP_p!=(terminal_step-1)){

            double u_limit=0;
            double toruk=1;
            double u_limit_max=43-(43*toruk/3.8);
            u_limit_max=u_limit_max*(2*M_PI/60);
            double to_max=3.8;
            for(int i=0;i<leg_max;i++){
                for(int j=0;j<3;j++){

                    read=real_vel[K_number+t_num](3*i+j,STEP_p);

                    if(sqrt(read*read)>u_limit_max){
                        sum_read+=10000*(sqrt(read*read)-u_limit_max);
                    }
                }
            }

        }
        else{
            distance_num[K_number+t_num]=pos_y;
        }

        qq=sum_read;
        if(off_noise==1&&myutation_judge==0){

        }
///ロバスト化関連コスト////////////////////////////////////////////////////////////////////////////////////////////////


        //ベース位置の取得
        state_vector(0,0)=pos_x;
        state_vector(1,0)=pos_y;
        state_vector(2,0)=pos_z;
        //ベース姿勢の取得

        state_vector(3,0)=Rotation_base[0][0];
        state_vector(4,0)=Rotation_base[0][4];
        state_vector(5,0)=Rotation_base[0][8];

        //モータ角度の取得
        state_vector(6,0)=dJointGetHingeAngle(Joint_Motor[t_num][0][0]);
        state_vector(7,0)=dJointGetHingeAngle(Joint_Motor[t_num][0][1]);
        state_vector(8,0)=dJointGetHingeAngle(Joint_Motor[t_num][0][2]);

        double penalty=0.0;

        for(int i=0;i<3;i++){
            penalty+=posPena*(state_vector(i,0)-ref_state(i,STEP_p))*(state_vector(i,0)-ref_state(i,STEP_p));
            penalty+=rotPena*(state_vector(i+3,0)-ref_state(i+3,STEP_p))*(state_vector(i+3,0)-ref_state(i+3,STEP_p));
            penalty+=angPena*(state_vector(i+6,0)-ref_state(i+6,STEP_p))*(state_vector(i+6,0)-ref_state(i+6,STEP_p));
        }

        penalty=penalty*(terminal_step-STEP_p);//=penalty*(terminal_step-STEP_p);

        set_colm(statecostkt[K_number+t_num],STEP_p)=-(temp-sum_read)+penalty;

///end//////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
    dSpaceCollide(space[t_num],0,&nearCallback);
    dWorldStep(world[t_num],step_size);
    dJointGroupEmpty(contactgroup[t_num]);
}

static void start()
{
  static float xyz[3] = {   0.8, 0.0, 0.5};
  static float hpr[3] = {-180.0, -30.0, 0.0};
  dsSetViewpoint(xyz,hpr);               // 視点，視線の設定
  dsSetSphereQuality(3);                 // 球の品質設定
}


void make(int j){
    leg_number=0;
                 createMonoBot(j);
  for (int i=0;i<leg_max;i++){

  createleg(j);

  }
}

void initialized_parameter_epic(){
K_count=0;
eroor_count=0;
epic=0;
distance_maximam=0;
a_p=0;
episode=1;
jal=0;
off_noise=1;
p_c=zeros_matrix<double>(num,1);
p_succ_ave=0;
p_succ=0;
max_k_max=0;
max_dist=0;
time(&timer1);
sigma  =identity_matrix<double>(num);
distribution_p=initial_dist;
theta=zeros_matrix<double>(num,1);
theta_CMA_best=zeros_matrix<double>(num,1);
a_p_best=0;
best_cost2=0;
old_theta=zeros_matrix<double>(num,1);
m_count=0;
sample_total=0;
myu_step=sqrt(initial_myu_step);
myu_step_d=0.5;
}

void create_myutation(){
    printf("myutation\n");
    cout<<(sqrt(a_p_best*a_p_best)/(sqrt(a_p*a_p)+0.0001))<<endl;
    printf("K_max\n");
    cout<<K_max_o<<endl;
    double test;
    test=(sqrt(a_p_best*a_p_best)/(sqrt(a_p*a_p)+0.0001));
   if(test>2){

                                myu_step_d=(exp(-(5*m_count)/sample_total)+exp(-(sqrt((a_p-a_p_best)*(a_p-a_p_best))/sqrt(a_p*a_p))));
                                distribution_p*=myu_step_d;
                                if(jal==10000){
                                  myu_step*=myu_step_d;
                                }

                                 printf("myutoooooooooooooooooo\n");
                            m_count=0;
                             sample_total=0;

                        }else{
                        if(K_max_o<4){
                            jal=10000;
                            max_update=0;
                              myu_step=myu_step*(exp(-(2*m_count)/sample_total)+exp(-(sqrt((a_p-a_p_best)*(a_p-a_p_best))/sqrt(a_p*a_p))));
                            distribution_p+=(max_dist*myu_step);
                             sigma+=identity_matrix<double>(num);

                             m_count=0;
                             sample_total=0;


                        }
                        }


}


void create_total_cost_0(){
  for(int i=0;i<terminal_step;i++){
            set_colm(controlcostkt[0],i)=trans((theta))*S*(theta);
            }
        immediate_cost[0]=statecostkt[0]+controlcostkt[0];
        sum_controlcost[0]   = terminal_step*(trans((theta))*S*(theta));


        ///fprintf(outputfile5[MM],"%f\n",sum_statecost[0]);
        cost[0] =  sum_statecost[0] + sum_controlcost[0];
}

void update_input(){
        const dReal *posii;
        double terminal_x_u,terminal_y_u;
        double temp222;
        posii=dBodyGetPosition(base[0][0].body);
        terminal_x_u=posii[0];
        terminal_y_u=posii[1];
        temp222=terminal_y_u;
        CMA_old_theta=old_theta;
  if(a_p>sum(statecostkt[0])||episode==1){
                if(sum(statecostkt[0])>a_p_best&&episode!=1){
                    a_p=a_p_best;
                    theta=theta_CMA_best;
                    distance_maximam=distance_max;
                    old_theta=theta;
                    printf("DMA_DMA_DMA_lll\n");

                }else{
        distance_maximam=temp222;
        a_p=sum(statecostkt[0]);
        old_theta=theta;
                }
        best_theta_count=(epic+1);
        }else{
            if(a_p>a_p_best){
                    printf("CMA_CMA_CMA_lllllll\n");
                 a_p=a_p_best;
                 distance_maximam=distance_max;
                    theta=theta_CMA_best;
                    old_theta=theta;

            }else{
            theta=old_theta;
            }
        }

        printf("epic=%d,a_p=%f,sum_statecost[0]=%f\n",epic,a_p,sum(statecostkt[0]));
            off_noise=0;
}

void decide_sampling_number(){

///カーネル回帰への入力
for(int i=0;i<thread_number;i++){
    sum_statecost[K_number+i]=sum(statecostkt[K_number+i]);
}
///生成した軌道について特徴量判断
for(int i=0;i<thread_number;i++){
    judge_number=0;

    if(ALD_judge==0){
        ALD_algorithm(i);
    }
     ///異なる特徴が出た場合judge_number=10000でなかったらjudge_number=0
    if(judge_number==0){

        ///生成軌道数確定
        if(eroor_count>0){
            eroor_count=0;
        }else{
            K_max=K_number+i+1;
        }

        ALD_judge=10000;
        d=1;
        ddeltat=0;
        temp_K=zeros_matrix<double>(division,K_limit);
        K_nolm_m=zeros_matrix<double>(1,K_limit);
        printf("k_max=%d\n",K_max);

        ///PI2でu+duを求める
        PI2();

        K_max_rest=K_max;
         sample_total+=K_max;
        m_count++;
        copy_m(epic+1,3)=K_max;
        K_max=1;
        K_number=0;
        off_noise=1;
        break;
            }
        }

        if(off_noise==0&&ALD_judge==0){
            ///生成した軌道数ではサンプル数が確定しなかった場合
            K_number=K_number+thread_number;

        }
        else{

            ALD_judge=0;
            temp_K=zeros_matrix<double>(division,K_limit);
            K_nolm_m=zeros_matrix<double>(1,K_limit);
            K_max=1;
            K_number=0;
        }
        judge_number=0;
}

void proposed(int MM){
    printf("MM=%d\n",MM);
    while(epic<=100){
        if(MM>=(max_trial_num)){
            dCloseODE();
        }

        if(off_noise==1){
            ///モデル生成
            make(0);
            ///モータへの摂動を除去した入力生成
            excute_policy(0,theta,zeros_matrix<double>(num,terminal_step));
            ///摂動を除去した入力の付加
            for(int j=0;j<terminal_step+100;j++){
                simLoop(0,0,j);
            }


            ///入力更新
            update_input();
             if(episode==1){
            }
            else{
                ///CMAES
                CMA_ES(K_max_rest);
                ///突然変異判断
                create_myutation();
            }
            if(episode==1){
                episode=2;
            }
            else{
                epic++;
            }
            if(epic==10){
                for(int i=0;i<num;i++){
                    double temp_tra_o;
                    temp_tra_o=theta(i,0);
                    if(i!=num-1){
                        fprintf(output_goal,"%.32f,",temp_tra_o);
                    }
                    else{
                        fprintf(output_goal,"%.32f;\n",temp_tra_o);
                    }
                }
            }
            if(epic==50){
                for(int i=0;i<num;i++){
                    double temp_tra_o;
                    temp_tra_o=theta(i,0);
                    if(i!=num-1){
                        fprintf(output_goal,"%.32f,",temp_tra_o);}
                    else{
                        fprintf(output_goal,"%.32f;\n",temp_tra_o);
                    }
                }
            }
            if(epic==100){
                for(int i=0;i<num;i++){
                    double temp_tra_o;
                    temp_tra_o=theta(i,0);
                    if(i!=num-1){
                        fprintf(output_goal,"%.32f,",temp_tra_o);
                    }
                    else{
                        fprintf(output_goal,"%.32f;\n",temp_tra_o);
                    }
                }
                fprintf(output_goal,"\n\n\n");
            }
            time(&timer2);
            double time_q;
            time_q=timer2-timer1;
            copy_m(epic,0)=time_q/60;
            copy_m(epic,1)=distance_maximam;
            copy_m(epic,2)=a_p;
            ///動力学モデルの初期化
            destroyMonoBot(0);
            dJointGroupDestroy(contactgroup[0]);     // ジョイントグループの破壊
            contactgroup[0] = dJointGroupCreate(0);  // ジョイントグループの生成
        }
        else{
            ///論理プロセッサ分の入力とモデルの作成
            parallel_exe();
            for(int j=0;j<thread_number;j++){
                make(j);
            }
            ///論理プロセッサ分の軌道を生成
            #pragma omp parallel for
            for(int i=0;i<thread_number;i++){
                for(int j=0;j<terminal_step+100;j++){
                    simLoop(0,i,j);
                }
            }
            ///生成軌道数の判断
            decide_sampling_number();

            ///動力学モデルの初期化

            for(int i=0;i<thread_number;i++){
                destroyMonoBot(i);
                dJointGroupDestroy(contactgroup[i]);     // ジョイントグループの破壊
                contactgroup[i] = dJointGroupCreate(0);  // ジョイントグループの生成
            }
        }
    }

}

int main (int argc, char *argv[])
{
    thread_number=omp_get_max_threads();
   // thread_number=1;
    distribution_p=initial_dist;
    time(&timer1);
    umax=10;
 ///寸法パラメータ/////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    ///重み行列の設定
    Wp=zeros_matrix<double>(state_number,state_number);

    for(int i=0;i<3;i++){
        Wp(i,i)=posPena;
        Wp(i+3,i+3)=rotPena;
        Wp(i+6,i+6)=angPena;
    }
///end///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    dInitODE();




    if(reflection_ON==1){
        outputfile17[0]=fopen("policy_modified.txt","w");
    }
    else{
        outputfile17[0]=fopen("policy_original.txt","w");
    }
    outputfile19[0]=fopen("excel_1.txt","w");
    output_goal=fopen("goal.txt","w");
    inputfile1=fopen("ref_state.txt","r");
    if(inputfile1=NULL){
        printf("file open error");
    }

    for(int i=0;i<palallel_p;i++){
        world[i]=dWorldCreate();
        space[i]=dHashSpaceCreate(0);
        contactgroup[i]=dJointGroupCreate(0);
        dWorldSetGravity(world[i],0,0,-grav);
        ground[i]=dCreatePlane(space[i],0,0,1,0);
        dWorldSetERP(world[i], 0.2);          // ERPの設定
        dWorldSetCFM(world[i], 1e-4);         // CFMの設定
    }


    E=identity_matrix<double>(num);
    S=W_R*E;
    sigma  =1* W_sigma *E;


    theta=zeros_matrix<double> (num,1);
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
    //DMPパラメータ
    //phi_T = tf/6;
    alphaq = 25;


    //DMP線形基底関数の設定
    //f=1;
    r0 = 25*25/4; // 初期振幅
    rg = 25*25/4; // 目標振幅
    //phi0 = 0; // 初期位相
    printf("E2");
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
    theta=zeros_matrix<double>(num,1);
    //多変量正規分布の作成 clear
    meanzerot=zeros_matrix<double>(num,1);
    printf("E5");
    restart_judge=0;
///参照状態量の取得////////////////////////////////////////////////////////////////////////////////////////////////////
    double data;
    for(int i=0;i<state_number;i++){
        for(int j=0;j<terminal_step;j++){
            fscanf(inputfile1,"%lf,",&data);
            ref_state(i,j)=data;
        }
    }
    fclose(inputfile1);




    int MM=0;
    a_p=0;
    off_noise=1;




    for(int MM=0;MM<max_trial_num;MM++){

        proposed(MM);
        set_colm(excel,range((MM)*4,(MM+1)*4-1))=copy_m;
        for(int i=0;i<num;i++){
            double temp_tra_o;
            temp_tra_o=theta(i,0);

            fprintf(outputfile17[0],"%.32f,",temp_tra_o);

        }
        fprintf(outputfile17[0],"\n\n");

        initialized_parameter_epic();
    }
    fclose(outputfile17[0]);
    printf("20 times finish");

    for(int t=0;t<terminal_step;t++){
            for(int j=0;j<max_trial_num*4;j++){
                double tempk11;
                tempk11=excel(t,j);
                fprintf(outputfile19[0],"%.2f,",tempk11);
            }
    fprintf(outputfile19[0],"\n");
    }


    dCloseODE();

    return 0;
}
