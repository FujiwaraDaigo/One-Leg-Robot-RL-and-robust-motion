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
#define palallel 4
#define K_limit 10
#define terminal_step 1200
#define iteration_number 100
#define step_size 0.01
///ロバスト化関連/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define max_trial_num 20
///end///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
dWorldID world[palallel];  // 動力学計算用ワールド
dSpaceID space[palallel];  // 衝突検出用スペース
dGeomID  ground[palallel]; // 地面
dJointGroupID contactgroup[palallel]; // コンタクトグループ
dReal r = 0.2, m  = 1.0;
dsFunctions fn;
dlib::rand  rnd;
typedef struct {       // MyObject構造体
  dBodyID body;        // ボディ(剛体)のID番号（動力学計算用）
  dGeomID geom;        // ジオメトリのID番号(衝突検出計算用）
  double  l,r,m;       // 長さ[m], 半径[m]，質量[kg]
} MyObject;

static MyObject base[palallel][2],Motor[palallel][leg_max][3],Pla[palallel][leg_max][3],Al[palallel][leg_max][2],pole[palallel],weight[palallel],denti[palallel],base_under[palallel];    // leg[0]:上脚, leg[1]:下脚
static dReal base_r,base_lz,Motor_lx,Motor_ly,Motor_lz,Pla_lx,Pla_ly,Pla_lz,Al_lx[2],Al_ly[2],Al_lz[2],base_m,Motor_m,Pla_m,Al_m[2],base_x[2],base_y[2],base_z[2],pole_r,pole_l,weight_r;
static dJointID Joint_Motor[palallel][leg_max][3],Joint_Pla[palallel][leg_max][3],Joint_Al[palallel][leg_max],Joint_fix[palallel][leg_max],Joint_pole[palallel],Joint_weight[palallel],joint_denti[palallel],joint_base_under[palallel]; // ヒンジ, スライダー

static int STEPS = -1;            // シミュレーションのステップ数
static dReal S_LENGTH = 0.0;     // スライダー長
static dReal H_ANGLE  = 0.0;
FILE *outputfile1,*outputfile2,*outputfile3,*outputfile4,*outputfile5,*outputfile6,*outputfile7,*outputfile8,*outputfile9,*outputfile10,*outputfile11,*outputfile12,*outputfile13,*outputfile14,*outputfile15,*outputfile16,*outputfile17;
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
static dReal Al_x[2];
static dReal Al_y[2];
static dReal Al_z[2];
static dReal pole_x,pole_y,pole_z;
static dReal Joint_M_x[3];
static dReal Joint_M_y[3];
double Al_l,Al_r;
static dReal Joint_M_z[3];
dReal PI=atan(1)*4;
dReal theta2;
int H;
int leg_number=0;
int b;
//PI2
//PI2パラメータ
int K2=10;
dReal ddt=step_size;
int update_max=30000;
dReal cost_limit=0;
//lambda/Sの制約
dReal W_R=0.001;
dReal W_sigma=0.1;
matrix <double> K_nolm_m(1,100);
double K_nolm;
matrix <double> E(num,num);
matrix <double> S(num,num);
matrix <double> sigma(num,num);
dReal lamda;
matrix <double,num_policy,terminal_step> epst[100];
matrix <double,num,1> meanzerot;
//matrix <double> WN1(10,10);
//matrix <double> WN2(10,10);
matrix<double> WN2(1,terminal_step);
double old_angle[leg_max][3];
dReal RR;
int initial_set=0;
int K_number=0;
int JH=0;
//dmatrix siguma=W_sigma*E;
dReal lambda=W_R*W_sigma;
int update=0;
int eroor=0;
//dmatrix epst[10]=zeros_matrix<double>(10,10);
//dmatrix trajectorykt[10]=zero_matrix<double>()
//dmatrix meanzerot=zero_matrix<double>(1,10);
//FILE *outputfile1,*outputfile2;
time_t timer1,timer2;
matrix <double,1,terminal_step> statecostkt[100];
matrix <double,1,terminal_step> controlcostkt[100];
matrix <double,1,1> terminalcostkt[100];
matrix <double,DOF_number*2+num_base*3,terminal_step> trajectory[100];
matrix <double> theta(num,1);
matrix <double> state_x(DOF_number*2,1);
matrix<double,1,terminal_step> immediate_cost[100];

matrix <double> dthetait(num,terminal_step);
matrix <double> first_theta;
int off_noise=0;
double sum_controlcost[100],sum_statecost[100];
double best_cost,best_taskcount,best_update;
matrix <double> best_theta;
matrix <double> best_trajectory;
matrix <double,num,terminal_step> eps[100];
matrix <double> epsi;
double cost[100];
double taskcount;
matrix <double> meanzero[100];
matrix <double> dthetai(num,terminal_step);
matrix <double> parimmediatecostk[100];
matrix <double> parterminalcostk[100];

int *pause_p;



matrix <double> thetat;
matrix <double,num,num> exe_theta[100];
matrix <double> dtheta(num,1);
int h;

matrix <double,num,1> normal_vector[terminal_step];
//要変更
matrix <double,num,num> M[terminal_step];
matrix <double> Mi;
matrix <double> Mi1;
//
double check;
int errort;
double cost1,best_cost1;
matrix <double> best_trajectory1;
matrix <double> WN1(terminal_step,terminal_step);
matrix <double> dthetat(num,1);

matrix <double> t_epst(num,terminal_step);

matrix <double> One_temp;


matrix <double,num,num> temp_exetheta[100];
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
matrix <double,DOF_number,terminal_step> psi[100];
matrix <double> rr0(DOF_number*2,1);
matrix <double> rrg(DOF_number*2,1);
matrix <double> tt(1,terminal_step);
matrix <double> dt_test;
matrix <double> kernel_output;
matrix <double> k_st;
matrix <double> temp_K(division,100);
matrix <double> Gausian_kernel;
matrix <double,num,1> theta_k[100];
matrix <double,num,1> mean_f;
matrix <double> covariance_K;
matrix <double,1,terminal_step> sum_cost;
///ロバスト化関連//////////////////////////////////////////////////////////////////////////////////////////////////////
matrix <double,max_trial_num,num> policy;
double Al_friction;
double Pla_friction;
double grav;
int tra=0;
int add_model_noise=0;
double noise_ratio=0.1;//0.1;
int stop_mode=0;
float zoom=0.0;
int key_s=0;
float dy=0;
///end//////////////////////////////////////////////////////////////////////////////////////////////////////////////

int memo_num=1;
///Q_Learning関係
double Q[1000][5];
double ttemp=0;
matrix <double,1,terminal_step> old_sum_cost;
int sum_abs_old=0;
int old_number;
int tt_p;
double Q_R=0;
int max_number=0;
    int sum_abs=0;
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
double old_posi;
//強化学習セット
dReal u[100000];//State
int a[11]={1,2,3,4,5,6,7,8,9,10,11};;//action
dReal R;//Reward

dReal discount_rate,learning_rate;
int new_state,old_state;
int old_action,new_action;
int STEP;
int best_action;
dReal umax;
int count_p=0;
double angle[1000][leg_max][3];
matrix <double,3*leg_max,terminal_step> real_vel[500];
double ang_1[400],ang_2[400],ang_3[400];
double base_under_m=1*0.001;

void mvnrnd(matrix <double> meanzerot1,matrix <double>E1){
 //多変量正規分布の作成 clear
double test;

//r.get_random_gaussian();
test=rnd.get_random_gaussian();
    for (int i=0;i<terminal_step;i++){
            for(int j=0;j<num;j++){

               normal_vector[i](j,0)= rnd.get_random_gaussian();

            }
    }
    typedef matrix <double> matrix_exp_type;

     matrix <double> FF[terminal_step];
    matrix <double> GG[terminal_step];
    matrix <double> temp_H;
    temp_H=E1;
        cholesky_decomposition<matrix_exp_type> cholesky_decomposition(E1);
    #pragma omp parallel for
    for(int i=0;i<terminal_step;i++){
            //matrix <double> temp_epst(num,terminal_step);
matrix <double,num,num> temp_epst;


        temp_epst=cholesky_decomposition.get_l();


        FF[i]=temp_epst*normal_vector[i]+meanzerot1;
       // GG[i]=meanzerot1;



        //theta=epst;
        //theta=ones_matrix<double>(DOF_number,terminal_step);
    }
    for(int i=0;i<terminal_step;i++){
        set_colm(t_epst,i)=FF[i];
    }
    //cout<<t_epst<<endl;
//    for(int i=0;i<terminal_step;i++){
//
//
//    }
//        for(int i=0;i<terminal_step;i++){
//
//        }

}

// 衝突検出計算
static void nearCallback(void *data, dGeomID o1, dGeomID o2)
{
  static const int N = 30;     // 接触点数
  dContact contact[N];

  int isGround = ((ground[0] == o1) || (ground[0] == o2));

  // 2つのボディがジョイントで結合されていたら衝突検出しない
  dBodyID b1 = dGeomGetBody(o1);
  dBodyID b2 = dGeomGetBody(o2);
  if (b1 && b2 && dAreConnectedExcluding(b1,b2,dJointTypeContact)) return;

  int n =  dCollide(o1,o2,N,&contact[0].geom,sizeof(dContact));
  if (n>0)  {
    for (int i = 0; i < n; i++) {
      contact[i].surface.mode   =  dContactSoftERP |
                                  dContactSoftCFM;
      contact[i].surface.soft_erp   = 0.2;   // 接触点のERP
      contact[i].surface.soft_cfm   = 0.001; // 接触点のCFM
      if(b1==Al[0][0][0].body||b2==Al[0][0][0].body){
         contact[i].surface.mu     = Al_friction; // 摩擦係数:無限大

      }else{
         contact[i].surface.mu     = Pla_friction; // 摩擦係数:無限大
      }

      dJointID c = dJointCreateContact(world[0],
                                       contactgroup[0],&contact[i]);
      dJointAttach (c,dGeomGetBody(contact[i].geom.g1),
                      dGeomGetBody(contact[i].geom.g2));
    }
  }
}


static void drawMonoBot()
{
  const dReal *pos1[leg_max][3],*R1[leg_max][3],*pos2[leg_max][3],*R2[leg_max][3],*pos3[leg_max][2],*R3[leg_max][2];
  const dReal *posr[2],*Rr[2],*pos_denti,*R_denti,*pos_under,*R_under;
dVector3 sides0,sides1,sides2,sides3,sides00,sides01,sides_denti;
///dGeomBoxGetLengths(Motor[0][0][0].geom,sides0);
dGeomBoxGetLengths(Motor[0][0][1].geom,sides00);
dGeomBoxGetLengths(Pla[0][0][0].geom,sides1);
dGeomBoxGetLengths(Pla[0][0][1].geom,sides01);
//dGeomBoxGetLengths(denti[0].geom,sides_denti);
//dGeomBoxGetLengths(Al[0][0][0].geom,sides2);
//dGeomBoxGetLengths(Al[0][0][1].geom,sides3);
  //ベースの描写
for(int i=0;i<1;i++){
posr[i]=dBodyGetPosition(base[0][i].body);
Rr[i]=dBodyGetRotation(base[0][i].body);
dsDrawCylinder(posr[i],Rr[i],base_lz,base_r);}

  // 脚部の描画
for (int i =0;i<leg_max;i++){
 for (int j=0;j<3;j++){
  pos1[i][j]=dBodyGetPosition(Motor[0][i][j].body);
  R1[i][j]=dBodyGetRotation(Motor[0][i][j].body);

  pos2[i][j]=dBodyGetPosition(Pla[0][i][j].body);
  R2[i][j]=dBodyGetRotation(Pla[0][i][j].body);

  if(j==0){
  dsSetColor(0,0,0);
   dsDrawBox(pos1[i][j],R1[i][j],sides00);

  }
  else{dsSetColor(0,0,0);
   dsDrawBox(pos1[i][j],R1[i][j],sides00);


  }
  if(j==1){
    dsSetColor(1,1,1);
  dsDrawBox(pos2[i][j],R2[i][j],sides01);
  }else{
   dsSetColor(1,1,1);
  dsDrawBox(pos2[i][j],R2[i][j],sides1);
  }


  }
//  for (int j=0;j<1;j++){
//    pos3[i][j]=dBodyGetPosition(Al[0][i][j].body);
//  R3[i][j]=dBodyGetRotation(Al[0][i][j].body);
//  if(j==0){
//  dsDrawBox(pos3[i][j],R3[i][j],sides2);}
//  else{dsDrawBox(pos3[i][j],R3[i][j],sides3);}
//  }
pos3[i][0]=dBodyGetPosition(Al[0][i][0].body);
  R3[i][0]=dBodyGetRotation(Al[0][i][0].body);
  dsDrawCapsule(pos3[i][0],R3[i][0],Al_l,Al_r);



}
//    dsSetColor(0,0,1);
//  pos_under=dBodyGetPosition(base_under[0].body);
//  R_under=dBodyGetRotation(base_under[0].body);
//  dsDrawCylinder(pos_under,R_under,1*0.001,base_r);
//  dsSetColor(1,1,1);
//pos_denti=dBodyGetPosition(denti[0].body);
//R_denti=dBodyGetRotation(denti[0].body);
//dsDrawBox(pos_denti,R_denti,sides_denti);

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
// //アルミ固定用
// Joint_Al[p_n][leg_number]=dJointCreateFixed(world[p_n],0);
// dJointAttach(Joint_Al[p_n][leg_number],Al[p_n][leg_number][0].body,Al[p_n][leg_number][1].body);
// dJointSetFixed(Joint_Al[p_n][leg_number]);
leg_number++;

}

void createMonoBot(int p_n) {
  dMass mass;
  dMatrix3 TA;
  dRFromAxisAndAngle(TA,0,1,0,M_PI/2);
    //double base_under_m=10;

for (int i=0;i<num_base;i++){
base[p_n][i].body=dBodyCreate(world[p_n]);
dMassSetCylinderTotal(&mass,base_m,2,base_r,base_lz);
dBodySetMass(base[p_n][i].body,&mass);
dBodySetPosition(base[p_n][i].body,base_x[i]-base_lz/2-Pla_lx,base_y[i],base_z[i]);
dBodySetRotation(base[p_n][i].body,TA);
base[p_n][i].geom=dCreateCylinder(space[p_n],base_r,base_lz);
dGeomSetBody(base[p_n][i].geom,base[p_n][i].body);


}
//denti[p_n].body=dBodyCreate(world[p_n]);
//dMassSetBoxTotal(&mass,261*0.001,0.0186,63*0.001,80*0.001);
//dBodySetMass(denti[p_n].body,&mass);
//dBodySetPosition(denti[p_n].body,-Pla_lx-36*0.001,0,base_z[0]);
//
//denti[p_n].geom=dCreateBox(space[p_n],0.0186,63*0.001,80*0.001);
//dGeomSetBody(denti[p_n].geom,denti[p_n].body);
//joint_denti[p_n]=dJointCreateFixed(world[p_n],0);
//dJointAttach(joint_denti[p_n],base[p_n][0].body,denti[p_n].body);
//dJointSetFixed(joint_denti[p_n]);



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
  //  printf("x[%d]\n",i1);
    //cout<<x[i1]<<endl;

}

//printf("ss2");
//printf("xt\n");
//cout<<xt<<endl;
//printf("d=%d\n",d);

//cout<<-hyper_p*(trans(x[0]-xt))*(x[0]-xt)<<endl;

for(int i1=0;i1<d;i1++){
        set_rowm(kstt,i1)=exp(-hyper_pt*((trans(x[i1]-xt))*(x[i1]-xt)));
        //printf("trans(x[i1]-xt)\n");
//cout<<(x[i1]-xt)<<endl;
}
//printf("k_st");
//cout<<kstt<<endl;


k_st=kstt;
//printf("k_st\n");
//cout<<kstt<<endl;

//printf("ss3");
for(int i1=0;i1<d;i1++){
    for(int j=0;j<d;j++){
        kernel_output_temp(i1,j)=exp(-hyper_pt*(trans(x[j]-x[i1])*(x[j]-x[i1])));
    }
}
kernel_output=kernel_output_temp;
//printf("kernel_output\n");
//cout<<-hyper_pt*(trans(x[j]-x[i1])*(x[j]-x[i1]))<<endl;

//printf("ss4");
        kt=exp(-hyper_pt*(trans(xt)*(xt)));
        kt=1;

}
void ALD_algorithm(matrix <double> trajectoryk){
//matrix <double> xt(DOF_number*2,1);


//printf("trajectory\n");
//cout<<trajectoryk<<endl;
matrix <double> tra_k(division+1,1);
matrix <double> t_k(1,terminal_step);
matrix <double> ttt(1,terminal_step);
matrix <double> cost_division(division,1);
matrix <double> temp1(1,terminal_step);
matrix <double> temp2(1,terminal_step);
for(int i=0;i<terminal_step;i++){
    ttt(0,i)=i*step_size;
}
//printf("s0");
//printf("L0");
//sum_cost(0,0)=statecostkt[K_number](0,0);
//for(int j=1;j<terminal_step;j++){
//    sum_cost(0,j)=sum_cost(0,j-1)+statecostkt[K_number](0,j);
//}
 set_rowm(cost_division,0)=sum_cost(0,0);
for(int i=0;i<division;i++){
  set_rowm(cost_division,i)=sum_cost(0,(terminal_step/division-1)+i*terminal_step/division);
}
K_nolm=sqrt(trans(cost_division)*cost_division);
//printf("s1");
 set_rowm(tra_k,range(0,division-1))=normalize(cost_division);
 //set_rowm(tra_k,division)=K_nolm;
//printf("s2");
//printf("tra_k\n");
//cout<<normalize(tra_k)<<endl;

//cout<<tra_k<<endl;

double myu=0.001;
//printf("L1");
if(initial_set==0){
    judge_number=10000;
    //printf("L1.5");
    set_colm(temp_K,0)=rowm(tra_k,range(0,division-1));
    set_colm(K_nolm_m,0)=K_nolm;
    //printf("L1.6");
    //printf("temp_K\n");
    //cout<<temp_K<<endl;
    initial_set=10000;
}
else{
//printf("s3");
matrix <double> kst;
matrix <double> ct;
matrix <double> test(division+1,d+1);
matrix <double> ktt(1,1);
double deltat;
matrix <double> max_matrix;
matrix <double> xt(division+1,1);
//if(i==0){
//    set_colm(temp_K,0)=colm(trajectoryk,0);
//}
//printf("s4");
//printf("L2");
///xt,vtのセット
//printf("L2.1");
//cout<<colm(trajectoryk,i)<<endl;
set_colm(temp1,range(0,d-1))=colm(K_nolm_m,range(0,d-1));
//printf("s4.5");
set_colm(temp1,d)=K_nolm;
//printf("s4.6");
temp2=normalize(temp1);
//printf("s4.7");
//xt=normalize(tra_k);
set_rowm(xt,range(0,division-1))=rowm(tra_k,range(0,division-1));
//printf("s4.8");
set_rowm(xt,division)=colm(temp2,d);
//printf("s5");
//printf("L3");
//printf("i1=%d\n",i1);
//printf("L2.5");
//test=colm(temp_K,range(0,d-1));
matrix <double> temp_K_temp(division+1,100);
set_rowm(temp_K_temp,range(0,division-1))=temp_K;
set_colm(test,range(0,d-1))=colm(temp_K_temp,range(0,d-1));
set_rowm(test,d)=colm(temp2,range(0,d-1));
max_matrix=abs(test-xt*ones_matrix<double>(1,d));
hyper_p=0.9;
//printf("s6");
Kernel(test,hyper_p,xt,d);
//printf("L5");
//printf("L3");
//printf("i3=%d\n",i1);
kst=k_st;
//printf("i4=%d\n",i1);
ktt=kt;
//printf("i5=%d\n",i1);
//K[1]=kernel_output;
//printf("i6=%d\n",i1);
//printf("s6.5");
ct=inv(kernel_output)*kst;
//printf("s7");

deltat=ktt(0,0)-trans(kst)*ct;
//printf("deltat");
//cout<<deltat<<endl;

if(deltat>0.05){
    //printf("L4.1");
    //printf("d=%d\n",d);
    //cout<<xt<<endl;
    //printf("s8");
    judge_number=10000;
    set_colm(temp_K,d)=rowm(xt,range(0,division-1));
    //printf("s9");
    set_colm(K_nolm_m,d)=K_nolm;
    // printf("temp_K\n");
        // cout<<colm(temp_K,range(0,d))<<endl;
          //printf("K_nolm\n");
         //cout<<colm(K_nolm_m,range(0,d));
    ddeltat=deltat;
     //printf("L4.2");
    d++;
}
else{
    ddeltat=abs(deltat-ddeltat);
    //printf("ddeltat=%f\n",ddeltat);
    if(ddeltat>0.025){
         judge_number=10000;
         //printf("s10");
         set_colm(temp_K,d)=rowm(xt,range(0,division-1));
         //printf("temp_K\n");
         //cout<<colm(temp_K,range(0,d))<<endl;
        //printf("s11");
        set_colm(K_nolm_m,d)=K_nolm;
         //printf("K_nolm\n");
         //cout<<colm(K_nolm_m,range(0,d));
          d++;
    }

    ddeltat=deltat;

}


}
//printf("ALD");
///

}

void arg_max(){

    int count_down=1;
    double noise_rate[5];
    noise_rate[0]=0.25;
    noise_rate[1]=0.5;
    noise_rate[2]=1;
    noise_rate[3]=2;
    noise_rate[4]=4;
    double s_c=0;
    double s_o_c=0;
    s_c=sum_cost(0,terminal_step-1);
    s_o_c=old_sum_cost(0,terminal_step-1);
//printf("sa");

///argmax
if(old_sum_cost(0,terminal_step-1)==0){


    }else{


sum_abs=(int)(((s_c-s_o_c)/old_sum_cost(0,terminal_step-1))*100);
Q_R=s_c-s_o_c;
//cout<<sum_cost<<endl;
//printf("double\n");
//cout<<(s_c-s_o_c)/s_o_c<<endl;
if(sum_abs>=1&&sum_abs<5){
    sum_abs=1;
}else if(sum_abs>=5&&sum_abs<10)
{
sum_abs=2;
}else if(sum_abs>=10&&sum_abs<15){
sum_abs=3;
}else if(sum_abs>=15&&sum_abs<20){
sum_abs=4;
}else if(sum_abs>=20){
sum_abs=5;
}else if(sum_abs<=-1&&sum_abs>-5){
sum_abs=6;
}else if(sum_abs<=-5&&sum_abs>-10){
sum_abs=7;
}else if(sum_abs<=-10&&sum_abs>-15){
sum_abs=8;
}else if(sum_abs<=-15&&sum_abs>-20){
sum_abs=9;
}else if(sum_abs<=-20){
sum_abs=10;
}else{
sum_abs=0;
}
//printf("sum_abs=%d\n",sum_abs);
ttemp=0;
int decision=0;

for(int i=0;i<5;i++){
    if(Q[sum_abs][i]>ttemp||decision==0){
        max_number=i;
        ttemp=Q[sum_abs][i];
        //printf("max_number2=%d\n",i);
        //printf("Q_max\n");
        //cout<<ttemp<<endl;
        decision=1;
    }else{
    count_down++;
    }
}
if(count_down==6){
    max_number=2;
}
 max_number=2;
sigma=100*W_sigma*E*noise_rate[max_number];

    }


    old_sum_cost=sum_cost;
    //printf("sa\n");
    //printf("max_number=%d\n",max_number);
}

void Q_learning(){
    double L_rate=0.9;
    double d_rate=0.9;
//    if(sum_cost(0,terminal_step-1)-old_sum_cost(0,terminal_step-1)>0){
//
//    }

if(old_sum_cost(0,terminal_step-1)==0){

}else{
Q[sum_abs_old][old_number]=Q[sum_abs_old][old_number]+L_rate*(-Q_R+d_rate*tt_p-Q[sum_abs_old][old_number]);
//printf("Q\n");
//cout<<Q[sum_abs_old][old_number]<<endl;
}
//old_sum_cost=sum_cost;

old_number=max_number;
sum_abs_old=sum_abs;

}

void make_rhythmic(double r0,double rg){
double dc ;
for(int i=0;i<terminal_step;i++){
    tt(0,i)=i*step_size;

}
//printf("s1");
dc=2 * pi / num_policy;
matrix <double> c(num_policy,1);

for(int i=0;i<num_policy;i++){
    set_rowm(c,i)=i*dc+dc-M_PI;
}
//printf("s2");
double h = 1 / (2 * (dc*dc) );
//cout<<phi0*ones_matrix<double>(1,terminal_step)+tt/tau<<endl;

phi=phi0*ones_matrix<double>(1,terminal_step)+tt/tau;

double alphag = alphaq/2;

///st調整 tt➡terminal stが目標値に
//printf("s3");

 st = (r0 - rg) * exp(-alphag*(tt)/tau) + r0 * ones_matrix<double>(1,terminal_step);
//printf("s4");
if(st(0,0)==0){
    st(0,0)=0.001;

}
 //printf("t\n");
 //cout<<t<<endl;
 //printf("st\n");
 //cout<<st<<endl;
matrix <double> temp[num_policy];
//for(int i=0;i<num_policy;i++){
//    temp[i]=(st-c(i,0)*ones_matrix<double>(1,terminal_step));
//    temp[i](0,i)=temp[i](0,i)*temp[i](0,i);
//
//}
//printf("test\n");

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
//printf("s3");
 //printf("w\n");
 //cout << w<<endl;
//printf("s3");
 //GPGPU

 for(int i=0;i<terminal_step;i++){
        if(sum(colm(w,i))==0){
    printf("w_eroor");

}
    set_colm(G,i)=colm(w,i)*st(0,i)/sum(colm(w,i));
 }


 //printf("w\n");
   // cout << w<<endl;
 //printf("G\n");
 //cout<<G<<endl;

//printf("s4");

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

void excute_policy(int k,matrix <double> thetag,matrix<double> noise){
   // printf("thetat\n");
   typedef matrix <double> matrix_exp_type;
matrix <double> theta_trans(1,num);
matrix <double> theta1;

///ロバスト化関連///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
theta_trans=rowm(policy,tra);
if(add_model_noise==1){
    printf("%d(Pla_friction=%.32f)\n",tra+1,Pla_friction);
}
else{
    printf("%d(Pla_friction=%.32f)\n",tra+1,Pla_friction);
}
///end///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

thetag=trans(theta_trans);
theta1=thetag*ones_matrix<double>(1,terminal_step);
//theta_k[k]=thetat;
//cout<<thetat<<endl;
    double parnum_policy =num_policy;
    double parnum_DOFs=DOF_number;
    double parnum_base=num_base;
    double partau=tau;
    matrix <double> parstart_state;
   // parstart_state=colm(trajectory[0],0);
    matrix <double> dmp(DOF_number*2,1);
    matrix <double> DMP(DOF_number*2,terminal_step);
    matrix <double> parf(DOF_number,1);
    matrix <double> f(DOF_number,terminal_step);
    matrix <double> parGs;
    matrix <double> parthetat;
    matrix <double> k1;
    matrix <double> k2;
    matrix <double> k3;
    matrix <double> k4;
    matrix <double> ddmp;
    double dt=step_size;

   // cout<<Gs<<endl;
    //初期値
    for(int D=0;D<DOF_number;D++){
        set_rowm(dmp,2*D)=0;
        set_rowm(dmp,2*D+1)=0;
    }
    //printf("dmp\n");
    //cout<<dmp<<endl;
    //DMP=dmp;
    //非線形近似f
    //GPGPU
    for (int i=0;i<terminal_step;i++){
            parGs=colm(Gs,i);
            parthetat=colm(theta1,i);

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

        make_dmp(dmp,temp1,temp2);
        k1=dt_test;

        //printf("k1\n");
        //cout<<k1<<endl;
        //cout<<make_dmp(dmp,colm(gs,i),colm(f,i))<<endl;

        make_dmp(dmp+k1*dt/2,temp1,temp2);
        k2=dt_test;

        make_dmp(dmp+k2*dt/2,temp1,temp2);
        k3=dt_test;

        make_dmp(dmp+k3*dt,temp1,temp2);
        k4=dt_test;

        ddmp = (k1 + 2*k2 + 2*k3 + k4)*dt/6;
        //printf("ddmp\n");
        //cout<<ddmp<<endl;

        dmp+=ddmp;
        set_colm(DMP,i)=dmp;
    }
   // printf("DMP\n");
    //cout<<DMP<<endl;


    for (int D=0;D<DOF_number;D++){
        set_rowm(psi[k],D)=rowm(DMP,2*D);
    }

    //printf("psi[%0]\n",k);
    //cout<<psi[0]<<endl;

if(off_noise==1){
//printf("pso[%d]\n",k);

//cout<<psi[k]<<endl;

}
else{

}
  for(int t=0;t<terminal_step;t++){
             for(int l=0;l<leg_max;l++){
                for(int j=0;j<3;j++){
                    double read;
                    read=psi[0](3*l+j,t);

                }
            }
          }
          matrix <double> trans_psi(terminal_step,num);


}


void reward_func(){
const dReal *leg_pos[3];
const dReal *base_posi,*Al_posi[leg_number],*Motor_posi[leg_number][3],*Pla_posi[leg_number][3];
double center_leg[2];
double center_G[2];
double temp_x=0,temp_y=0,temp_z=0;
double total_m;
for(int i=0;i<leg_number;i++){
    leg_pos[i]=dBodyGetPosition(Al[0][i][1].body);
}

    for(int j=0;j<leg_number;j++){
        temp_x+=leg_pos[j][0];
        temp_y+=leg_pos[j][1];

    }
center_leg[0]=temp_x/leg_number;
center_leg[1]=temp_y/leg_number;

for(int i=0;i<leg_number;i++){
    for(int j=0;j<3;j++){
    Motor_posi[i][j]=dBodyGetPosition(Motor[0][i][j].body);
    Pla_posi[i][j]=dBodyGetPosition(Pla[0][i][j].body);
    }
    Al_posi[i]=dBodyGetPosition(Al[0][i][0].body);
}
base_posi=dBodyGetPosition(base[0][0].body);
for(int i=0;i<leg_number;i++){
    for(int j=0;j<3;j++){
        center_G[0]+=Motor_posi[i][j][0]*Motor_m+Pla_posi[i][j][0]*Pla_m;
         center_G[1]+=Motor_posi[i][j][1]*Motor_m+Pla_posi[i][j][1]*Pla_m;
         total_m=+Motor_m+Pla_m;
    }
    center_G[0]+=Al_posi[i][0]*Al_m[0]+leg_pos[i][0]*Al_m[1];
     center_G[1]+=Al_posi[i][1]*Al_m[0]+leg_pos[i][1]*Al_m[1];
     total_m+=Al_m[0]+Al_m[1];

}
center_G[0]/=total_m;
center_G[1]/=total_m;
//printf("cg-x\n");
//cout<<center_G[0]<<endl;
//printf("leg-center-x\n");
//cout<<center_leg[0]<<endl;
double tt;
tt=center_G[0];
//printf("tt\n");
//cout<<tt<<endl;
RR=tt;


}

void destroyMonoBot()
{//ボディの破壊

  for (int i = 0; i < leg_max; i++) {
   for(int j=0;j<3;j++){
    dBodyDestroy(Motor[0][i][j].body);
    dGeomDestroy(Motor[0][i][j].geom);
    dBodyDestroy(Pla[0][i][j].body);
    dGeomDestroy(Pla[0][i][j].geom);
   }//printf("haaaaaaaa");
   for (int j=0;j<1;j++){
    dBodyDestroy(Al[0][i][j].body);
    dGeomDestroy(Al[0][i][j].geom);
   }
  }
  for(int i=0;i<num_base;i++){
    dBodyDestroy(base[0][i].body);
    dGeomDestroy(base[0][i].geom);
  }
//ジョイントの破壊
for (int i=0;i<leg_max;i++){
    for (int j=0;j<3;j++){
        dJointDestroy(Joint_Motor[0][i][j]);
        dJointDestroy(Joint_Pla[0][i][j]);
    }

    dJointDestroy(Joint_fix[0][i]);
}
//dJointDestroy(Joint_ff);

}
void control()
{//printf("temp_p[3][0]=%f\n",temp_p[3][0]);
//printf("original_d=%d\n",d);//printf("Q=%f\n",1/Q1);
dReal  fMax = 1.0,KD=0.01;
 double angle_r[leg_max][3];
double umax,umin;
umax=6.28,umin=-6.28;
restart_judge=0;
//double angle[1000][leg_max][3];
if(STEPS>=100){
for(int i=0;i<leg_max;i++){
    for(int j=0;j<3;j++){
//            if(SSS!=100){
//for(int i=1;i<=k;i++){
//    for(int j=0;j<3;j++){
//        ut1[i][j]=-5;
//        u[d][i][j]=-5;
//    }
//}SSS=100;
//}
//if(i==1||i==4){ut1[i][1]=1;ut1[i][2]=1;}
//printf("tempu[%d][%d]=%f\n",i,j,ut1[i][j]);

//printf("ut1[%d][%d]=%f\n",i,j,ut1[i][j]);
//angle_r[i][j]=dJointGetHingeAngle(Joint_Motor[0][i][j]);
//if(psi[0](3*i+j,STEPS)>umax){
//    dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, umax);
//}else if(psi[0](3*i+j,STEPS)<umin){
//    dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel,umin);
//}
//else{
//    dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, psi[0](3*i+j,STEPS-100));
//}

double JA;
//JA=(psi[0](3*i+j,STEPS-100-(STEPS%5)+5)-angle[0][i][j])/(step_size*5);


if(STEPS%1==0){
angle[0][i][j]=dJointGetHingeAngle(Joint_Motor[0][i][j]);
JA=-(angle[0][i][j]-psi[0](3*i+j,STEPS+1-(STEPS%1)-100))/(step_size);
real_vel[0](3*i+j,STEPS-100)=JA;
// if(j==0){
//         dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, 1);
//
//     }else{
//      dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, 0);
//     }
    dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, JA);
       switch (j){

    }
}else{
     JA=-(angle[0][i][j]-psi[0](3*i+j,STEPS+1-(STEPS%1)-100))/(step_size);
     real_vel[0](3*i+j,STEPS-100)=JA;
//     if(j==0){
//         dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, 1);
//
//     }else{
//      dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, 0);
//     }
 //dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, JA);
}

//dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, psi[0](3*i+j,STEPS));
///JA=(psi[0](3*i+j,STEPS-100-(STEPS%5)+5)-old_angle[i][j])/(step_size*5);
 dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, JA);
dJointSetHingeParam(Joint_Motor[0][i][j], dParamFMax, fMax);
//old_temp_u[i][j]=ut1[i][j];
//old_temp_p[i][j]=temp_pt[i][j];
//old_temp_v[i][j]=temp_vt[i][j];
      }
    }
    if((STEPS-100)/5>=80){
        for(int i=0;i<leg_max;i++){


        }
//dCloseODE();
    }
}else{

for(int i=0;i<leg_max;i++){
    for(int j=0;j<3;j++){
//            if(SSS!=100){
//for(int i=1;i<=k;i++){
//    for(int j=0;j<3;j++){
//        ut1[i][j]=-5;
//        u[d][i][j]=-5;
//    }
//}SSS=100;
//}
//if(i==1||i==4){ut1[i][1]=1;ut1[i][2]=1;}
//printf("tempu[%d][%d]=%f\n",i,j,ut1[i][j]);

//printf("ut1[%d][%d]=%f\n",i,j,ut1[i][j]);
angle_r[i][j]=dJointGetHingeAngle(Joint_Motor[0][i][j]);
dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel,-angle_r[i][j]/step_size);

dJointSetHingeParam(Joint_Motor[0][i][j], dParamFMax, fMax);
//old_temp_u[i][j]=ut1[i][j];
//old_temp_p[i][j]=temp_pt[i][j];
//old_temp_v[i][j]=temp_vt[i][j];
      }
    }
}
    ///
}
// シミュレーションの再スタート
void restart()
{
  STEPS    = -1;      // ステップ数の初期化
leg_number=0;
//episode++;
//weight_vector_b+=b_vary;
memo_num=1;
 if(judge_number==10000){
    if(initial_set==10000){
        initial_set=20000;
    }else{
     K_max++;
      //printf("k_max_temp=%d\n",K_max);
    }

        judge_number=0;


    }
    else if(judge_number==0){
      ;
        d=1;

        initial_set=0;
        K_max=1;
        ddeltat=0;
        temp_K=zeros_matrix<double>(division,100);
        K_nolm_m=zeros_matrix<double>(1,100);
    }
  destroyMonoBot();  // ロボットの破壊

  dJointGroupDestroy(contactgroup[0]);     // ジョイントグループの破壊
  contactgroup[0] = dJointGroupCreate(0);  // ジョイントグループの生成
  createMonoBot(0);

  for (int i=0;i<leg_max;i++){
  createleg(0);

  }

//  Joint_ff=dJointCreateFixed(world,0);
//  dJointAttach(Joint_ff,Al[2][0].body,base[1].body);
  //dJointSetFixed(Joint_ff);
   old_posi=0;
   restart_judge=0;                  // ロボットの生成
}


// シミュレーションループ
static void simLoop(int pause)
{

    int JAS;
    JAS=0;
    if(STEPS>terminal_step+100-1){
        JAS=0;
        restart();
        excute_policy(0,theta,zeros_matrix<double>(num,terminal_step));
    }
    else{
        if(stop_mode && (STEPS%100==0 || STEPS==terminal_step+99)){
            pause=1;
            if(key_s){
                pause=0;
                key_s=0;
            }
        }
      if (!pause) {
        const dReal *pos;
        double x,y;
        pos=dBodyGetPosition(base[0][0].body);
        x=pos[0];
        y=pos[1];


        if((STEPS-99)%1600==0){

            count_p++;
            if(count_p<0){
                double angle_r[leg_max][3];
                JAS=1;

                for(int i=0;i<leg_max;i++){
                    for(int j=0;j<3;j++){

                        angle_r[i][j]=dJointGetHingeAngle(Joint_Motor[0][i][j]);

                        dJointSetHingeParam(Joint_Motor[0][i][j],  dParamVel, -angle_r[i][j]/step_size);


                        dJointSetHingeParam(Joint_Motor[0][i][j], dParamFMax, 2);

                    }
                }
            }
            else{
                JAS=0;
                count_p=0;
                STEPS++;
            }
        }
        else{
            JAS=0;
            STEPS++;
        }

        const dReal *Rotation_base[2];
        double R1[4][10],R2[4][10];
        double vector_c[4],h_g[4],h_g_2[4];
        double inner_nolm[2];
        double mp;
        double vector_m[4];
        for(int i=0;i<1;i++){
            Rotation_base[i]=dBodyGetRotation(base[0][i].body);
        }
        for(int i=1;i<=3;i++){
            if(i!=3){h_g[i]=0;}else {h_g[i]=-1;}
            if(i!=2){h_g_2[i]=0;}else{h_g_2[i]=1;}
                vector_c[i]=0;
                vector_m[i]=0;
        }
        for(int i=0;i<3;i++){
            R1[1][i+1]=Rotation_base[0][i];
            R1[2][i+1]=Rotation_base[0][i+4];
            R1[3][i+1]=Rotation_base[0][i+8];
        }
        for(int l=1;l<=3;l++){
            for(int j=1;j<=3;j++){
                vector_c[l]+=R1[l][j]*h_g[j];
                vector_m[l]+=R1[l][j]*h_g_2[j];
            }
        }

        double yaw;
        yaw=atan2(R1[2][1],R1[1][1])/M_PI*180;
        //printf("pitch=%f\n",yaw);
        double roll;
        roll=atan2(R1[3][1],R1[3][2]);
        roll=180*roll/M_PI;

        if(STEPS>=100){
            for(int i=0;i<leg_max;i++){
                for(int j=0;j<3;j++){
                    double read1,read2;
                    read1=dJointGetHingeAngleRate(Joint_Motor[0][i][j]);
                    read2=dJointGetHingeAngle(Joint_Motor[0][i][j]);
                }
            }
        }

        if(JAS==1){

        }
        else{
            control();
        }



        dSpaceCollide(space[0],0,&nearCallback);
        dWorldStep(world[0],0.01);
        dJointGroupEmpty(contactgroup[0]);
      }
      drawMonoBot(); // ロボットの描画
    }
}

// ロボットの生成

// ロボットの破壊
static void start()
{
  static float xyz[3] = {   0.8, 0.0, 0.5};
  static float hpr[3] = {-180, -30 , 0.0};
  float xyz0[3] = {0.8,0.0,0.5};
  for(int i=0;i<3;i++){
    xyz[i]=xyz0[i]+zoom*xyz0[i];
  }
  xyz[1]+=dy;
  dsSetViewpoint(xyz,hpr);               // 視点，視線の設定
  dsSetSphereQuality(3);                 // 球の品質設定
}

// キー操作
static void command(int cmd)
{
 switch (cmd) {
///ロバスト化関連//////////////////////////////////////////////////////////////////////////////////////////////////////////
    case 'n':
       if(tra!=max_trial_num-1){
            tra++;
       }
       break;

    case 'b':
       if(tra!=0){
            tra--;
       }
       break;
    case 'a':
        if(add_model_noise==0){
            Pla_friction+=noise_ratio*Pla_friction;
            add_model_noise=1;
        }
        break;
    case 'd':
        if(add_model_noise==1){
            Pla_friction=0.5;
            add_model_noise=0;
        }
        break;

    case 's':
        if(stop_mode){
            key_s=1;
        }
        break;

    case 'm':
        stop_mode=!stop_mode;
        break;

    case 'z':
        zoom-=0.1;
        start();
        break;

    case 'x':
        zoom+=0.1;
        start();
        break;

    case 'h':
        dy+=0.1;
        start();
        break;

    case 'g':
        dy-=0.1;
        start();
        break;


///end////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   case 'r':restart()                           ; break;
   default :printf("key missed \n")             ; break;
 }
}



void setDrawStuff()           /*** 描画関数の設定 ***/
{
  fn.version = DS_VERSION;    // ドロースタッフのバージョン
  fn.start   = &start;        // 前処理 start関数のポインタ
  fn.step    = &simLoop;      // simLoop関数のポインタ
  fn.command = &command;      // キー入力関数へのポインタ
  fn.path_to_textures = "../../drawstuff/textures"; // テクスチャ
}


int main (int argc, char *argv[])
{

time(&timer1);
umax=10;
learning_rate=0.1;
discount_rate=0.3;
///寸法パラメータ/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
///end///////////////////////////////////////////////////////////////////////////////////////////////



  setDrawStuff();
    dInitODE();
///ロバスト化関連////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 inputfile1=fopen("policy.txt","r");
if(inputfile1==NULL){
    printf("file open error");
 }
 double data;
 for(int i=0;i<max_trial_num;i++){
    for(int j=0;j<num;j++){
        fscanf(inputfile1,"%lf,",&data);
        policy(i,j)=data;
    }
 }
 fclose(inputfile1);
///end/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  world[0]        = dWorldCreate();
  space[0]        = dHashSpaceCreate(0);
  contactgroup[0] = dJointGroupCreate(0);

  dWorldSetGravity(world[0], 0,0, -grav);
  dWorldSetERP(world[0], 0.2);          // ERPの設定
  dWorldSetCFM(world[0], 1e-4);         // CFMの設定
  ground[0] = dCreatePlane(space[0], 0, 0, 1, 0);
leg_number=0;
 createMonoBot(0);
  for (int i=0;i<leg_max;i++){
  createleg(0);

  //初期設定
  //if(i%3==2){Motor_y[0]=Motor_y[0]+0.08;Pla_y[0]=Motor_y[0];Motor_y[1]=Pla_y[0]; Pla_y[1]=Motor_y[1]; Motor_y[2]=Pla_y[1];Pla_y[2]=Motor_y[2]; Al_y[0]=Pla_y[2];Al_y[1]=Al_y[0];}
  }
    E=identity_matrix<double>(num);
    S=W_R*E;
    sigma  = W_sigma *E;
    theta=zeros_matrix<double> (num,1);
    double dt;
    double phi_T;
    dt=step_size;
    tf=dt*terminal_step;
    phi_T=tf/3;
    tau=phi_T/(2*M_PI);
    //printf("tau\n");
    //cout<<tau<<endl;

for(int i=0;i<terminal_step;i++){
    t(0,i)=i;

}
printf("E1");
    //DMPパラメータ
    //phi_T = tf/6;
    alphaq = 25;
   // outputfile1=fopen("cost.txt","w");

    //betaq = alphaq/4;
    //tau = phi_T /(2*pi);

    //DMP線形基底関数の設定
    //f=1;
    r0 = 25*25/4; // 初期振幅
    rg = 25*25/4; // 目標振幅
  //  printf("mark_pre1\n");


      //  printf("mark1\n");
     //   cout <<rr0<<endl;

     //   cout <<rrg<<endl;
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

   // printf("Gs\n");
   // cout<<Gs<<endl;

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
    //printf("gs\n");
    //cout<<gs<<endl;

//  for(int i=1;i<=10;i++){
//  epst[i]=zeros_matrix<double>(10,10);
//  meanzerot=zeros_matrix<double>(10,1);
//  }
    WN2=0;
    printf("E3");
    for(int i=0;i<terminal_step;i++){
        set_colm(t,i)=i;
    }
//WN1=zeros_matrix<double>(1,terminal_step);
//printf("qqqqqqqqq");
    for(int i=0;i<terminal_step;i++){
        WN2(0,i)=(terminal_step-i-1);
        WN1(i,i)=terminal_step-(i+1);
    }
    //printf("WN2\n");

    //cout<<WN2<<endl;

    //printf("HHHHHHHHH");
    //重み行列の作成
        //omploop


        //printf("S\n");
        //cout<<S<<endl;
        printf("3.5");
        ///初期状態のMでエラーが発生
        #pragma omp parallel for
        for(int i=0;i<terminal_step;i++){
                //printf("Gs\n");
                //cout<<Gs<<endl;
                matrix <double> M_test,M_test2;
        double test;
        //printf("eroor\n");
        //cout<<inv(S)<<endl;

                M_test=trans(colm(Gs,i))*inv(S)*colm(Gs,i);
               // printf("M_test");
                //cout<<M_test<<endl;
        M_test2=inv(M_test);
        test=M_test2(0,0);

            M[i]=(inv(S)*colm(Gs,i)*trans(colm(Gs,i)))*test;
            //printf("M[%d]\n",i);
            //cout<<M[i]<<endl;

           // printf("M[%d]\n",i);
            //cout<<M[i]<<endl;
        }
        printf("E4");

    theta=zeros_matrix<double>(num,1);
    //多変量正規分布の作成 clear
     meanzerot=zeros_matrix<double>(num,1);
     printf("E5");
//    for(int k=0;k<K_limit;k++){
//            //printf("A");
// //eps[k]         = zeros_matrix<double>(DOF_number,terminal_step);
////printf("E6");
// //多変量正規分布の乱数ノイズの生成
// mvnrnd(meanzerot,100000000*E);
// eps[k]=t_epst;
// //printf("eps[%d]\n",k);
//  //cout <<  eps[8]<< endl;
// //printf("exe_theta[%d]\n",k);
// //printf("E3");
////excute_policy(k,theta,eps[k]);
////for(int i=0;i<terminal_step;i++){
////    set_colm(temp_exetheta[k],i)=theta;
////}
//// exe_theta[k]=temp_exetheta[k]+eps[k];
//// //cout << exe_theta[9]<< endl;
// //cout <<psi[k] <<endl;
//}
//#pragma omp parallel for
for(int k=0;k<K_limit;k++){
   // excute_policy(k,theta,eps[k]);
}

  ///
printf("ww");
printf("psi\n");
//cout<<psi[0]<<endl;
 excute_policy(0,theta,zeros_matrix<double>(num,terminal_step));
 if(JH==0){ for(int t=0;t<1200;t++){
        for(int i=0;i<leg_max;i++){
            for(int j=0;j<3;j++){
                if(j==0){
            fprintf(outputfile15,"ang_1[%d]=%f,",t,psi[0](3*i+j,t));
        }else if(j==1){
         fprintf(outputfile16,"ang_2[%d]=%f,",t,psi[0](3*i+j,t));
        }else if(j==2){
         fprintf(outputfile17,"ang_3[%d]=%f,",t,psi[0](3*i+j,t));
        }
            }
        }


 }
 JH=100000;
 }
 for(int t=0;t<terminal_step;t++){
    for(int i=0;i<leg_max;i++){
        for(int j=0;j<3;j++){

        }
    }

 }

 //fprintf(outputfile14,"\n");
  dsSimulationLoop (argc, argv, 640, 480, &fn);
//  while(1){
//
//    simLoop(0);
//  }
  dWorldDestroy (world[0]);
  dCloseODE();

  return 0;
}
